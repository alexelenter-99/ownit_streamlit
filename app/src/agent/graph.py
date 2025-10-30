"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import base64
import logging
import os
import re  # <-- Import regex
import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.state import InputState, State
from agent.tools import TOOLS, upload_to_gcs
from agent.utils import load_chat_model

# logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn=conn)
# memory = AsyncSqliteSaver.from_conn_string("memory.sqlite")
memory = InMemorySaver()


configuration = Configuration.from_context()
model = load_chat_model(configuration.model).bind_tools(TOOLS)

async def call_model(state: State) -> dict[str, Any]:
    """
    Call the LLM. It also checks for and processes image tool outputs before
    calling the model and attaches the image to the final response.
    """
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # --- MODIFICATION: Extract <Plan> into response_metadata ---
    user_facing_content = response.content
    internal_plan = None

    # Start with existing metadata, or an empty dict
    new_response_metadata = response.response_metadata or {}

    if isinstance(user_facing_content, str):
        # Use regex to find and extract the plan
        # re.DOTALL makes '.' match newline characters.
        match = re.search(r"<Plan>(.*?)</Plan>(.*)", user_facing_content, re.DOTALL)
        if match:
            internal_plan = match.group(1).strip()
            user_facing_content = match.group(2).strip()
            # Add the extracted plan to the metadata
            new_response_metadata["internal_plan"] = internal_plan

    # Create a new AIMessage with the cleaned content and the new metadata
    cleaned_response = AIMessage(
        content=user_facing_content,
        id=response.id,
        tool_calls=response.tool_calls,
        # Pass through other attributes
        name=getattr(response, "name", None),
        invalid_tool_calls=getattr(response, "invalid_tool_calls", None),
        tool_call_chunks=getattr(response, "tool_call_chunks", None),
        usage_metadata=getattr(response, "usage_metadata", None),
        # Set the new metadata
        response_metadata=new_response_metadata,
    )
    # --- END MODIFICATION ---

    # Handle the case when it's the last step and the model still wants to use a tool
    # Use the 'cleaned_response' for this check
    if state.is_last_step and cleaned_response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=cleaned_response.id,  # Use ID from cleaned response
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                ),
            ]
        }

    return {"messages": [cleaned_response]}  # Return the message with metadata


async def custom_tool_node(state: State) -> dict[str, Any]:
    """
    Custom node to execute tools. It modifies file paths to include the user's
    email as a directory.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    user_email = state.email or "unknown_user"
    # All user-specific files will be saved under 'images/<email>/'
    base_path = os.path.join("images", user_email)  # <-- Using images/ dir
    os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists

    tool_messages = []
    new_artifacts = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]

        # Find the corresponding tool function
        tool_to_run = next((t for t in TOOLS if t.name == tool_name), None)
        if not tool_to_run:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        try:            
            if tool_name == "create_image":
                image_num = args.get("image_number", 1)
                # Ensure output path is in the user's directory
                args["output_path"] = os.path.join(base_path, f"design-{image_num}.png")

            if tool_name == "convert_black_to_transparent":
                img_name = os.path.basename(args["image_path"])
                out_name = os.path.basename(args["output_path"])
                args["image_path"] = os.path.join(base_path, img_name)
                args["output_path"] = os.path.join(base_path, out_name)

            # Execute the tool with modified arguments
            tool_output = await tool_to_run.ainvoke(args)
            tool_messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

            if (
                tool_name in ("create_image", "convert_black_to_transparent")
            ):
                logging.info(f"Tool {tool_name} finished, output to local path: {tool_output}")
                
                # create artifact for frontend display
                try:
                    with open(tool_output, "rb") as image_file:
                        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
                    new_artifacts.append({"type": "image", "b64": image_b64}) 
                    logging.info(f"Created artifact for {tool_output}")
                except Exception as e:
                    logging.error(f"Failed to create artifact from {tool_output}: {e}")                
                
                # upload to google bucket
                try:
                    public_url = upload_to_gcs(tool_output, user_email)
                    if public_url:
                        logging.info(f"Successfully uploaded to GCS: {public_url}")
                        tool_output = public_url
                    else:
                        logging.error("Failed to get public URL from GCS.")
                except Exception as e:
                    logging.error(f"Failed to upload {tool_output} to GCS: {e}")
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"Error executing tool {tool_name}: {e}",
                    tool_call_id=tool_call["id"],
                )
            )
            logging.error(f"Error executing tool {tool_name}: {e}")  # For server logs

    return {
        "messages": [*state.messages, *tool_messages],
        "artifacts": new_artifacts
    }


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determines the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END # type: ignore allow END constant
    return "tools"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node("call_model", call_model)
builder.add_node("tools", custom_tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

# --- Compile with memory ---
graph = builder.compile(
    name="Ownit",
    checkpointer=memory,
)
