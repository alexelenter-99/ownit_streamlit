"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import base64
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.prompts import FINISHING_PROMPT
from agent.state import InputState, State
from agent.tools import (
    TOOLS,
    convert_black_to_transparent,  # Helper
    execute_production_file,
    finalize_design,
    upload_to_gcs,  # Helper
)
from agent.utils import load_chat_model

load_dotenv()

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

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    user_facing_content = response.content
    internal_plan = None

    new_response_metadata = response.response_metadata or {}

    if isinstance(user_facing_content, str):
        match = re.search(r"<Plan>(.*?)</Plan>(.*)", user_facing_content, re.DOTALL)
        if match:
            internal_plan = match.group(1).strip()
            user_facing_content = match.group(2).strip()
            new_response_metadata["internal_plan"] = internal_plan

    cleaned_response = AIMessage(
        content=user_facing_content,
        id=response.id,
        tool_calls=response.tool_calls,
        name=getattr(response, "name", None),
        invalid_tool_calls=getattr(response, "invalid_tool_calls", None),
        tool_call_chunks=getattr(response, "tool_call_chunks", None),
        usage_metadata=getattr(response, "usage_metadata", None),
        response_metadata=new_response_metadata,
    )

    if state.is_last_step and cleaned_response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=cleaned_response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                ),
            ]
        }

    return {"messages": [cleaned_response]}


async def call_finishing_model(state: State) -> dict[str, Any]:
    """
    Call the LLM (Finishing state). It uses the simple FINISHING_PROMPT.
    """
    configuration = Configuration.from_context()
    FINISHING_TOOLS = [execute_production_file, finalize_design]
    model = load_chat_model(configuration.model).bind_tools(FINISHING_TOOLS)
    # --- Use the new prompt ---
    system_message = FINISHING_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # --- (This logic is identical to call_model) ---
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

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and cleaned_response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=cleaned_response.id,  # Use ID from cleaned response
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                ),
            ]
        }

    return {"messages": [cleaned_response]}


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
    new_image_count = state.image_count

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]

        if tool_name not in ["create_image", "create_image_prompt"]:
            continue

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
                new_image_count = image_num

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

            if tool_name == "create_image":
                logging.info(f"Tool {tool_name} finished, output to local path: {tool_output}")

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
        "artifacts": state.artifacts + new_artifacts,
        "image_count": new_image_count,
    }


async def production_node(state: State) -> dict[str, Any]:
    """
    Executes the final production logic (convert, upload) after the
    finishing agent calls execute_production_file.
    """
    last_message = state.messages[-1]
    if (
        not isinstance(last_message, AIMessage)
        or not last_message.tool_calls
        or last_message.tool_calls[0].get("name") != "execute_production_file"
    ):
        return {}

    tool_call = last_message.tool_calls[0]
    args = tool_call["args"]
    tool_messages = []
    user_email = state.email or "unknown_user"
    base_path = os.path.join("images", user_email)

    try:
        logging.info(f"Executing production_node logic for tool call: {tool_call}")

        # 1. Get arguments
        design_num = args.get("design_number")
        size = str(args.get("size", "talle-desconocido")).lower()
        prod_type = str(args.get("product_type", "tipo-desconocido")).lower()

        if not design_num:
            raise ValueError(
                "design_number is missing from execute_production_file call"
            )

        # 2. Construct paths
        input_file = f"design-{design_num}.png"
        output_file = f"talle-{size}-{prod_type}.png"
        local_input_path = os.path.join(base_path, input_file)
        local_output_path = os.path.join(base_path, output_file)

        # 3. Call helper: convert_black_to_transparent
        logging.info(f"Converting {local_input_path} to {local_output_path}...")
        convert_black_to_transparent(local_input_path, local_output_path)

        # 4. Call helper: upload_to_gcs
        logging.info(f"Uploading {local_output_path} to GCS...")
        public_url = upload_to_gcs(local_output_path, user_email)

        if not public_url:
            raise Exception("Failed to upload final file to GCS")

        logging.info(f"Final file available at: {public_url}")
        tool_output = public_url

        tool_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"],
                name="execute_production_file",
            )
        )
    except Exception as e:
        logging.error(f"Error in production_node: {e}", exc_info=True)
        tool_messages.append(
            ToolMessage(
                content=f"Error finalizing design: {e}",
                tool_call_id=tool_call["id"],
                name="execute_production_file",
            )
        )

    return {"messages": [*state.messages, *tool_messages], "artifacts": state.artifacts}


def route_entry(state: State) -> Literal["call_finishing_model", "call_model"]:
    """
    Route the user to the correct agent based on the image count
    at the beginning of each new turn.
    """
    if state.image_count >= 3:
        logging.info(f"Image count is {state.image_count}. Routing to finishing model.")
        return "call_finishing_model"

    logging.info(f"Image count is {state.image_count}. Routing to main model.")
    return "call_model"


def route_model_output(
    state: State,
) -> Literal["call_finishing_model", "tools", "__end__"]:
    """Determines the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END  # type: ignore


    if state.image_count >= 3:
        logging.info("Image limit reached. Forcing to finishing model.")
        return "call_finishing_model"
    
    # Check for the "user is happy" signal
    if last_message.tool_calls[0].get("name") == "finalize_design":
        logging.info("User is happy. Routing to finishing model.")
        return "call_finishing_model"

    return "tools"


def route_after_tools(state: State) -> Literal["call_finishing_model", "call_model"]:
    """
    Route to the finishing state or back to the main loop based on image count.
    This router only runs after the 'tools' (iteration) node.
    """
    # 1. If we just hit 3 images, force finalization
    if state.image_count >= 3:
        logging.info("Image count reached 3. Forcing finalization state.")
        return "call_finishing_model"

    # 2. Otherwise, continue the iteration loop
    return "call_model"


def route_finishing_model(
    state: State,
) -> Literal["production_node", "__end__", "call_finishing_model"]:
    """
    Routes the output of the finishing model.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END  # type: ignore

    # If it called the production tool, run it
    if last_message.tool_calls[0].get("name") == "execute_production_file":
        return "production_node"

    # If it called another tool (e.g. user asked a weird question),
    # just loop back to the finishing agent.
    return "call_finishing_model"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

# --- MODIFIED GRAPH DEFINITION ---
builder.add_node("call_model", call_model)
builder.add_node("tools", custom_tool_node)
builder.add_node("call_finishing_model", call_finishing_model)
builder.add_node("production_node", production_node) # <-- ADDED

builder.add_conditional_edges(
    START,
    route_entry,
    {
        "call_finishing_model": "call_finishing_model",
        "call_model": "call_model",
    },
)

# Main iteration loop
builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {
        "tools": "tools",
        "call_finishing_model": "call_finishing_model",
        "__end__": END,
    },
)
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "call_finishing_model": "call_finishing_model",
        "call_model": "call_model",
    },
)

# Finishing loop
builder.add_conditional_edges(
    "call_finishing_model",
    route_finishing_model,
    {
        "production_node": "production_node",
        "call_finishing_model": "call_finishing_model",
        "__end__": END,
    },
)

# End of production
builder.add_edge("production_node", END)
# --- END MODIFIED GRAPH DEFINITION ---


# --- Compile with memory ---
graph = builder.compile(
    name="Ownit",
    checkpointer=memory,
)
