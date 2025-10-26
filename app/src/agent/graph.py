"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import base64
import os

# from langgraph.prebuilt import ToolNode # Not used
import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.state import InputState, State
from agent.tools import TOOLS
from agent.utils import load_chat_model

load_dotenv()

conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn=conn)

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

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                ),
            ]
        }

    return {"messages": [response]}


async def custom_tool_node(state: State) -> dict[str, Any]:
    """
    Custom node to execute tools. It modifies file paths to include the user's
    email as a directory.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    user_email = state.email or "unknown_user"
    # All user-specific files will be saved under 'static/<email>/'
    base_path = os.path.join("static", user_email)
    os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists

    tool_messages = []

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
            # --- Modified logic to handle all file paths ---
            if tool_name == "create_image":
                image_num = args.get("image_number", 1)
                # Ensure output path is in the user's directory
                args["output_path"] = os.path.join(base_path, f"design-{image_num}.png")

            if tool_name == "convert_black_to_transparent":
                # The AI provides relative filenames (e.g., "design-2.png")
                # We prepend the user's path to make them correct
                args["image_path"] = os.path.join(base_path, args["image_path"])
                args["output_path"] = os.path.join(base_path, args["output_path"])

            # Execute the tool with modified arguments
            tool_output = await tool_to_run.ainvoke(args)
            tool_messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"Error executing tool {tool_name}: {e}",
                    tool_call_id=tool_call["id"],
                )
            )
            print(f"Error executing tool {tool_name}: {e}")  # For server logs

    return {"messages": [*state.messages, *tool_messages]}


def send_artifact_to_frontend(state: State) -> dict[str, Any]:
    """
    Checks the last message for an image tool output and, if found,
    opens the image file, encodes it, and creates a display "artifact".
    """
    last_message = state.messages[-1]

    # Check if the last message is the result of the create_image tool
    if isinstance(last_message, ToolMessage) and last_message.name == "create_image":
        image_path = last_message.content

        try:
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

            image_artifact = {
                "type": "image",
                "b64": image_b64,  # Using 'b64' key
            }
            return {"artifacts": [*state.artifacts, image_artifact]}
        except FileNotFoundError:
            print(f"Error: Could not find image file at path: {image_path}")
            pass
        except Exception as e:
            print(f"An error occurred while creating image artifact: {e}")
            pass

    return {}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determines the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END
    return "tools"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node("call_model", call_model)
builder.add_node("tools", custom_tool_node)
builder.add_node("send_artifact_to_frontend", send_artifact_to_frontend)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "send_artifact_to_frontend")
builder.add_edge("send_artifact_to_frontend", "call_model")

# --- Compile with memory ---
graph = builder.compile(
    name="Ownit",
    checkpointer=memory,
)
