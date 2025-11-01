# pyright: reportAttributeAccessIssue=false
"""
Streamlit chatbot interface for the Ownit agent.

This app provides a user-friendly web interface to interact with the LangGraph-based agent.
"""

import asyncio
import base64
import uuid  # <-- Import UUID for thread IDs
from typing import Any, Optional

import streamlit as st
from agent.configuration import Configuration
from agent.graph import graph as agent_graph
from agent.state import InputState
from langchain_core.messages import AIMessage, HumanMessage

# Configure Streamlit page
st.set_page_config(
    page_title="Ownit Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (no changes)
st.markdown(
    """
<style>
    /* ... [your existing CSS] ... */
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "artifacts" not in st.session_state:
    st.session_state.artifacts = []

# --- Add thread_id for memory ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def display_artifact(artifact: dict[str, Any], index: int | None = None) -> None:
    """Display an artifact in the Streamlit interface."""
    with st.container():
        st.markdown('<div class="artifact-container">', unsafe_allow_html=True)

        artifact_type = artifact.get("type", "unknown")

        if artifact_type == "image":
            image_data = artifact.get("b64")
            if image_data:
                try:
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",")[1]

                    image_bytes = base64.b64decode(image_data)
                    st.image(image_bytes)
                except Exception as e:
                    st.error(f"Error displaying image: {e!s}")

        elif artifact_type == "text":
            content = artifact.get("data", artifact.get("content", ""))
            st.text_area("Content:", value=content, height=200, disabled=True)

        else:
            st.json(artifact)

        st.markdown("</div>", unsafe_allow_html=True)


async def run_agent(email: str | None = None) -> dict[str, Any]:
    """
    Run the agent with the *entire* current session state.
    The 'user_input' is already in st.session_state.messages.
    """
    try:
        # Create input state from the full message history
        input_state = InputState(messages=st.session_state.messages, email=email)

        # Configure the agent with the thread_id
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,  # <-- Pass the thread_id
                "system_prompt": Configuration().system_prompt,
                "model": Configuration().model,
            }
        }

        # Run the agent
        result = await agent_graph.ainvoke(input_state, config=config) # type: ignore config attribute
        return result

    except Exception as e:
        st.error(f"Error running agent: {e!s}")
        return {
            "messages": st.session_state.messages,
            "artifacts": st.session_state.artifacts,
        }


def setup_sidebar() -> str:
    """Setup the sidebar configuration and return user email."""
    with st.sidebar:
        user_email = st.text_input(
            "Email:",  # <-- Made label shorter
            placeholder="your.email@example.com",
            help="Your email is required to start the chat.",
        )

        if st.button("🗑️ Borrars Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.artifacts = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")
        st.subheader("📦 Imagenes Generadas")
        st.subheader("Tienes un limite de 3")
        st.markdown(f"**Total:** {len(st.session_state.artifacts)}")

        if st.session_state.artifacts:
            for index, artifact in enumerate(st.session_state.artifacts):
                if index > 0:
                    st.markdown("---")
                if index == 0:
                    st.markdown("🥇 **Primera Imagen**")
                elif index == 1:
                    st.markdown("🥈 **Segunda Imagen**")
                elif index == 2:
                    st.markdown("🥉 **Tercera Imagen**")
                display_artifact(artifact, index)
        else:
            st.caption("No se generaron imagenes todavia.")

        st.markdown("---")

    return user_email


def display_chat_history() -> None:
    """Display chat messages from session state."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                if message.content:
                    st.write(message.content)
                # if hasattr(message, "response_metadata") and message.response_metadata: TODO: uncomment if want to show
                #     internal_plan = message.response_metadata.get("internal_plan")
                #     if internal_plan:
                #         with st.expander("🤔 Internal Plan"):
                #             st.markdown(internal_plan)

                # if hasattr(message, "tool_calls") and message.tool_calls:
                #     with st.expander("🔧 Tool Calls"):
                #         for tool_call in message.tool_calls:
                #             st.json(
                #                 {
                #                     "name": tool_call.get("name", "unknown"),
                #                     "args": tool_call.get("args", {}),
                #                 }
                #             )


def display_artifacts_section() -> None:
    """Display generated artifacts section."""
    pass


def handle_chat_input(user_email: str) -> None:
    """Handle chat input and agent processing."""
    if prompt := st.chat_input("Ask me anything..."):
        
        if not user_email:
            st.error("Please enter your email in the sidebar to begin.")
            st.stop()
        
        # Add user message to session state
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Show thinking indicator and process
        with st.chat_message("assistant"), st.spinner("🤔 Thinking..."):
            try:
                # Run agent with the full session state
                result = asyncio.run(run_agent(user_email))
                # Update session state with the agent's full history
                process_agent_result(result)

            except Exception as e:
                st.error(f"❌ Error: {e!s}")
                st.session_state.messages.append(
                    AIMessage(content=f"I encountered an error: {e!s}")
                )

        # Rerun to display the new state (AI message and artifacts)
        st.rerun()


def process_agent_result(result: dict[str, Any]) -> None:
    """Process the agent result and update session state."""
    # --- Replace local state with the full state from the agent's memory ---
    st.session_state.messages = result.get("messages", [])
    new_artifacts = result.get("artifacts", [])
    current_count = len(st.session_state.artifacts)
    result_count = len(new_artifacts)
    if result_count > current_count:
        truly_new_artifacts = new_artifacts[current_count:]
        st.session_state.artifacts.extend(truly_new_artifacts)


def main():
    """Main Streamlit application."""
    st.title("🤖 Ownit Agent Chatbot")
    st.markdown("Chat with your AI agent powered by LangGraph")

    user_email = setup_sidebar()

    display_chat_history()

    handle_chat_input(user_email)


if __name__ == "__main__":
    main()
