#streamlit_ui.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# Import the compiled app from react agent
from react_agent import app as react_app

# Page configuration
st.set_page_config(
    page_title="ReAct Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .tool-badge {
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-right: 5px;
        display: inline-block;
    }
    .tool-get_weather { background-color: #2196F3; }
    .tool-calculate_math { background-color: #9C27B0; }
    .tool-book_hotel { background-color: #FF9800; }
    .tool-book_flight { background-color: #00BCD4; }
    .tool-web_search { background-color: #4CAF50; }
    .tool-default { background-color: #607D8B; }
    .stChatMessage {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Tool icon mapping
TOOL_ICONS = {
    "get_weather": "🌤️",
    "calculate_math": "🧮",
    "book_hotel": "🏨",
    "book_flight": "✈️",
    "web_search": "🔍"
}

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 ReAct Agent")
    st.markdown("---")

    # Login / Thread ID for memory persistence
    st.subheader("👤 User Session")
    user_id = st.text_input("Enter User ID (for memory persistence)", value="user1")
    st.caption(f"Thread ID: `{user_id}`")

    st.markdown("---")

    # Available tools info
    st.subheader("🛠️ Available Tools")
    st.markdown("""
    - 🌤️ **Weather** - Real-time weather & 3-day forecast
      *(python_weather API)*
    - 🧮 **Math** - Calculate expressions using eval()
      *(LLM + Python eval)*
    - 🏨 **Hotel Booking** - Book hotels
      *(Simulated)*
    - ✈️ **Flight Booking** - Book flights
      *(Simulated)*
    - 🔍 **Web Search** - Search the web
      *(Tavily API + RAG)*
    """)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # New conversation button
    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph + OpenAI + python_weather")

# Main chat interface
st.title("💬 Multi-Agent Chatbot")
st.caption("Ask me about weather, math, bookings, or anything else!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to create tool badge HTML
def get_tool_badge(tool_name):
    icon = TOOL_ICONS.get(tool_name, "🔧")
    css_class = f"tool-{tool_name}" if tool_name in TOOL_ICONS else "tool-default"
    return f'<span class="tool-badge {css_class}">{icon} {tool_name}</span>'

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools_used"):
            tools_html = " ".join([get_tool_badge(tool) for tool in message["tools_used"]])
            st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Configure with user's thread_id
                config = {"configurable": {"thread_id": user_id}}

                tools_used = []
                final_response = ""

                # Stream the response
                for event in react_app.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="values"
                ):
                    if event.get("messages"):
                        last_msg = event["messages"][-1]

                        # Track tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            tools_used.extend([tc['name'] for tc in last_msg.tool_calls])

                        # Get final AI response
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                                final_response = last_msg.content

                # Display the response
                st.markdown(final_response)

                # Show tools used with color-coded badges
                if tools_used:
                    unique_tools = list(set(tools_used))
                    tools_html = " ".join([get_tool_badge(tool) for tool in unique_tools])
                    st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tools_used": list(set(tools_used)) if tools_used else None
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
