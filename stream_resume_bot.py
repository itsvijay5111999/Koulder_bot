import streamlit as st
from langgraph_chatbot import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

# ======================= Helper Functions =======================

def generate_thread_name_from_message(message):
    """Creates a short thread name from the first user message."""
    return message[:40] + "..." if len(message) > 40 else message

def send_email_with_gmail(subject, body, to_email):
    """Sends an email using credentials from st.secrets and provides feedback."""
    try:
        gmail_user = st.secrets["GMAIL_USER"]
        gmail_password = st.secrets["GMAIL_APP_PASSWORD"]
        
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        
        st.toast(f"Email successfully sent to {to_email}", icon="📧")
    except KeyError:
        st.error("Email feature is not configured. Please add GMAIL_USER and GMAIL_APP_PASSWORD to your Streamlit secrets.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def format_conversation_for_email(messages):
    """Formats the entire chat history into a single string for an email."""
    formatted_lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "")
        
        if msg.get("display_type") != "youtube_videos":
             formatted_lines.append(f"{role}: {content}")

        if msg.get("display_type") == "youtube_videos" and isinstance(content, list):
            formatted_lines.append("\n--- YouTube Results ---")
            for video in content:
                formatted_lines.append(f"Title: {video.get('title')}\nLink: {video.get('link')}\n")
    return "\n\n".join(formatted_lines)

# ======================= Optimized Thread Switching =======================

def switch_thread(thread_id):
    """
    Switches to a new thread. Uses a cache to avoid slow database lookups.
    """
    st.session_state["thread_id"] = thread_id
    st.session_state.editing_thread_id = None
    
    if thread_id in st.session_state.thread_histories:
        st.session_state.messages = st.session_state.thread_histories[thread_id]
        return

    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        db_messages = state.values.get("messages", [])
        
        ui_messages = []
        for msg in db_messages:
            if isinstance(msg, HumanMessage):
                ui_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ToolMessage) and msg.name == "search_youtube_videos":
                try:
                    video_content = json.loads(msg.content)
                    if isinstance(video_content, list):
                        ui_messages.append({
                            "role": "assistant",
                            "content": video_content,
                            "display_type": "youtube_videos"
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                 ui_messages.append({"role": "assistant", "content": msg.content})

        st.session_state.thread_histories[thread_id] = ui_messages
        st.session_state.messages = ui_messages

    except Exception as e:
        st.error(f"Could not retrieve conversation history: {e}")
        st.session_state.messages = []

def reset_chat():
    """Resets the chat to a new, empty thread."""
    new_thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = new_thread_id
    st.session_state.messages = []
    st.session_state.chat_threads[new_thread_id] = "New Chat"
    st.session_state.thread_histories[new_thread_id] = []
    st.session_state.editing_thread_id = None

# ======================= YouTube Display Function =======================

def display_youtube_thumbnails(videos):
    """Renders clickable YouTube video thumbnails."""
    if not isinstance(videos, list):
        return
    for video in videos:
        title = video.get("title", "No Title Available")
        link = video.get("link", "#")
        thumbnail_url = video.get("thumbnail", "")
        if thumbnail_url and link != "#":
            st.markdown(
                f"""
                <a href="{link}" target="_blank" style="text-decoration: none; color: inherit;">
                    <div style="display: flex; align-items: center; border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                        <img src="{thumbnail_url}" style="width: 160px; height: 90px; border-radius: 8px; margin-right: 15px; object-fit: cover;">
                        <div style="flex-grow: 1;">
                            <b style="font-size: 16px;">{title}</b>
                        </div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )

# ======================= Corrected Session State Initialization ===================

# This block runs only ONCE when the app starts or the session is new.
if "thread_id" not in st.session_state:
    st.session_state.editing_thread_id = None
    st.session_state.messages = []
    # Initialize the history cache here
    st.session_state.thread_histories = {} 
    
    try:
        # 1. Fetch all persisted thread IDs from the database.
        persisted_threads = retrieve_all_threads()
        
        # 2. If threads exist, populate the chat_threads dictionary.
        if persisted_threads:
            st.session_state.chat_threads = {tid: f"Chat {str(tid)[:8]}..." for tid in persisted_threads}
            # 3. Set the current thread to the most recent one.
            st.session_state.thread_id = persisted_threads[-1] 
            # 4. Immediately load the history for the most recent thread.
            switch_thread(st.session_state.thread_id)
        else:
            # 5. If no threads exist, create a fresh "New Chat".
            new_thread_id = str(uuid.uuid4())
            st.session_state.thread_id = new_thread_id
            st.session_state.chat_threads = {new_thread_id: "New Chat"}
            st.session_state.thread_histories = {new_thread_id: []}

    except Exception as e:
        # Fallback if the database connection fails on startup.
        st.sidebar.error("Could not retrieve past conversations.")
        st.error(f"Database connection error: {e}")
        # Create a single temporary chat session.
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        st.session_state.chat_threads = {new_thread_id: "New Chat"}

# ============================ Sidebar UI ============================

with st.sidebar:
    st.title("Koulder Chatbot")
    
    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    email_feature_enabled = "GMAIL_USER" in st.secrets and "GMAIL_APP_PASSWORD" in st.secrets
    with st.expander("📧 Email Conversation", expanded=not email_feature_enabled):
        if email_feature_enabled:
            recipient = st.text_input("Recipient's Email", key="email_recipient")
            if st.button("Send Full Conversation", use_container_width=True):
                if recipient and st.session_state.messages:
                    conversation_body = format_conversation_for_email(st.session_state.messages)
                    thread_name = st.session_state.chat_threads.get(st.session_state.thread_id, "Chat")
                    send_email_with_gmail(
                        subject=f"Conversation: {thread_name}",
                        body=conversation_body,
                        to_email=recipient
                    )
                elif not recipient:
                    st.warning("Please enter a recipient's email.")
                else:
                    st.warning("There are no messages to send.")
        else:
            st.warning("Email feature is disabled.")
            st.info("Add `GMAIL_USER` and `GMAIL_APP_PASSWORD` to your Streamlit secrets to enable.")

    st.header("My Conversations")
    
    if "chat_threads" in st.session_state:
        thread_items = reversed(list(st.session_state.chat_threads.items()))
        for thread_id, thread_name in thread_items:
            if st.session_state.editing_thread_id == thread_id:
                col1, col2, col3 = st.columns([3, 1, 1])
                new_name = col1.text_input("New name", value=thread_name, key=f"edit_{thread_id}", label_visibility="collapsed")
                if col2.button("✔️", key=f"save_{thread_id}", use_container_width=True):
                    st.session_state.chat_threads[thread_id] = new_name
                    st.session_state.editing_thread_id = None
                    st.rerun()
                if col3.button("✖️", key=f"cancel_{thread_id}", use_container_width=True):
                    st.session_state.editing_thread_id = None
                    st.rerun()
            else:
                col1, col2 = st.columns([4, 1])
                is_current = str(thread_id) == str(st.session_state.thread_id)
                button_type = "primary" if is_current else "secondary"
                if col1.button(thread_name, key=f"thread_{thread_id}", use_container_width=True, type=button_type):
                    if not is_current:
                        switch_thread(thread_id)
                        st.rerun()
                if col2.button("✏️", key=f"editbtn_{thread_id}", use_container_width=True):
                    st.session_state.editing_thread_id = thread_id
                    st.rerun()

# ============================ Main Chat UI ============================

# Display historical messages
if "messages" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("display_type") == "youtube_videos":
                display_youtube_thumbnails(msg["content"])
            else:
                st.markdown(msg.get("content", ""))

# Handle new user input
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        youtube_display_placeholder = st.empty()
        final_text_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            youtube_videos_found = []
            final_text_content = ""

            for chunk in chatbot.stream({"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="values"):
                last_message = chunk.get("messages", [])[-1] if chunk.get("messages") else None
                if not last_message:
                    continue

                if isinstance(last_message, ToolMessage) and last_message.name == "search_youtube_videos":
                    try:
                        results = json.loads(last_message.content)
                        if isinstance(results, list):
                            youtube_videos_found = results
                            with youtube_display_placeholder.container():
                                display_youtube_thumbnails(youtube_videos_found)
                    except (json.JSONDecodeError, TypeError):
                        pass

                elif isinstance(last_message, AIMessage) and last_message.content:
                    final_text_content = last_message.content
                    final_text_placeholder.markdown(final_text_content)

            if youtube_videos_found:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": youtube_videos_found,
                    "display_type": "youtube_videos"
                })
            
            if final_text_content:
                 st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text_content
                })

            st.session_state.thread_histories[st.session_state.thread_id] = st.session_state.messages
