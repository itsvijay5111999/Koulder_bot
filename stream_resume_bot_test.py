import streamlit as st
from langgraph_chatbot import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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
        
    except Exception as e:
        st.error(f"Failed to send email: {e}")


def format_conversation_for_email(messages):
    """Formats the entire chat history into a single string for an email."""
    formatted_lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted_lines)


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state.editing_thread_id = None
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage): role = "user"
        elif isinstance(msg, AIMessage): role = "assistant"
        else: continue
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages


def reset_chat():
    new_thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = new_thread_id
    st.session_state.chat_threads[new_thread_id] = "New Chat"
    st.session_state.message_history = []
    st.session_state.editing_thread_id = None


# ======================= Session State Initialization ===================


if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "chat_threads" not in st.session_state:
    try:
        persisted_threads = retrieve_all_threads()
        st.session_state.chat_threads = {tid: f"Chat {str(tid)[:8]}..." for tid in persisted_threads}
    except Exception as e:
        st.session_state.chat_threads = {}
        st.sidebar.error("Could not retrieve past conversations.")
if "thread_id" not in st.session_state:
    if st.session_state.chat_threads:
        st.session_state.thread_id = list(st.session_state.chat_threads.keys())[-1]
        switch_thread(st.session_state.thread_id)
    else:
        reset_chat()
elif st.session_state.thread_id not in st.session_state.chat_threads:
    reset_chat()
if "editing_thread_id" not in st.session_state:
    st.session_state.editing_thread_id = None
if "stop_generating" not in st.session_state:
    st.session_state.stop_generating = False


# ============================ Sidebar UI ============================


st.sidebar.title("Koulder Chatbot")
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()


with st.sidebar.expander("📧 Email Conversation"):
    recipient = st.text_input("Recipient's Email", key="email_conv_recipient")
    if st.button("Send Full Conversation", use_container_width=True):
        if recipient:
            if st.session_state.message_history:
                conversation_body = format_conversation_for_email(st.session_state.message_history)
                thread_name = st.session_state.chat_threads.get(st.session_state.thread_id, "Chat Conversation")
                send_email_with_gmail(
                    subject=f"Conversation: {thread_name}",
                    body=conversation_body,
                    to_email=recipient
                )
            else:
                st.warning("There are no messages in this conversation to send.")
        else:
            st.warning("Please enter a recipient's email address.")


st.sidebar.header("My Conversations")
thread_items = reversed(list(st.session_state.chat_threads.items()))
for thread_id, thread_name in thread_items:
    if st.session_state.editing_thread_id == thread_id:
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        new_name = col1.text_input("New name", value=thread_name, key=f"edit_{thread_id}", label_visibility="collapsed")
        if col2.button("✔️", key=f"save_{thread_id}", use_container_width=True):
            st.session_state.chat_threads[thread_id] = new_name
            st.session_state.editing_thread_id = None
            st.rerun()
        if col3.button("✖️", key=f"cancel_{thread_id}", use_container_width=True):
            st.session_state.editing_thread_id = None
            st.rerun()
    else:
        col1, col2 = st.sidebar.columns([4, 1])
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


for i, message in enumerate(st.session_state.message_history):
    with st.chat_message(message["role"]):
        col1, col2 = st.columns([10, 1])
        col1.markdown(message["content"])
        if col2.button("📧", key=f"email_{i}", help="Email this message"):
            st.session_state[f"send_email_form_{i}"] = not st.session_state.get(f"send_email_form_{i}", False)
        
        if st.session_state.get(f"send_email_form_{i}"):
            with st.form(key=f"email_form_{i}"):
                recipient_email = st.text_input("Enter recipient's email address:")
                submit_button = st.form_submit_button(label="Send Email")
                if submit_button:
                    if recipient_email:
                        send_email_with_gmail(
                            subject=f"A message from Chatbot: {message['content'][:30]}...",
                            body=message["content"],
                            to_email=recipient_email
                        )
                    else:
                        st.warning("Please enter a recipient email.")
                    st.session_state[f"send_email_form_{i}"] = False
                    st.rerun()


stop_button_placeholder = st.empty()


user_input = st.chat_input("Ask me anything...")
if user_input:
    if st.session_state.chat_threads.get(st.session_state.thread_id) == "New Chat":
        new_name = generate_thread_name_from_message(user_input)
        st.session_state.chat_threads[st.session_state.thread_id] = new_name
    st.session_state.message_history.append({"role": "user", "content": user_input})
    st.session_state.stop_generating = False
    st.rerun()


if st.session_state.message_history and st.session_state.message_history[-1]["role"] == "user":
    user_message_content = st.session_state.message_history[-1]["content"]
    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Show stop button during generation
    if stop_button_placeholder.button("⏹️ Stop Generation", key="stop_gen_btn"):
        st.session_state.stop_generating = True
    
    with st.chat_message("assistant"):
        with st.spinner("🔴Thinking..."):
            status_holder = {"box": None}
            
            def stream_response():
                full_content = ""
                for chunk in chatbot.stream({"messages": [HumanMessage(content=user_message_content)]}, config=CONFIG, stream_mode="values"):
                    if st.session_state.get("stop_generating", False):
                        break
                    
                    if "messages" in chunk and chunk["messages"]:
                        last_message = chunk["messages"][-1]
                        if isinstance(last_message, ToolMessage):
                            tool_name = getattr(last_message, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(f"🔧 Using `{tool_name}`…")
                            else:
                                status_holder["box"].update(label=f"🔧 Using `{tool_name}`…")
                            continue
                        if isinstance(last_message, AIMessage):
                            new_part = last_message.content[len(full_content):]
                            full_content = last_message.content
                            yield new_part

            ai_message = st.write_stream(stream_response())
            
            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
    
    # After generation, check if it was stopped.
    if st.session_state.get("stop_generating", False):
        # If stopped, show Regenerate button
        if stop_button_placeholder.button("🔄 Regenerate", key="regen_btn"):
            st.session_state.message_history.pop() # Remove the partial AI response
            st.session_state.stop_generating = False # Reset flag
            st.rerun() # Rerun to start generation again
    else:
        # If not stopped, clear the button placeholder
        stop_button_placeholder.empty()
        # THE FINAL RERUN IS REMOVED TO PREVENT THE DOUBLE-RUN BUG
