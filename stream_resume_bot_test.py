import streamlit as st
from langgraph_chatbot import chatbot, retrieve_all_threads, generate_stability_image
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import base64
from streamlit_lottie import st_lottie


#=================================message box===========================
if st.button("🚀 See What's Coming!", use_container_width=True):
    st.info("""
    **✨ MCP Server: Coming Soon! ✨**
    - Email Access
    - Chat History Access
    - And much more!
    *Stay tuned for exciting updates!*
    """)



#==============================================================
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json
import datetime

# Function to load Lottie from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        st.error("Failed to load Lottie JSON from URL")
        return None

# Get current Streamlit theme mode
theme = st.get_option("theme.base")  # 'dark' or 'light'

# Lottie URLs for dark and light mode (update with actual URLs to your JSONs)
lottie_url_light = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/4hMtG8PCKS.json"   # dark animation for light mode UI
lottie_url_dark = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/4hMtG8PCKS_dark.json"  # light animation for dark mode UI (you need to create this variant)

# Load appropriate animation based on theme
if theme == "dark":
    lottie_json = load_lottieurl(lottie_url_dark)
else:
    lottie_json = load_lottieurl(lottie_url_light)

# Centered container for animation and title
st.markdown("""
    <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 28px; margin-bottom: 24px;'>
        <div id='lottie-animation'>
""", unsafe_allow_html=True)

if lottie_json:
    st_lottie(lottie_json, speed=1, loop=True, quality="high", height=200, width=200)

st.markdown("""
        </div>
        <div style='height:18px;'></div>
        <h2 style='text-align:center; margin-bottom:0;'>How can I help you today?</h2>
    </div>
""", unsafe_allow_html=True)


# ============================== error lottie =====================
# Error logging function
def log_error(message: str, logfile: str = "error_log.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# URLs for error animation variants (add a light and a dark version for error animation)
error_lottie_url_light = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/yjxBJXnTsi.json"
error_lottie_url_dark = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/yjxBJXnTsi_dark.json"

# Load error animation based on theme
if theme == "dark":
    error_lottie = load_lottieurl(error_lottie_url_dark)
else:
    error_lottie = load_lottieurl(error_lottie_url_light)



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

# ======================= Image Generation Functions =======================

def display_generated_image(image_data, prompt=""):
    """Decodes and displays a Base64 encoded image."""
    try:
        image_bytes = base64.b64decode(image_data)
        st.image(image_bytes, caption=f"Generated: \"{prompt}\"" if prompt else "", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {e}")

# ======================= Optimized Thread Switching =======================

def switch_thread(thread_id):
    """
    Switches to a new thread. Uses a cache to avoid slow database lookups.
    """
    st.session_state["thread_id"] = thread_id
    st.session_state.editing_thread_id = None
    
    # Clear cache for this thread to force fresh load from database
    if thread_id in st.session_state.thread_histories:
        del st.session_state.thread_histories[thread_id]
    
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
        # st.error(f"Could not retrieve conversation history: {e}")
        st.error(f" conversation history is disabled: {e}")
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

# ======================= Session State Initialization ===================

# Initialize view mode if not present
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"  # "chat" or "image_gen"

if "thread_id" not in st.session_state:
    st.session_state.editing_thread_id = None
    st.session_state.messages = []
    st.session_state.thread_histories = {} 
    
    try:
        persisted_threads = [item[0] for item in retrieve_all_threads()]
        
        if persisted_threads:
            st.session_state.chat_threads = {tid: f"Chat {str(tid)[:8]}..." for tid in persisted_threads}
            st.session_state.thread_id = persisted_threads[-1] 
            switch_thread(st.session_state.thread_id)
        else:
            new_thread_id = str(uuid.uuid4())
            st.session_state.thread_id = new_thread_id
            st.session_state.chat_threads = {new_thread_id: "New Chat"}
            st.session_state.thread_histories = {new_thread_id: []}

    except Exception as e:
        st.sidebar.error("Could not retrieve past conversations.")
        st.error(f"Database disabled: {e}")
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        st.session_state.chat_threads = {new_thread_id: "New Chat"}

# ============================ Sidebar UI ============================

with st.sidebar:
    st.title("Koulder Chatbot")
    
    # View mode switcher buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.view_mode == "chat" else "secondary"):
            st.session_state.view_mode = "chat"
            st.rerun()
    with col2:
        if st.button("🎨 Images", use_container_width=True, type="primary" if st.session_state.view_mode == "image_gen" else "secondary"):
            st.session_state.view_mode = "image_gen"
            st.rerun()
    
    st.divider()
    
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

# ============================ Main UI - Conditional Display ============================

if st.session_state.view_mode == "chat":
    # ============================ CHAT VIEW ============================
    
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
        # Don't manually append to messages - let LangGraph handle it
        # st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            youtube_display_placeholder = st.empty()
            final_text_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                youtube_videos_found = []
                final_text_content = ""

                # Stream through the chatbot
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

                # After streaming completes, reload from database to get the complete, persisted state
                try:
                    state = chatbot.get_state(config=CONFIG)
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
                    
                    # Update both session state locations
                    st.session_state.messages = ui_messages
                    st.session_state.thread_histories[st.session_state.thread_id] = ui_messages
                    
                except Exception as e:
                    # st.error(f"Error reloading messages: {e}")
                    st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                    st.error("Oops! Something went wrong. Please try again.")
                    # log_error(f"Chatbot error: {str(e)}")
                    

else:
    # ============================ IMAGE GENERATION VIEW ============================
    
    st.title("🎨 AI Image Generator")
    st.write("Create stunning AI-generated images from text descriptions!")
    
    st.divider()
    
    # Image generation form
    with st.form("image_generation_form", clear_on_submit=False):
        image_prompt = st.text_area(
            "Describe the image you want to generate:",
            placeholder="Example: A serene mountain landscape at sunset with a crystal clear lake reflection, photorealistic, 4K",
            height=120,
            help="Be descriptive! The more details you provide, the better the result."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            generate_button = st.form_submit_button("✨ Generate Image", use_container_width=True, type="primary")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear All", use_container_width=True)
        with col3:
            back_button = st.form_submit_button("⬅️ Back to Chat", use_container_width=True)
    
    if back_button:
        st.session_state.view_mode = "chat"
        st.rerun()
    
    if clear_button:
        if "generated_images" in st.session_state:
            del st.session_state.generated_images
        st.rerun()
    
    if generate_button:
        if image_prompt:
            with st.spinner("🎨 Creating your masterpiece... This may take a few seconds."):
                try:
                    response_json = generate_stability_image(image_prompt)
                    response_data = json.loads(response_json)
                    
                    if "image_data" in response_data:
                        # Store generated image in session state
                        if "generated_images" not in st.session_state:
                            st.session_state.generated_images = []
                        
                        st.session_state.generated_images.append({
                            "prompt": image_prompt,
                            "image_data": response_data["image_data"],
                            "format": response_data.get("format", "png")
                        })
                        st.success("✅ Image generated successfully!")
                        st.rerun()
                    elif "error" in response_data:
                        # st.error(f"❌ Image generation failed: {response_data['error']}")
                        st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                        st.error("Oops! Something went wrong. Please try again.")
                        # log_error(f"Chatbot error: {str(e)}")
                except Exception as e:
                    # st.error(f"❌ An error occurred: {str(e)}")
                    st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                    st.error("Oops! Something went wrong. Please try again.")
                    # log_error(f"Chatbot error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a description for the image you want to generate.")
    
    # Display generated images
    if "generated_images" in st.session_state and st.session_state.generated_images:
        st.divider()
        st.subheader(f"📸 Your Generated Images ({len(st.session_state.generated_images)})")
        
        for idx, img_data in enumerate(reversed(st.session_state.generated_images)):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**Prompt:** {img_data['prompt']}")
                with col2:
                    st.markdown(f"*Image #{len(st.session_state.generated_images) - idx}*")
                
                display_generated_image(img_data["image_data"])
                
                # Download button for each image
                col_a, col_b, col_c = st.columns([2, 2, 2])
                with col_a:
                    st.download_button(
                        label="⬇️ Download Image",
                        data=base64.b64decode(img_data["image_data"]),
                        file_name=f"ai_generated_{len(st.session_state.generated_images) - idx}.{img_data.get('format', 'png')}",
                        mime=f"image/{img_data.get('format', 'png')}",
                        key=f"download_{idx}",
                        use_container_width=True
                    )
                
                if idx < len(st.session_state.generated_images) - 1:
                    st.divider()
    else:
        st.info("💡 No images generated yet. Enter a prompt above to get started!")
