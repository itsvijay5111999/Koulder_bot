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
import requests
import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote
import os

# Import RAG system
try:
    from backend_rag import ResearchPaperRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: backend_rag.py not found. RAG features will be disabled.")

# ======================= Helper Functions =======================

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        st.error("Failed to load Lottie JSON from URL")
        return None

def log_error(message: str, logfile: str = "error_log.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def get_latest_news(category="general", country="us", page_size=100):
    """Fetch news articles with full descriptions from NewsAPI"""
    api_key = "a5b4563c7e4244508c554840b6186921"
    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&pageSize={page_size}&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [{
                'title': a.get('title', 'No Title'),
                'url': a.get('url', '#'),
                'image': a.get('urlToImage', ''),
                'description': a.get('description', 'No description available.'),
                'source': a.get('source', {}).get('name', 'Unknown Source'),
                'published_at': a.get('publishedAt', '')
            } for a in articles]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
    return []

def get_arxiv_papers(category="cs.AI", max_results=100, sort_by="submittedDate"):
    """
    Fetch latest research papers from arXiv API
    
    Categories:
    - cs.AI: Artificial Intelligence
    - cs.LG: Machine Learning
    - cs.CL: Computation and Language (NLP)
    - cs.CV: Computer Vision
    - cs.NE: Neural and Evolutionary Computing
    - stat.ML: Machine Learning (Statistics)
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"cat:{category}"
    
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': 'descending'
    }
    
    query_url = base_url + "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
    
    try:
        response = requests.get(query_url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                    'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                    'authors': [author.find('{http://www.w3.org/2005/Atom}name').text 
                               for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                    'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                    'updated': entry.find('{http://www.w3.org/2005/Atom}updated').text,
                    'pdf_url': '',
                    'arxiv_url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                    'categories': []
                }
                
                # Get PDF link
                for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                        break
                
                # Get categories
                for category in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                    paper['categories'].append(category.get('term'))
                
                papers.append(paper)
            
            return papers
    except Exception as e:
        st.error(f"Error fetching arXiv papers: {e}")
    return []

def generate_thread_name_from_message(message):
    return message[:40] + "..." if len(message) > 40 else message

def send_email_with_gmail(subject, body, to_email):
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
        st.error("Email feature is not configured.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def format_conversation_for_email(messages):
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

def display_generated_image(image_data, prompt=""):
    try:
        image_bytes = base64.b64decode(image_data)
        st.image(image_bytes, caption=f"Generated: \"{prompt}\"" if prompt else "", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {e}")

def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state.editing_thread_id = None
    
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
        st.error(f"Conversation history is disabled: {e}")
        st.session_state.messages = []

def reset_chat():
    new_thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = new_thread_id
    st.session_state.messages = []
    st.session_state.chat_threads[new_thread_id] = "New Chat"
    st.session_state.thread_histories[new_thread_id] = []
    st.session_state.editing_thread_id = None

def display_youtube_thumbnails(videos):
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

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"

# Initialize RAG system
if "rag_system" not in st.session_state and RAG_AVAILABLE:
    try:
        # Try to get Groq API key from multiple sources
        groq_key = None
        
        # 1. Try st.secrets first
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            groq_key = st.secrets["GROQ_API_KEY"]
            print("✅ Groq API key loaded from Streamlit secrets")
        
        # 2. Try environment variables
        elif "GROQ_API_KEY" in os.environ:
            groq_key = os.environ["GROQ_API_KEY"]
            print("✅ Groq API key loaded from environment variables")
        
        # 3. Try .env file (if python-dotenv is installed)
        else:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    print("✅ Groq API key loaded from .env file")
            except ImportError:
                pass
        
        if groq_key:
            st.session_state.rag_system = ResearchPaperRAG(groq_key)
            st.session_state.rag_initialized = True
            print("✅ RAG system initialized successfully")
        else:
            st.session_state.rag_initialized = False
            print("⚠️ No Groq API key found")
    except Exception as e:
        st.session_state.rag_initialized = False
        print(f"❌ RAG initialization error: {e}")
else:
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False

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

# Load animations
lottie_url = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/U6zQ3L2iUa.json"
error_lottie_url = "https://raw.githubusercontent.com/itsvijay5111999/Agentic_bot/main/yjxBJXnTsi.json"
lottie_json = load_lottieurl(lottie_url)
error_lottie = load_lottieurl(error_lottie_url)

# ============================ Sidebar UI ============================

with st.sidebar:
    st.title("Koulder Chatbot")
    
    # View mode switcher
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.view_mode == "chat" else "secondary"):
            st.session_state.view_mode = "chat"
            st.rerun()
    with col2:
        if st.button("🎨 Images", use_container_width=True, type="primary" if st.session_state.view_mode == "image_gen" else "secondary"):
            st.session_state.view_mode = "image_gen"
            st.rerun()
    
    # News button
    if st.button("📰 Latest News", use_container_width=True, type="primary" if st.session_state.view_mode == "news" else "secondary"):
        st.session_state.view_mode = "news"
        st.rerun()
    
    # Research Papers button
    if st.button("📚 Research Papers", use_container_width=True, type="primary" if st.session_state.view_mode == "research" else "secondary"):
        st.session_state.view_mode = "research"
        st.rerun()
    
    # RAG Assistant button - Always show, with status indicator
    rag_button_label = "🤖 RAG Assistant"
    if RAG_AVAILABLE and st.session_state.get('rag_initialized', False):
        rag_button_label = "🤖 RAG Assistant ✅"
    elif RAG_AVAILABLE:
        rag_button_label = "🤖 RAG Assistant ⚠️"
    else:
        rag_button_label = "🤖 RAG Assistant ❌"
    
    if st.button(rag_button_label, use_container_width=True, type="primary" if st.session_state.view_mode == "rag" else "secondary"):
        st.session_state.view_mode = "rag"
        st.rerun()
    
    # Show RAG status info
    if not RAG_AVAILABLE or not st.session_state.get('rag_initialized', False):
        with st.expander("ℹ️ RAG Setup Info", expanded=False):
            if not RAG_AVAILABLE:
                st.warning("backend_rag.py not found")
            elif not st.session_state.get('rag_initialized', False):
                st.warning("Groq API key not configured")
                st.info("""
                **Add to .env file:**
                ```
                GROQ_API_KEY=your_key_here
                ```
                Or set environment variable:
                ```bash
                export GROQ_API_KEY=your_key
                ```
                """)
    
    st.divider()
    
    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    # Coming soon message
    if st.button("🚀 See What's Coming!", use_container_width=True):
        st.info("""
        **✨ MCP Server: Coming Soon! ✨**
        - Email Access
        - Chat History Access
        - And much more!
        *Stay tuned for exciting updates!*
        """)

    # Email feature
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
            st.info("Add `GMAIL_USER` and `GMAIL_APP_PASSWORD` to enable.")

    st.header("My Conversations")
    
    # Thread list
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

    st.divider()

# ============================ Main UI Area ============================

# Header with animation (only for chat view)
if st.session_state.view_mode == "chat" and not st.session_state.messages:
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

# ============================ CHAT VIEW ============================

if st.session_state.view_mode == "chat":
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
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            youtube_display_placeholder = st.empty()
            final_text_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                youtube_videos_found = []
                final_text_content = ""

                try:
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

                    # Reload from database
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
                    
                    st.session_state.messages = ui_messages
                    st.session_state.thread_histories[st.session_state.thread_id] = ui_messages
                    
                except Exception as e:
                    if error_lottie:
                        st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                    st.error("Oops! Something went wrong. Please try again.")
                    log_error(f"Chatbot error: {str(e)}")

# ============================ NEWS VIEW ============================

elif st.session_state.view_mode == "news":
    st.title("📰 Global News Center")
    st.write("Stay informed with the latest headlines from around the world")
    
    st.divider()
    
    # News filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        news_region = st.selectbox(
            "Select Region",
            options=[
                ("🌍 International", "us"),
                ("🇺🇸 United States", "us"),
                ("🇬🇧 United Kingdom", "gb"),
                ("🇮🇳 India", "in"),
                ("🇨🇦 Canada", "ca"),
                ("🇦🇺 Australia", "au"),
                ("🇩🇪 Germany", "de"),
                ("🇫🇷 France", "fr"),
                ("🇯🇵 Japan", "jp"),
                ("🇨🇳 China", "cn")
            ],
            format_func=lambda x: x[0]
        )
    
    with col2:
        news_category = st.selectbox(
            "Select Category",
            options=[
                ("📰 General", "general"),
                ("💼 Business", "business"),
                ("⚙️ Technology", "technology"),
                ("🎬 Entertainment", "entertainment"),
                ("⚽ Sports", "sports"),
                ("🔬 Science", "science"),
                ("❤️ Health", "health")
            ],
            format_func=lambda x: x[0]
        )
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    # Fetch and display news
    with st.spinner("📡 Fetching latest news..."):
        news_articles = get_latest_news(
            category=news_category[1],
            country=news_region[1],
            page_size=100
        )
    
    if news_articles:
        st.success(f"✅ Found {len(news_articles)} articles")
        st.divider()
        
        for idx, article in enumerate(news_articles, 1):
            with st.container():
                # Article header with source and date
                col_meta1, col_meta2 = st.columns([3, 1])
                with col_meta1:
                    st.markdown(f"**{article['source']}**")
                with col_meta2:
                    if article['published_at']:
                        try:
                            pub_date = datetime.datetime.strptime(
                                article['published_at'], 
                                "%Y-%m-%dT%H:%M:%SZ"
                            ).strftime("%b %d, %Y")
                            st.markdown(f"*{pub_date}*")
                        except:
                            pass
                
                # Main content
                col_img, col_content = st.columns([1, 2])
                
                with col_img:
                    if article['image']:
                        try:
                            st.image(article['image'], use_column_width=True)
                        except:
                            st.info("📷 Image unavailable")
                    else:
                        st.info("📷 No image")
                
                with col_content:
                    st.markdown(f"### [{article['title']}]({article['url']})")
                    st.markdown(f"{article['description']}")
                    st.markdown(f"[📖 Read full article →]({article['url']})")
                
                st.divider()
                
                # Add some spacing every 10 articles for better readability
                if idx % 10 == 0:
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No news articles found")
        st.info("""
        **Possible reasons:**
        - NewsAPI key not configured
        - No articles available for this region/category
        - API rate limit reached
        
        **To fix:** Add your API key from https://newsapi.org/ to the `get_latest_news()` function
        """)
    
    # Back to chat button at the bottom
    st.divider()
    if st.button("⬅️ Back to Chat", use_container_width=True):
        st.session_state.view_mode = "chat"
        st.rerun()

# ============================ RESEARCH PAPERS VIEW ============================

elif st.session_state.view_mode == "research":
    st.title("📚 AI/ML Research Papers")
    st.write("Latest research papers from arXiv - Stay ahead in GenAI & Machine Learning")
    
    st.divider()
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        paper_category = st.selectbox(
            "Select Research Area",
            options=[
                ("🤖 Artificial Intelligence (AI)", "cs.AI"),
                ("🧠 Machine Learning", "cs.LG"),
                ("💬 Natural Language Processing", "cs.CL"),
                ("👁️ Computer Vision", "cs.CV"),
                ("🧬 Neural & Evolutionary Computing", "cs.NE"),
                ("📊 Statistical Machine Learning", "stat.ML"),
                ("🔐 Cryptography & Security", "cs.CR"),
                ("🤝 Human-Computer Interaction", "cs.HC"),
                ("🎮 Multiagent Systems", "cs.MA")
            ],
            format_func=lambda x: x[0]
        )
    
    with col2:
        sort_option = st.selectbox(
            "Sort By",
            options=[
                ("📅 Latest First (Submitted)", "submittedDate"),
                ("🔄 Recently Updated", "lastUpdatedDate"),
                ("📈 Relevance", "relevance")
            ],
            format_func=lambda x: x[0]
        )
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            if 'cached_papers' in st.session_state:
                del st.session_state.cached_papers
            st.rerun()
    
    # Number of papers to fetch
    num_papers = st.slider("Number of papers to display", min_value=10, max_value=100, value=50, step=10)
    
    st.divider()
    
    # Fetch papers (with caching for better performance)
    cache_key = f"{paper_category[1]}_{sort_option[1]}_{num_papers}"
    
    if 'cached_papers' not in st.session_state or st.session_state.get('cache_key') != cache_key:
        with st.spinner("🔍 Fetching latest research papers from arXiv..."):
            papers = get_arxiv_papers(
                category=paper_category[1],
                max_results=num_papers,
                sort_by=sort_option[1]
            )
            st.session_state.cached_papers = papers
            st.session_state.cache_key = cache_key
    else:
        papers = st.session_state.cached_papers
    
    if papers:
        st.success(f"✅ Found {len(papers)} research papers")
        st.divider()
        
        # Search/filter within results
        search_term = st.text_input("🔍 Search within results", placeholder="Enter keywords to filter papers...")
        
        if search_term:
            filtered_papers = [
                p for p in papers 
                if search_term.lower() in p['title'].lower() 
                or search_term.lower() in p['summary'].lower()
            ]
            st.info(f"Showing {len(filtered_papers)} papers matching '{search_term}'")
        else:
            filtered_papers = papers
        
        st.divider()
        
        # Display papers
        for idx, paper in enumerate(filtered_papers, 1):
            with st.expander(f"**{idx}. {paper['title']}**", expanded=False):
                # Metadata
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    try:
                        pub_date = datetime.datetime.strptime(
                            paper['published'][:10], 
                            "%Y-%m-%d"
                        ).strftime("%B %d, %Y")
                        st.markdown(f"📅 **Published:** {pub_date}")
                    except:
                        st.markdown(f"📅 **Published:** {paper['published'][:10]}")
                
                with col_meta2:
                    if paper['categories']:
                        st.markdown(f"🏷️ **Category:** {', '.join(paper['categories'])}")
                
                # Authors
                if paper['authors']:
                    authors_text = ", ".join(paper['authors'][:5])
                    if len(paper['authors']) > 5:
                        authors_text += f" +{len(paper['authors']) - 5} more"
                    st.markdown(f"👥 **Authors:** {authors_text}")
                
                st.divider()
                
                # Abstract/Summary
                st.markdown("### 📄 Abstract")
                st.markdown(paper['summary'])
                
                st.divider()
                
                # Action buttons
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if paper['pdf_url']:
                        st.link_button(
                            "📥 Download PDF",
                            paper['pdf_url'],
                            use_container_width=True
                        )
                
                with col_btn2:
                    st.link_button(
                        "🔗 View on arXiv",
                        paper['arxiv_url'],
                        use_container_width=True
                    )
                
                with col_btn3:
                    # Copy citation button
                    citation = f"{', '.join(paper['authors'][:3])} et al. ({paper['published'][:4]}). {paper['title']}. arXiv preprint {paper['arxiv_url'].split('/')[-1]}."
                    if st.button(f"📋 Copy Citation", key=f"cite_{idx}", use_container_width=True):
                        st.code(citation, language=None)
                
                # Add some spacing
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Pagination info
        if filtered_papers:
            st.divider()
            st.info(f"📊 Displaying {len(filtered_papers)} of {len(papers)} papers")
    
    else:
        st.warning("⚠️ No research papers found")
        st.info("""
        **Possible reasons:**
        - Network connectivity issues
        - arXiv API temporarily unavailable
        
        **Note:** arXiv is a free service and doesn't require an API key!
        """)
    
    # Back to chat button
    st.divider()
    if st.button("⬅️ Back to Chat", use_container_width=True, key="research_back"):
        st.session_state.view_mode = "chat"
        st.rerun()

# ============================ RAG ASSISTANT VIEW ============================

elif st.session_state.view_mode == "rag":
    st.title("🤖 RAG Research Assistant")
    st.write("Ask questions about AI/ML research papers - powered by your knowledge base")
    
    st.divider()
    
    # Check if RAG is available and initialized
    if not RAG_AVAILABLE:
        st.error("❌ RAG system not available. Please ensure `backend_rag.py` is in the same directory.")
        st.info("Required packages: `pip install chromadb groq`")
        if st.button("⬅️ Back to Chat", use_container_width=True):
            st.session_state.view_mode = "chat"
            st.rerun()
        st.stop()
    
    if not st.session_state.rag_initialized:
        st.error("❌ RAG system not initialized. Please add your GROQ_API_KEY to Streamlit secrets.")
        st.info("""
        **To enable RAG:**
        1. Get your Groq API key from https://console.groq.com
        2. Add it to `.streamlit/secrets.toml`:
        ```
        GROQ_API_KEY = "your_groq_api_key_here"
        ```
        3. Install required packages:
        ```
        pip install chromadb groq
        ```
        """)
        if st.button("⬅️ Back to Chat", use_container_width=True):
            st.session_state.view_mode = "chat"
            st.rerun()
        st.stop()
    
    # RAG is available - show interface
    rag = st.session_state.rag_system
    
    # Database management section
    with st.expander("⚙️ Database Management", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Update Database", use_container_width=True):
                with st.spinner("Fetching latest papers from arXiv..."):
                    categories = st.multiselect(
                        "Select categories to update:",
                        ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
                        default=["cs.AI", "cs.LG", "cs.CL"]
                    )
                    if categories:
                        result = rag.update_daily_papers(categories=categories)
                        if result['success']:
                            st.success(f"✅ Added {result['total_added']} new papers!")
                            for cat, stats in result['categories'].items():
                                st.info(f"{cat}: Fetched {stats['fetched']}, Added {stats['added']}")
                        else:
                            st.error("Failed to update database")
        
        with col2:
            stats = rag.get_stats()
            if stats['success']:
                st.metric("Total Papers", stats['total_papers'])
            else:
                st.error("Error getting stats")
        
        with col3:
            if st.button("🗑️ Clear Database", use_container_width=True):
                if st.checkbox("Are you sure?"):
                    result = rag.clear_database()
                    if result['success']:
                        st.success("Database cleared!")
                        st.rerun()
    
    st.divider()
    
    # Quick update button
    col_update1, col_update2 = st.columns([3, 1])
    with col_update1:
        st.info("💡 Update your database regularly to get the latest research papers!")
    with col_update2:
        if st.button("⚡ Quick Update", use_container_width=True):
            with st.spinner("Updating..."):
                result = rag.update_daily_papers(categories=["cs.AI", "cs.LG", "cs.CL"])
                if result['success']:
                    st.toast(f"Added {result['total_added']} papers!", icon="✅")
                    st.rerun()
    
    st.divider()
    
    # Initialize RAG chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Display RAG chat history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources if available
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 View {len(msg['sources'])} Source Papers"):
                    for idx, source in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        **{idx}. {source['title']}**
                        - Authors: {source['authors']}
                        - Published: {source['published']}
                        - [View on arXiv]({source['url']})
                        """)
                        st.divider()
    
    # Example questions
    if not st.session_state.rag_messages:
        st.markdown("### 💬 Try asking questions like:")
        example_questions = [
            "What are the latest developments in large language models?",
            "Summarize recent papers about diffusion models",
            "What are the new techniques in prompt engineering?",
            "Tell me about recent advances in computer vision",
            "What are the challenges in AI alignment?",
        ]
        
        cols = st.columns(2)
        for idx, question in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(f"💡 {question}", key=f"example_{idx}", use_container_width=True):
                    st.session_state.rag_question_input = question
                    st.rerun()
    
    # Chat input
    if "rag_question_input" in st.session_state:
        user_question = st.session_state.rag_question_input
        del st.session_state.rag_question_input
    else:
        user_question = st.chat_input("Ask anything about AI/ML research...")
    
    if user_question:
        # Add user message
        st.session_state.rag_messages.append({
            "role": "user",
            "content": user_question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get answer from RAG
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching research papers and generating answer..."):
                result = rag.answer_question(user_question, n_results=5)
                
                if result['success']:
                    st.markdown(result['answer'])
                    
                    # Show sources
                    if result['sources']:
                        with st.expander(f"📚 View {len(result['sources'])} Source Papers"):
                            for idx, source in enumerate(result['sources'], 1):
                                st.markdown(f"""
                                **{idx}. {source['title']}**
                                - Authors: {source['authors']}
                                - Published: {source['published']}
                                - [View on arXiv]({source['url']})
                                """)
                                st.divider()
                    
                    # Add assistant message
                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                else:
                    error_msg = result['answer']
                    st.error(error_msg)
                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
    
    # Clear chat button
    st.divider()
    col_clear, col_back = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.rag_messages = []
            st.rerun()
    with col_back:
        if st.button("⬅️ Back to Chat", use_container_width=True, key="rag_back"):
            st.session_state.view_mode = "chat"
            st.rerun()

# ============================ IMAGE GENERATION VIEW ============================

else:
    st.title("🎨 AI Image Generator")
    st.write("Create stunning AI-generated images from text descriptions!")
    
    st.divider()
    
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
            with st.spinner("🎨 Creating your masterpiece..."):
                try:
                    response_json = generate_stability_image(image_prompt)
                    response_data = json.loads(response_json)
                    
                    if "image_data" in response_data:
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
                        if error_lottie:
                            st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                        st.error("Oops! Something went wrong. Please try again.")
                        log_error(f"Image generation error: {response_data['error']}")
                except Exception as e:
                    if error_lottie:
                        st_lottie(error_lottie, speed=1, loop=True, quality="high", height=180, width=180)
                    st.error("Oops! Something went wrong. Please try again.")
                    log_error(f"Image generation error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a description for the image.")
    
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