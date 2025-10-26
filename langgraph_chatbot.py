from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.utilities import SerpAPIWrapper
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool, Tool  # FIXED: Added Tool here, removed old import
from dotenv import load_dotenv
import base64
import sqlite3
import requests
import os
import json 

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'chatbot-project'


SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not set in environment")


# Initialize SerpAPI wrapper
serpapi_instance = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# -------------------
# 2. Initialize LLM
# -------------------
model = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=1000
)


from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the tool directly. LangChain will handle the argument mapping.
ddg_tool = DuckDuckGoSearchRun(description="A privacy-respecting search engine. Use this for general web searches.")


# SerpAPI tool with logging
def serpapi_with_logging(query: str) -> str:
    print(f"[LOG] Running SerpAPI Search with query: {query}")
    return serpapi_instance.run(query)


serpapi_tool = Tool(
    name="SerpAPI_Search",
    func=serpapi_with_logging,
    description="A real-time search engine. Use this to get up-to-date factual web search results."
)


# Tavily tool with logging
tavily_tool = TavilySearch(
    max_results=5,
    include_images=True,
    log_searches=True,
    description="An AI-optimized search engine. Best for complex, multi-step research questions."
)


SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


@tool
def search_youtube_videos(query: str) -> str:
    """
    Searches YouTube for videos based on a user's query.
    Use this tool whenever a user asks for videos on a specific topic.
    Returns a JSON string containing a list of video details, including titles,
    links, and thumbnail URLs.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return json.dumps({"error": "YouTube API key not found."})

    try:
        # Build the YouTube service object
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Perform the search
        search_request = youtube.search().list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=5  # Fetch up to 5 videos
        )
        search_response = search_request.execute()

        videos = []
        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnails = item["snippet"]["thumbnails"]
            
            # Get the best available thumbnail URL
            thumbnail_url = thumbnails.get("high", {}).get("url") or thumbnails.get("default", {}).get("url", "")
            
            videos.append({
                "title": title,
                "link": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": thumbnail_url
            })
            
        # Return the results as a JSON string, as this is a robust format for tool outputs
        return json.dumps(videos)

    except HttpError as e:
        return json.dumps({"error": f"An API error occurred: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {e}"})

#---------------------------------
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not set in environment. This is required for image generation.")

@tool
def generate_stability_image(prompt: str, negative_prompt: str = "blurry, low quality, text, watermark") -> str:
    """
    Generates a high-quality, realistic image from a text prompt using Stable Diffusion XL.
    Use this tool whenever a user asks to create, draw, or generate an image.
    The prompt should be descriptive for best results.
    Returns a JSON string containing the Base64 encoded image data.
    """
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {"negative_prompt": negative_prompt}
    }
    
    print(f"Generating image with prompt: '{prompt}'")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            image_bytes = response.content
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            print("Image generated successfully.")
            return json.dumps({"image_data": image_base64, "format": "jpeg"})
        else:
            error_message = response.json().get("error", response.text)
            print(f"Image generation failed: {error_message}")
            return json.dumps({"error": f"Failed to generate image: {error_message}"})
            
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return json.dumps({"error": f"An API request error occurred: {e}"})


# Calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# Stock price tool
@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol using Alpha Vantage API"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo") 
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


# -------------------
# 4. Bind tools to LLM
# -------------------
tools = [tavily_tool, search_youtube_videos, generate_stability_image, serpapi_tool, get_stock_price, calculator]
llm_with_tools = model.bind_tools(tools)


# -------------------
# 5. Chat graph setup
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """
    This node invokes the LLM with the current state of messages and returns the AI's response.
    It ensures the response is properly formatted for serialization.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# SQLite checkpointer for persistence
conn = sqlite3.connect(database='chatbot.db3', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpointer)

# 
# -------------------
# 6. Helper to list threads
# -------------------
def retrieve_all_threads():
    all_threads = set()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id FROM checkpoints")
        rows = cursor.fetchall()
        for row in rows:
            all_threads.add(row)
    return list(all_threads)


def get_latest_news():
    api_key = "Ya5b4563c7e4244508c554840b6186921"
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [(a["title"], a["url"]) for a in articles[:10]]
    return []