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
from langchain_core.tools import tool
from langchain.agents import Tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
import json 

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'chatbot-project'


SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not set in environment")


# # Initialize SerpAPI wrapper
serpapi_instance = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# # -------------------
# # 2. Initialize LLM
# # -------------------
model = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", # Switched to a more common and powerful model
    temperature=0.5,
    max_tokens=1000
)



from langchain_community.tools import DuckDuckGoSearchRun

# # Initialize the tool directly. LangChain will handle the argument mapping.
ddg_tool = DuckDuckGoSearchRun(description="A privacy-respecting search engine. Use this for general web searches.")


#hi
# # SerpAPI tool with logging
def serpapi_with_logging(query: str) -> str:
    print(f"[LOG] Running SerpAPI Search with query: {query}")
    return serpapi_instance.run(query)


serpapi_tool = Tool(
    name="SerpAPI_Search",
    func=serpapi_with_logging,
    description="A real-time search engine. Use this to get up-to-date factual web search results."
)


# # Tavily tool with logging
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



# # Calculator tool
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


# # Stock price tool
@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol using Alpha Vantage API"""
    # NOTE: You need to get your own free API key from Alpha Vantage
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo") 
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


# # -------------------
# # 4. Bind tools to LLM
# # -------------------
tools = [tavily_tool, search_youtube_videos, serpapi_tool, get_stock_price, calculator]
llm_with_tools = model.bind_tools(tools)


# # -------------------
# # 5. Chat graph setup
# # -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# # ===== THIS IS THE CORRECTED CHAT NODE =====
def chat_node(state: ChatState) -> dict:
    """
    This node invokes the LLM with the current state of messages and returns the AI's response.
    It ensures the response is properly formatted for serialization.
    """
    response = llm_with_tools.invoke(state["messages"])
    
    # The AIMessage from the LLM is already in a serializable format if it contains
    # tool_calls. We just need to ensure it is added to the state.
    # The 'add_messages' function in the state handles this correctly.
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

# # Use an in-memory checkpointer for simplicity, or keep your SQLite one
# # For production, SqliteSaver is great. For quick testing, memory is easier.
# memory = SqliteSaver.from_conn_string(":memory:")
conn = sqlite3.connect(database='chatbot.db3', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpointer)


# # -------------------
# # 6. Helper to list threads
# # -------------------
def retrieve_all_threads():
    all_threads = set()
    # Check if conn is not None before using it
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id FROM checkpoints")
        rows = cursor.fetchall()
        for row in rows:
            all_threads.add(row)
    return list(all_threads)

#-----------------------------new edit-----------------------------

# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.utilities import SerpAPIWrapper
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.tools import tool
# from langchain.agents import Tool
# from dotenv import load_dotenv
# from io import BytesIO
# import base64

# import sqlite3
# import requests
# import os
# import json

# # -------------------
# # 1. Load Environment & Initialize APIs
# # -------------------
# load_dotenv()
# SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
# if not SERPAPI_API_KEY:
#     raise ValueError("SERPAPI_API_KEY not set in environment")

# -------------------
# 2. Initialize LLM
# -------------------
# model = ChatGroq(
#     model="deepseek-r1-distill-llama-70b", # Updated to a generally reliable Groq model
#     temperature=0.5,
#     max_tokens=1000
# )

# -------------------
# 3. Define Tools
# -------------------

# --- Search Tools ---
# tavily_tool = TavilySearchResults(max_results=3, name="tavily_search") # Explicitly naming for consistency
# serpapi_instance = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
# serpapi_tool = Tool(
#     name="serpapi_search", # Renamed for clarity and consistency
#     func=serpapi_instance.run,
#     description="A real-time search engine. Use for up-to-date facts, news, or current events."
# )

# import os
# import json
# from langchain_core.tools import tool
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from dotenv import load_dotenv

# Load environment variables (recommended for storing API key)
# load_dotenv()

# # --- 1. Define the YouTube Search Tool ---
# @tool
# def search_youtube_videos(query: str) -> str:
#     """
#     Searches YouTube for videos based on a user's query.
#     Use this tool whenever a user asks for videos on a specific topic.
#     Returns a JSON string containing a list of video details, including titles,
#     links, and thumbnail URLs.
#     """
#     api_key = os.getenv("YOUTUBE_API_KEY")
#     if not api_key:
#         return json.dumps({"error": "YouTube API key not found."})

#     try:
#         # Build the YouTube service object
#         youtube = build('youtube', 'v3', developerKey=api_key)
        
#         # Perform the search
#         search_request = youtube.search().list(
#             q=query,
#             part="id,snippet",
#             type="video",
#             maxResults=5  # Fetch up to 5 videos
#         )
#         search_response = search_request.execute()

#         videos = []
#         for item in search_response.get("items", []):
#             video_id = item["id"]["videoId"]
#             title = item["snippet"]["title"]
#             thumbnails = item["snippet"]["thumbnails"]
            
#             # Get the best available thumbnail URL
#             thumbnail_url = thumbnails.get("high", {}).get("url") or thumbnails.get("default", {}).get("url", "")
            
#             videos.append({
#                 "title": title,
#                 "link": f"https://www.youtube.com/watch?v={video_id}",
#                 "thumbnail": thumbnail_url
#             })
            
#         # Return the results as a JSON string, as this is a robust format for tool outputs
#         return json.dumps(videos)

#     except HttpError as e:
#         return json.dumps({"error": f"An API error occurred: {e}"})
#     except Exception as e:
#         return json.dumps({"error": f"An unexpected error occurred: {e}"})




# # --- Other Tools (Defined correctly with @tool) ---
# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """Perform basic arithmetic (add, sub, mul, div)."""
#     # ... (implementation is correct) ...
#     try:
#         if operation == "add": result = first_num + second_num
#         elif operation == "sub": result = first_num - second_num
#         elif operation == "mul": result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0: return {"error": "Division by zero."}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation '{operation}'"}
#         return {"result": result}
#     except Exception as e:
#         return {"error": str(e)}

# @tool
# def get_stock_price(symbol: str) -> dict:
#     """Fetch the latest stock price for a given stock symbol."""
#     api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
#     url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
#     r = requests.get(url)
#     return r.json()

# -------------------
# 4. Bind Tools and Define State
# -------------------
# tools = [tavily_tool, search_youtube_videos, serpapi_tool, get_stock_price, calculator]
# model_with_tools = model.bind_tools(tools)

# class ChatState(TypedDict):
#     messages: Annotated[list, add_messages]

# -------------------
# 5. Define Graph Nodes
# -------------------
# def chat_node(state: ChatState) -> dict:
#     """This node invokes the LLM to decide on the next action."""
#     # The model now uses the tool definitions directly, no complex system prompt needed.
#     response = model_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# The prebuilt ToolNode handles executing the correct tool based on its name.
# tool_node = ToolNode(tools)

# -------------------
# 6. Build and Compile Graph
# -------------------
# conn = sqlite3.connect(database='chatbot.db3', check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

# graph = StateGraph(ChatState)
# graph.add_node("chat", chat_node)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chat")
# graph.add_conditional_edges(
#     "chat",
#     tools_condition, # Use the robust prebuilt condition
#     {"tools": "tools", END: END}
# )
# graph.add_edge('tools', 'chat')

# chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper to List Threads
# -------------------
# def retrieve_all_threads():
#     """Retrieves all unique thread IDs from the checkpoint database."""
#     if not conn: return []
#     with conn: # Use context manager for safety
#         cursor = conn.cursor()
#         cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
#         return [row[0] for row in cursor.fetchall()]

