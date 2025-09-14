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
from langchain_core.tools import tool
from langchain.agents import Tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
import json 
# -------------------
# 1. Load environment
# -------------------
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'chatbot-project'


SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not set in environment")


# Initialize SerpAPI wrapper
serpapi_instance = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)




# -------------------







# -------------------
# 2. Initialize LLM
# -------------------
model = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b", # Switched to a more common and powerful model
    temperature=0.5,
    max_tokens=1000
)


# -------------------
# 3. Define tools properly
# -------------------
# DuckDuckGo wrapped as Tool
# Corrected code
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
    # NOTE: You need to get your own free API key from Alpha Vantage
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo") 
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


# -------------------
# 4. Bind tools to LLM
# -------------------
tools = [ddg_tool, serpapi_tool, tavily_tool, get_stock_price, calculator]
llm_with_tools = model.bind_tools(tools)


# -------------------
# 5. Chat graph setup
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ===== THIS IS THE CORRECTED CHAT NODE =====
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

# Use an in-memory checkpointer for simplicity, or keep your SQLite one
# For production, SqliteSaver is great. For quick testing, memory is easier.
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


# -------------------
# 6. Helper to list threads
# -------------------
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

