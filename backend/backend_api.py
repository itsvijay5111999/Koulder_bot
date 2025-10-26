"""
FastAPI Backend Server
Exposes LangGraph Chatbot and RAG System as REST APIs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv  # <-- ADD THIS IMPORT

# Load .env file on startup
load_dotenv()

# Import your backend modules
from langgraph_chatbot import chatbot, retrieve_all_threads, generate_stability_image
from backend_rag_test import ResearchPaperRAGPinecone
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot Backend API",
    description="Complete backend for LangGraph Chatbot + RAG System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key for security (optional but recommended)
API_KEY = os.getenv("BACKEND_API_KEY", "your_secret_api_key_change_this")

# Global RAG instance
rag_system: Optional[ResearchPaperRAGPinecone] = None

# ==================== Request/Response Models ====================

# Chatbot Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    display_type: Optional[str] = Field(None, description="Type of display (youtube_videos, etc)")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    thread_id: str = Field(..., description="Conversation thread ID")

class ChatResponse(BaseModel):
    success: bool
    messages: List[Dict]
    thread_id: str
    timestamp: str

class ThreadsResponse(BaseModel):
    success: bool
    threads: List[str]

# RAG Models
class RAGQuestionRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    n_results: int = Field(5, ge=1, le=20)

class RAGUpdateRequest(BaseModel):
    categories: List[str] = Field(["cs.AI", "cs.LG", "cs.CL"])
    include_huggingface: bool = Field(True)

class RAGSearchRequest(BaseModel):
    query: str
    n_results: int = Field(5, ge=1, le=20)

class Source(BaseModel):
    title: str
    authors: str
    url: str
    published: str
    source: Optional[str] = None

class RAGAnswerResponse(BaseModel):
    success: bool
    answer: str
    sources: List[Source]
    papers_used: int
    timestamp: str

class RAGStatsResponse(BaseModel):
    success: bool
    total_papers: int
    index_name: str
    dimension: int

class RAGUpdateResponse(BaseModel):
    success: bool
    total_added: int
    total_fetched: int
    unique_papers: int
    categories: Dict
    timestamp: str

# Image Generation Models
class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    negative_prompt: str = Field("blurry, low quality, text, watermark")

class ImageResponse(BaseModel):
    success: bool
    image_data: Optional[str] = None
    format: Optional[str] = None
    error: Optional[str] = None

# ==================== Security ====================

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from header"""
    if API_KEY and API_KEY != "your_secret_api_key_change_this":
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    print("🚀 Starting FastAPI Backend Server...")
    
    # Initialize RAG if credentials available
    groq_key = os.getenv("GROQ_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    if groq_key and pinecone_key:
        try:
            print("Initializing RAG system...")
            rag_system = ResearchPaperRAGPinecone(
                groq_api_key=groq_key,
                pinecone_api_key=pinecone_key,
                pinecone_environment=pinecone_env
            )
            print("✅ RAG system initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize RAG: {e}")
            rag_system = None
    else:
        print("⚠️ RAG credentials not found. RAG endpoints will be disabled.")
    
    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🔴 Shutting down server...")

# ==================== Health Check ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "AI Chatbot Backend",
        "version": "1.0.0",
        "endpoints": {
            "chatbot": "/api/chat/*",
            "rag": "/api/rag/*",
            "image": "/api/image/*"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_available": True,
        "rag_available": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    }

# ==================== CHATBOT ENDPOINTS ====================

@app.post("/api/chat/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Send a message to the chatbot and get response"""
    try:
        CONFIG = {"configurable": {"thread_id": request.thread_id}}
        
        # Create message
        human_message = HumanMessage(content=request.message)
        
        # Get response from chatbot
        messages_list = []
        
        for chunk in chatbot.stream(
            {"messages": [human_message]},
            config=CONFIG,
            stream_mode="values"
        ):
            last_message = chunk.get("messages", [])[-1] if chunk.get("messages") else None
            
            if not last_message:
                continue
            
            # Handle different message types
            if isinstance(last_message, ToolMessage):
                # Handle tool responses (YouTube, etc.)
                if last_message.name == "search_youtube_videos":
                    try:
                        videos = json.loads(last_message.content)
                        messages_list.append({
                            "role": "assistant",
                            "content": videos,
                            "display_type": "youtube_videos"
                        })
                    except:
                        pass
            
            elif isinstance(last_message, AIMessage) and last_message.content:
                messages_list.append({
                    "role": "assistant",
                    "content": last_message.content
                })
        
        return {
            "success": True,
            "messages": messages_list,
            "thread_id": request.thread_id,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/threads", response_model=ThreadsResponse)
async def get_threads(authenticated: bool = Depends(verify_api_key)):
    """Get all conversation threads"""
    try:
        threads = retrieve_all_threads()
        return {
            "success": True,
            "threads": [str(t[0]) for t in threads]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{thread_id}")
async def get_chat_history(
    thread_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Get conversation history for a thread"""
    try:
        CONFIG = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config=CONFIG)
        
        db_messages = state.values.get("messages", [])
        ui_messages = []
        
        for msg in db_messages:
            if isinstance(msg, HumanMessage):
                ui_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ToolMessage) and msg.name == "search_youtube_videos":
                try:
                    videos = json.loads(msg.content)
                    ui_messages.append({
                        "role": "assistant",
                        "content": videos,
                        "display_type": "youtube_videos"
                    })
                except:
                    pass
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                ui_messages.append({"role": "assistant", "content": msg.content})
        
        return {
            "success": True,
            "messages": ui_messages,
            "thread_id": thread_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE GENERATION ENDPOINTS ====================

@app.post("/api/image/generate", response_model=ImageResponse)
async def generate_image(
    request: ImageRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Generate an image using Stable Diffusion"""
    try:
        result_json = generate_stability_image(request.prompt, request.negative_prompt)
        result = json.loads(result_json)
        
        if "image_data" in result:
            return {
                "success": True,
                "image_data": result["image_data"],
                "format": result.get("format", "jpeg"),
                "error": None
            }
        else:
            return {
                "success": False,
                "image_data": None,
                "format": None,
                "error": result.get("error", "Unknown error")
            }
    
    except Exception as e:
        return {
            "success": False,
            "image_data": None,
            "format": None,
            "error": str(e)
        }

# ==================== RAG ENDPOINTS ====================

@app.post("/api/rag/ask", response_model=RAGAnswerResponse)
async def rag_ask_question(
    request: RAGQuestionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Ask a question using RAG"""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
    
    try:
        result = rag_system.answer_question(
            request.question,
            n_results=request.n_results
        )
        
        if not result['success']:
            raise HTTPException(status_code=404, detail=result['answer'])
        
        return {
            **result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/search")
async def rag_search(
    request: RAGSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Search for relevant papers"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.search_papers(request.query, request.n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/stats", response_model=RAGStatsResponse)
async def rag_stats(authenticated: bool = Depends(verify_api_key)):
    """Get RAG database statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        stats = rag_system.get_stats()
        if not stats['success']:
            raise HTTPException(status_code=500, detail="Failed to get stats")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/update", response_model=RAGUpdateResponse)
async def rag_update(
    request: RAGUpdateRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Update RAG database with latest papers (background task)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Run in background
        def do_update():
            result = rag_system.update_daily_papers(
                categories=request.categories,
                include_huggingface=request.include_huggingface
            )
            print(f"✅ Background update: {result['total_added']} papers added")
        
        background_tasks.add_task(do_update)
        
        return {
            "success": True,
            "total_added": 0,
            "total_fetched": 0,
            "unique_papers": 0,
            "categories": {cat: {"status": "processing"} for cat in request.categories},
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/update-sync", response_model=RAGUpdateResponse)
async def rag_update_sync(
    request: RAGUpdateRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Update RAG database synchronously (may be slow)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.update_daily_papers(
            categories=request.categories,
            include_huggingface=request.include_huggingface
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/rag/clear")
async def rag_clear(authenticated: bool = Depends(verify_api_key)):
    """Clear RAG database"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.clear_database()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Utility Endpoints ====================

@app.get("/api/config")
async def get_config():
    """Get server configuration (non-sensitive)"""
    return {
        "rag_available": rag_system is not None,
        "chatbot_available": True,
        "image_generation_available": bool(os.getenv("HUGGINGFACE_TOKEN")),
        "youtube_search_available": bool(os.getenv("YOUTUBE_API_KEY")),
        "version": "1.0.0"
    }

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    print(f"""
    ╔════════════════════════════════════════╗
    ║   AI Chatbot Backend API Server        ║
    ╚════════════════════════════════════════╝
    
    🚀 Starting on {HOST}:{PORT}
    📚 API Docs: http://localhost:{PORT}/docs
    🔧 ReDoc: http://localhost:{PORT}/redoc
    
    Environment Check:
    ✓ GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Not Set'}
    ✓ PINECONE_API_KEY: {'Set' if os.getenv('PINECONE_API_KEY') else 'Not Set'}
    ✓ HUGGINGFACE_TOKEN: {'Set' if os.getenv('HUGGINGFACE_TOKEN') else 'Not Set'}
    ✓ YOUTUBE_API_KEY: {'Set' if os.getenv('YOUTUBE_API_KEY') else 'Not Set'}
    ✓ BACKEND_API_KEY: {'Set' if os.getenv('BACKEND_API_KEY') else 'Using default'}
    
    """)
    
    uvicorn.run(
        "fastapi_server:app",
        host=HOST,
        port=PORT,
        reload=True,  # Set to False in production
        log_level="info"
    )