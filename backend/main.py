import os
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from auth_supabase import verify_supabase_jwt
from jose.exceptions import JWTError
from rag import rag

load_dotenv()

app = FastAPI(title="Gita RAG API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, str]]

async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Returns user_id if authenticated, None otherwise."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        claims = await verify_supabase_jwt(token)
        return claims.get("sub") or claims.get("user_id")
    except (JWTError, Exception):
        return None

async def get_required_user(authorization: Optional[str] = Header(None)) -> str:
    """Requires authentication, raises 401 if not authenticated."""
    user_id = await get_optional_user(authorization)
    if not user_id:
        raise HTTPException(401, detail="Authentication required")
    return user_id

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, user_id: Optional[str] = Depends(get_optional_user)):
    """Query the Gita RAG. Works without auth for testing."""
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = rag.answer(req.question, history)

    # Only persist chat if user is authenticated and Supabase is configured
    if user_id and os.getenv("SUPABASE_URL"):
        try:
            from db_supabase import get_supabase
            sb = get_supabase()
            sb.table("chats").insert({
                "user_id": user_id,
                "message": req.question,
                "response": result["answer"],
            }).execute()
        except Exception as e:
            print(f"Failed to save chat: {e}")

    return QueryResponse(answer=result["answer"], context=result["context"])

@app.get("/chats")
async def get_chats(user_id: str = Depends(get_required_user)):
    """Get chat history. Requires authentication."""
    if not os.getenv("SUPABASE_URL"):
        raise HTTPException(503, detail="Database not configured")

    from db_supabase import get_supabase
    sb = get_supabase()
    res = sb.table("chats").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
    return {"chats": res.data}
