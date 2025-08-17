import os
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from auth_supabase import verify_supabase_jwt, JWTError
from db_supabase import get_supabase
from rag import rag

load_dotenv()

app = FastAPI(title="Gita RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, str]]

async def get_user_claims(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        claims = await verify_supabase_jwt(token)
    except JWTError as e:
        raise HTTPException(401, detail=str(e))
    return claims

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, claims: Dict[str, Any] = Depends(get_user_claims)):
    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid token claims")

    result = rag.answer(req.question)

    # persist chat
    sb = get_supabase()
    sb.table("chats").insert({
        "user_id": user_id,
        "message": req.question,
        "response": result["answer"],
    }).execute()

    return QueryResponse(answer=result["answer"], context=result["context"])

@app.get("/chats")
async def get_chats(claims: Dict[str, Any] = Depends(get_user_claims)):
    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid token claims")

    sb = get_supabase()
    res = sb.table("chats").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
    return {"chats": res.data}
