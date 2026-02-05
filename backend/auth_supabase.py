import requests
from fastapi import Depends, HTTPException, Header
from jose import jwt
from jose.exceptions import JWTError
import os

JWKS_URL = os.getenv("SUPABASE_JWKS_URL")

async def verify_supabase_jwt(token: str) -> dict:
    """Verify a Supabase JWT token and return the claims."""
    jwks = requests.get(JWKS_URL).json()
    try:
        payload = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        return payload
    except Exception as e:
        raise JWTError(f"Invalid token: {e}")

def get_current_user(authorization: str = Header(...)):
    token = authorization.split(" ")[1]  # "Bearer <token>"
    jwks = requests.get(JWKS_URL).json()
    try:
        payload = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        return payload["sub"]  # Supabase user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
