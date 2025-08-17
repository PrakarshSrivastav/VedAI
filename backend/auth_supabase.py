import requests
from fastapi import Depends, HTTPException, Header
from jose import jwt
import os

JWKS_URL = os.getenv("SUPABASE_JWKS_URL")

def get_current_user(authorization: str = Header(...)):
    token = authorization.split(" ")[1]  # "Bearer <token>"
    jwks = requests.get(JWKS_URL).json()
    try:
        payload = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        return payload["sub"]  # Supabase user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
