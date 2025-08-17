import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SERVICE_ROLE_KEY)
