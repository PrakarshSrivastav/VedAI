# rag.py
import os
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# -----------------------
# Config
# -----------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gemini-1.5-flash"  # Or another suitable Gemini model

FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.pkl"

TOP_K = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Read Gemini API key from env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GitaRAG:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=GEMINI_API_KEY)

        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("📖 Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("🔤 Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"🧠 Configuring Gemini generator: {GENERATION_MODEL_NAME}")
        self.generator = genai.GenerativeModel(
            model_name=GENERATION_MODEL_NAME,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_NEW_TOKENS,
            },
        )

        print("✅ RAG pipeline ready.")

    # ------------- RAG core -############
    def retrieve(self, query: str):
        """Embed query and return top-k verses (metadata)."""
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        D, I = self.index.search(np.array([query_embedding]), TOP_K)
        return [self.metadata[i] for i in I[0]]

    def build_prompt(self, query: str, verses):
        """Create the prompt to keep the model grounded in the Gita."""
        prompt = (
            "You are a wise assistant that answers strictly using the teachings of the Bhagavad Gita.\n\n"
            "Relevant verses:\n"
        )
        for v in verses:
            ref = f"Chapter {v['chapter_number']}, Verse {v['chapter_verse']}"
            prompt += f"- ({ref}): {v['translation']}\n"
        prompt += f"\nQuestion: {query}\nAnswer (base it ONLY on the verses above):"
        return prompt

    def generate_answer(self, query: str) -> str:
        verses = self.retrieve(query)
        prompt = self.build_prompt(query, verses)

        try:
            response = self.generator.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred during generation: {e}"


# ------------- CLI test -############
if __name__ == "__main__":
    try:
        rag = GitaRAG()
        print("\nType 'exit' to quit.")
        while True:
            q = input("\n❓ Ask something about the Gita: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            ans = rag.generate_answer(q)
            print("\n🧠 Answer:\n", ans)
    except ValueError as e:
        print(f"Error: {e}")
