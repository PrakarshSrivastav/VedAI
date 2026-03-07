# rag.py
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# -----------------------
# Config
# -----------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "llama-3.3-70b-versatile"

script_dir = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(script_dir, "faiss_index.index")
METADATA_PATH = os.path.join(script_dir, "metadata.pkl")

TOP_K = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GitaRAG:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )

        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("📖 Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("🔤 Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"🧠 Using Groq model: {GENERATION_MODEL_NAME}")
        print("✅ RAG pipeline ready.")

    def retrieve(self, query: str):
        """Embed query and return top-k verses (metadata)."""
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        D, I = self.index.search(np.array([query_embedding]), TOP_K)
        return [self.metadata[i] for i in I[0]]

    def build_prompt(self, query: str, verses):
        """Create the prompt to keep the model grounded in the Gita."""
        context = "Relevant verses:\n"
        for v in verses:
            ref = f"Chapter {v['chapter_number']}, Verse {v['chapter_verse']}"
            context += f"- ({ref}): {v['translation']}\n"
        return context, f"Question: {query}\nAnswer (base it ONLY on the verses above):"

    def _call_llm(self, query: str, verses) -> str:
        context, user_message = self.build_prompt(query, verses)
        response = self.client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a wise assistant that answers strictly using the teachings of the Bhagavad Gita.\n\n" + context,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content

    def generate_answer(self, query: str) -> str:
        verses = self.retrieve(query)
        try:
            return self._call_llm(query, verses)
        except Exception as e:
            return f"An error occurred during generation: {e}"

    def answer(self, query: str) -> dict:
        """Return answer and context for API responses."""
        verses = self.retrieve(query)
        try:
            answer_text = self._call_llm(query, verses)
        except Exception as e:
            answer_text = f"An error occurred during generation: {e}"

        context = [
            {
                "chapter_number": str(v["chapter_number"]),
                "chapter_verse": str(v["chapter_verse"]),
                "translation": v["translation"],
            }
            for v in verses
        ]
        return {"answer": answer_text, "context": context}


# Create module-level instance for import
rag = GitaRAG()

# ------------- CLI test -------------
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
