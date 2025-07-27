# rag.py
import os
import faiss
import pickle
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------
# Config
# -----------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Primary (gated) model
GENERATION_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Open fallback (no auth needed, smaller)
OPEN_FALLBACK_MODEL = "google/flan-t5-base"

FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.pkl"

TOP_K = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Read HF token from env (set one of these)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


class GitaRAG:
    def __init__(self,
                 gen_model: str = GENERATION_MODEL_NAME,
                 fallback_model: str = OPEN_FALLBACK_MODEL,
                 use_fallback_if_unauth=True):
        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("📖 Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("🔤 Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Decide dtype / device map
        device_map = "auto"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Build pipeline args once
        pipe_args = dict(
            task="text-generation",
            model=gen_model,
            tokenizer=gen_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        # HF auth handling
        if HF_TOKEN:
            # modern transformers uses `token`, older used `use_auth_token`
            # we’ll try `token` first, then fallback.
            try:
                pipe_args["token"] = HF_TOKEN
            except TypeError:
                pipe_args["use_auth_token"] = HF_TOKEN

        # Sampling config lives in call(), not pipeline() — but we can set defaults later.
        try:
            print(f"🧠 Loading generator: {gen_model}")
            self.generator = pipeline(**pipe_args)
            self.using_fallback = False
        except Exception as e:
            if not use_fallback_if_unauth:
                raise

            print(f"⚠️ Could not load {gen_model}. Reason:\n{e}\n")
            print(f"➡️ Falling back to open model: {fallback_model}")

            pipe_args["model"] = fallback_model
            pipe_args["tokenizer"] = fallback_model
            # Remove token in case it caused the error
            pipe_args.pop("token", None)
            pipe_args.pop("use_auth_token", None)

            self.generator = pipeline(**pipe_args)
            self.using_fallback = True

        print("✅ RAG pipeline ready.")

    # ------------- RAG core -------------
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
            ref = f"Chapter {v['chapter']}, Verse {v['verse']}"
            prompt += f"- ({ref}): {v['text']}\n"
        prompt += f"\nQuestion: {query}\nAnswer (base it ONLY on the verses above):"
        return prompt

    def generate_answer(self, query: str) -> str:
        verses = self.retrieve(query)
        prompt = self.build_prompt(query, verses)

        # Generation kwargs (sampling etc.)
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        # For encoder-decoder models like flan-t5, pipeline returns only the generated text.
        outputs = self.generator(prompt, **gen_kwargs)

        # transformers pipelines return list[dict] with 'generated_text'
        text = outputs[0].get("generated_text", "")
        # If the model echoes the prompt, strip it
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        return text


# ------------- CLI test -------------
if __name__ == "__main__":
    rag = GitaRAG()

    print("\nType 'exit' to quit.")
    while True:
        q = input("\n❓ Ask something about the Gita: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = rag.generate_answer(q)
        print("\n🧠 Answer:\n", ans)
