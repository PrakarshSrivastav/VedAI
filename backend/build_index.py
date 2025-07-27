# build_index.py

import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config
GITA_CSV_PATH = "../gita_data/Gita.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.pkl"


def load_gita(csv_path):
    """Load Gita verses from your full schema."""
    df = pd.read_csv(csv_path)

    required_cols = {'chapter_number', 'chapter_title', 'chapter_verse', 'translation'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing one or more required columns: {required_cols}")

    verses = []
    for _, row in df.iterrows():
        chapter_number = str(row['chapter_number']).strip()
        chapter_title = str(row['chapter_title']).strip()
        chapter_verse = str(row['chapter_verse']).strip()
        translation = str(row['translation']).strip()

        # Skip empty rows
        if not translation or translation.lower() == 'nan':
            continue

        verses.append({
            'chapter_number': chapter_number,
            'chapter_title': chapter_title,
            'chapter_verse': chapter_verse,
            'translation': translation
        })

    return verses


def build_faiss_index(verses, model):
    """Embed verses and build FAISS index."""
    texts = [v['translation'] for v in verses]
    print(f"Embedding {len(texts)} verses...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


def save_metadata(verses, path):
    """Save full verse metadata (chapter, title, verse, text)."""
    with open(path, 'wb') as f:
        pickle.dump(verses, f)


def main():
    print("🔄 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("📖 Reading Gita CSV...")
    verses = load_gita(GITA_CSV_PATH)

    print("📦 Building FAISS index...")
    index = build_faiss_index(verses, model)

    print(f"💾 Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"💾 Saving metadata to {METADATA_PATH}...")
    save_metadata(verses, METADATA_PATH)

    print("✅ Done! You can now use the RAG pipeline.")


if __name__ == "__main__":
    main()
