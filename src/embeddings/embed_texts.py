#embed_texts.py
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid
import os

# --- Paths ---
CLEAN_PATH = "../../data/processed/clean_chunks.json"
DB_PATH = "../../data/chroma_db"
COLLECTION_NAME = "astro_corpus"

# --- Helper: Clean metadata for Chroma ---
def sanitize_metadata(item):
    metadata = {
        "title": item.get("title", "Untitled"),
        "url": item.get("url", "Unknown"),
    }

    authors = item.get("authors")
    if isinstance(authors, list):
        metadata["authors"] = ", ".join(authors)
    elif isinstance(authors, str):
        metadata["authors"] = authors
    else:
        metadata["authors"] = "Unknown"

    return metadata


# --- Step 1: Load embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 2: Connect to persistent Chroma ---
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# --- Step 3: Create or load existing collection ---
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

# --- Step 4: Load cleaned data ---
data = json.load(open(CLEAN_PATH))

# --- Step 5: Get existing document URLs to skip duplicates ---
existing = collection.get(include=["metadatas"])
existing_urls = {m["url"] for m in existing["metadatas"] if "url" in m}

print(f"🧩 Found {len(existing_urls)} existing documents in collection.")

# --- Step 6: Embed and add new documents ---
new_count = 0
for item in tqdm(data, desc="Embedding new documents"):
    if item["url"] in existing_urls:
        continue  # Skip already embedded papers

    embedding = model.encode(item["text"]).tolist()
    metadata = sanitize_metadata(item)

    # Use stable ID (prefer URL) to avoid collisions
    uid = item.get("url", str(uuid.uuid4()))

    collection.add(
        ids=[uid],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[item["text"]],
    )
    new_count += 1

print(f"✅ Added {new_count} new document chunks to '{COLLECTION_NAME}'.")