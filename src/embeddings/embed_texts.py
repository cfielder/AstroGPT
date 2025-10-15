# embed_texts.py
from sentence_transformers import SentenceTransformer
import chromadb, json

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="astro_corpus")

data = json.load(open("../../data/processed/clean_chunks.json"))

for i, item in enumerate(data):
    embedding = model.encode(item["text"]).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        metadatas=[{"title": item["title"], "url": item["url"], "authors": item["authors"]}],
        documents=[item["text"]]
    )
