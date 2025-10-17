# query_vector_db.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Setup ---

# Load the same embedding model used to build the corpus
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the same persistent Chroma directory
chroma_client = chromadb.PersistentClient(path="../../data/chroma_db")

# Get your existing collection
collection_name = "astro_corpus"
collection = chroma_client.get_collection(collection_name)

# --- Query section ---

def query_corpus(query_text, n_results=3):
    """
    Query the vector database with a natural-language question
    and retrieve the top N most relevant documents + sources.
    """

    # 1. Embed the query
    query_embedding = model.encode(query_text).tolist()

    # 2. Search collection for similar embeddings
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Parse and print results
    print(f"\n🔍 Top {n_results} results for query: '{query_text}'\n")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"Result {i+1}")
        print(f"Title: {meta.get('title', 'Unknown')}")
        print(f"Authors: {meta.get('authors', 'Unknown')}")
        print(f"URL: {meta.get('url', 'Unknown')}")
        print(f"Similarity Score: {1 - dist:.3f}")  # higher = better
        print(f"Excerpt: {doc[:300]}...\n")  # preview the text snippet

# --- Example usage ---
if __name__ == "__main__":
    query_text = "How many Galactic globular clusters have streams"
    query_corpus(query_text, n_results=3)