# query_vector_db.py
query = "What methods measure distances to dwarf galaxies?"
q_emb = model.encode(query).tolist()
results = collection.query(query_embeddings=[q_emb], n_results=3)

for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
    print(meta["title"], "\n", textwrap.fill(doc[:300]), "\n")
