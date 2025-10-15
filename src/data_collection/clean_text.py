# clean_text.py
import json, re, textwrap

def clean(text):
    text = re.sub(r'\$.*?\$', '', text)  # remove inline math
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, size=800):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

data = json.load(open("../../data/raw/arxiv_galaxy.json"))
processed = []
for paper in data:
    clean_summary = clean(paper["summary"])
    for chunk in chunk_text(clean_summary):
        processed.append({
            "title": paper["title"],
            "text": chunk,
            "url": paper["url"],
            "authors": paper["authors"]
        })

json.dump(processed, open("../../data/processed/clean_chunks.json", "w"), indent=2)
