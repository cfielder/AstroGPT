# clean_text.py
import json, os, re

RAW_PATH = "../../data/raw/arxiv_streams.json"             # new raw file
PROCESSED_PATH = "../../data/processed/clean_chunks.json"
TRACKER_PATH = "../../data/processed/processed_ids.json"  # stores processed URLs

def clean(text):
    text = re.sub(r'\$.*?\$', '', text)  # remove inline math
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, size=800):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

# --- Load new raw data ---
raw_data = json.load(open(RAW_PATH))

# --- Load processed data (if any) ---
if os.path.exists(PROCESSED_PATH):
    processed = json.load(open(PROCESSED_PATH))
else:
    processed = []

# --- Load tracker file (keeps track of processed URLs) ---
if os.path.exists(TRACKER_PATH):
    processed_ids = set(json.load(open(TRACKER_PATH)))
else:
    processed_ids = set()

# --- Process only NEW papers ---
new_count = 0
for paper in raw_data:
    paper_id = paper["url"]
    if paper_id in processed_ids:
        continue  # skip already processed papers

    clean_summary = clean(paper["summary"])
    for chunk in chunk_text(clean_summary):
        processed.append({
            "title": paper["title"],
            "text": chunk,
            "url": paper["url"],
            "authors": paper["authors"]
        })

    processed_ids.add(paper_id)
    new_count += 1

# --- Save updated data and tracker ---
json.dump(processed, open(PROCESSED_PATH, "w"), indent=2)
json.dump(list(processed_ids), open(TRACKER_PATH, "w"), indent=2)

print(f"✅ Added {new_count} new papers. Total processed: {len(processed_ids)}")