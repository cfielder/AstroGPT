# fetch_arxiv.py
import arxiv, json
from tqdm import tqdm

client = arxiv.Client()
def fetch_arxiv_papers(query="galaxy evolution", max_results=100):
    search_query = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in tqdm(client.results(search_query)):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "url": result.entry_id
        })
    return papers

if __name__ == "__main__":
    papers = fetch_arxiv_papers()
    with open("../../data/raw/arxiv_galaxy.json", "w") as f:
        json.dump(papers, f, indent=2)
