# src/llm_pipeline/rag_pipeline.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import json
import re

# --- Embedding model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Vector store ---
vectordb = Chroma(
    collection_name="astro_corpus",
    persist_directory="../../data/chroma_db",
    embedding_function=embedding_model,
)

# --- Retriever ---
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# --- Local LLM via Ollama ---
llm = ChatOllama(
    model="llama3.1",  # or "mistral", "phi3", etc.
    temperature=0.1
)

# --- Prompt template ---
prompt_template = PromptTemplate.from_template("""
You are an expert astrophysicist assistant. Use the following context to answer the user's question.
Answer clearly using scientific reasoning.
Do NOT include citation markers like (1) or [1] in the answer.
Do NOT include a References section in the answer.
Do NOT include a Sources section in the answer.
Sources will be listed separately automatically.

Context:
{context}

Question:
{question}

Answer (include sources if relevant):
""")
#If you use specific papers or sections, cite their titles and URLs at the end.

# --- Query function ---
def answer_query(query: str):
    docs = retriever.invoke(query)  # LCEL-compatible call

    # --- Fallback: no relevant docs retrieved ---
    if not docs or len(docs) == 0:
        return {
            "query": query,
            "answer": "I couldn’t find relevant sources to confidently answer that.",
            "sources": []
        }

    # Build context only from docs that actually have content
    valid_docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 20]

    if not valid_docs:
        return {
            "query": query,
            "answer": "I couldn’t find relevant sources to confidently answer that.",
            "sources": []
        }

    context = "\n\n".join([d.page_content for d in docs])
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    # Clean the LLM answer
    final_answer = response.content.strip()
    # Remove unwanted "References" section if the model inserts one
    final_answer = re.split(r"(?i)sources?:", final_answer)[0].strip()
    # Remove inline numbered references like (1), [2]
    final_answer = re.sub(r"\(\d+\)", "", final_answer)
    final_answer = re.sub(r"\[\d+\]", "", final_answer)

    sources = [
        {
            "title": d.metadata.get("title", "Unknown"),
            "url": d.metadata.get("url", "No URL")
        }
        for d in valid_docs
    ]

    # --- Fallback: model answered but we have nothing to cite ---
    if len(sources) == 0:
        final_answer += "\n\nI couldn’t find relevant sources."

    output = {
        "query": query,
        "answer": final_answer,
        "sources": sources
    }

    #print(json.dumps(output, indent=2))  # pretty terminal output
    return output


# --- Example run ---
if __name__ == "__main__":
    query = "What is the mass of Earth?"
    result = answer_query(query)
    print("\n🔭 Answer:\n", result["answer"])
    print("\n📚 Sources:")
    for src in result["sources"]:
        print(f"- {src['title']} ({src['url']})")
