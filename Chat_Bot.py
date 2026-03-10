
import os
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "llama3"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_CONTEXT_CHARS = 12000



def load_docs_from_file(filepath: str) -> Dict[str, str]:
    docs = {}
    current_section = None
    buffer = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("####"):
                if current_section and buffer:
                    docs[current_section] = "\n".join(buffer).strip()
                    buffer = []
                current_section = line.strip("# ").strip()
            elif current_section:
                buffer.append(line)

        if current_section and buffer:
            docs[current_section] = "\n".join(buffer).strip()

    return docs



def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_whitespace(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

class RAGIndex:
    def __init__(self):
        self.chunks: List[DocChunk] = []
        self.index = None
        self.dim = None

    def build(self, client: Groq, raw_docs: Dict[str, str]):
        for doc_id, text in raw_docs.items():
            pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, t in enumerate(pieces):
                self.chunks.append(DocChunk(doc_id=doc_id, chunk_id=i, text=t))

        chunk_texts = [c.text for c in self.chunks]
        embs = embed_texts(chunk_texts)
        self.dim = embs.shape[1]

        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search(self, client: Groq, query: str, k: int = TOP_K) -> List[Tuple[DocChunk, float]]:
        q_emb = embed_texts([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k)
        result = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            result.append((self.chunks[idx], float(score)))
        return result

SYSTEM_PROMPT = """You are a helpful expert assistant for a mobile telecom/device company.
Answer strictly based on the provided context. If information is not in context, say you don't know.
Be concise, actionable, and include step-by-step instructions when helpful.
Always include a short 'Why this helps' note.
"""

def format_context(citations: List[Tuple[DocChunk, float]]) -> Tuple[str, List[str]]:
    sources_map: Dict[str, int] = {}
    ordered_ids: List[str] = []
    lines: List[str] = []
    for chunk, _ in citations:
        if chunk.doc_id not in sources_map:
            sources_map[chunk.doc_id] = len(sources_map) + 1
            ordered_ids.append(chunk.doc_id)
        num = sources_map[chunk.doc_id]
        lines.append(f"[{num}] {chunk.text}")
    context = "\n\n".join(lines)
    return context, ordered_ids

def build_user_prompt(query: str, context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    ctx = context[:max_chars]
    return f"""# User Question
{query}

# Context (Cited)
{ctx}

# Instructions
- Use only the Context to answer.
- Cite sources like [1], [2] corresponding to context blocks.
- If missing info, say 'I don't know from the provided docs.'
- Provide clear steps if it's a how-to.
- Add a brief 'Why this helps' sectbesion at the end.
"""

def answer_with_llm(client, query, hits):
    context, id_order = format_context(hits)
    prompt = build_user_prompt(query, context)

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return chat_completion.choices[0].message.content


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Please set Groq_API_KEY environment variable.")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    faq_path = "faq.txt"
    if not os.path.exists(faq_path):
        raise FileNotFoundError(f"Missing {faq_path}. Please provide the FAQ file.")

    raw_docs = load_docs_from_file(faq_path)

    rag = RAGIndex()
    print("Building index over domain docs...")
    rag.build(client, raw_docs)
    print("Ready! Ask me something.\n")

    print("\nType your questions below (type 'exit' to quit):\n")
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_question:
            continue
        print("=" * 90)
        print("Q:", user_question)
        hits = rag.search(client, user_question, k=TOP_K)
        ans = answer_with_llm(client, user_question, hits)
        print(ans, "\n")


if __name__ == "__main__":
    main()
