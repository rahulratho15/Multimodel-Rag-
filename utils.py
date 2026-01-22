import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import fitz
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    
@dataclass
class Result:
    chunk: Chunk
    score: float

def extract_pdf(path: str) -> List[Chunk]:
    chunks = []
    try:
        reader = PdfReader(path)
        name = Path(path).name
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            text = re.sub(r'\s+', ' ', text)
            words = text.split()
            for j in range(0, len(words), 400):
                t = ' '.join(words[j:j+450])
                if t.strip():
                    chunks.append(Chunk(text=t, source=name, page=i))
    except:
        pass
    return chunks

def pdf_to_image(path: str, page: int) -> Optional[Image.Image]:
    try:
        doc = fitz.open(path)
        if 0 <= page-1 < len(doc):
            pix = doc[page-1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        doc.close()
    except:
        pass
    return None

def embed(texts: List[str]) -> np.ndarray:
    return np.array(get_model().encode(texts, show_progress_bar=False)).astype('float32')

def build_index(chunks: List[Chunk]) -> Tuple[faiss.Index, List[Chunk]]:
    if not chunks:
        return None, []
    emb = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    return index, chunks

class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
    
    def search(self, q, k=5):
        scores = self.bm25.get_scores(q.lower().split())
        top = np.argsort(scores)[-k:][::-1]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

def search(query, index, chunks, bm25, k=5) -> List[Result]:
    if not chunks or index is None:
        return []
    qe = embed([query])
    faiss.normalize_L2(qe)
    dist, idx = index.search(qe, min(k*2, len(chunks)))
    vs = {int(i): float(dist[0][j]) for j, i in enumerate(idx[0]) if i >= 0}
    br = bm25.search(query, k*2)
    mx = max([s for _, s in br]) if br else 1.0
    bs = {i: s/mx for i, s in br}
    combined = {}
    for i in set(vs) | set(bs):
        combined[i] = 0.6*vs.get(i, 0) + 0.4*bs.get(i, 0)
    top = sorted(combined, key=lambda x: combined[x], reverse=True)[:k]
    return [Result(chunk=chunks[i], score=combined[i]) for i in top]

def ask_llm(query: str, context: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ.get("HF_TOKEN", ""))
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Answer based on the context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def get_context(results: List[Result]) -> str:
    return "\n---\n".join([f"[{r.chunk.source} p{r.chunk.page}]: {r.chunk.text}" for r in results[:5]])
