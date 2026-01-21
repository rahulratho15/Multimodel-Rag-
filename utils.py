import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw
import fitz
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import networkx as nx

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
CHUNK_SIZE = 500

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int = 0
    chunk_id: int = 0
    embedding: Optional[np.ndarray] = None

@dataclass
class OCRResult:
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[Image.Image] = None

@dataclass 
class SearchResult:
    chunk: DocumentChunk
    score: float

def extract_pdf_text(pdf_path: str) -> List[DocumentChunk]:
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        filename = Path(pdf_path).name
        chunk_id = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE):
                chunk_text = ' '.join(words[i:i + CHUNK_SIZE])
                if chunk_text.strip():
                    chunks.append(DocumentChunk(text=chunk_text, source=filename, page=page_num, chunk_id=chunk_id))
                    chunk_id += 1
    except Exception as e:
        print(f"PDF error: {e}")
    return chunks

def pdf_page_to_image(pdf_path: str, page_num: int):
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        doc.close()
    except:
        pass
    return None

def ocr_with_boxes(image_path: str) -> OCRResult:
    try:
        img = Image.open(image_path)
        return OCRResult(text="", words=[], image=img)
    except:
        return OCRResult(text="", words=[], image=None)

def create_image_chunks(ocr_result: OCRResult, image_path: str) -> List[DocumentChunk]:
    return []

def highlight_text_region(image, words, query_terms):
    return image.copy() if image else None

def create_embeddings(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype('float32')

def build_faiss_index(chunks: List[DocumentChunk]):
    if not chunks:
        return None, []
    texts = [c.text for c in chunks]
    embeddings = create_embeddings(texts)
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, chunks

class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
    
    def search(self, query: str, top_k: int = 5):
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

def hybrid_search(query, faiss_index, chunks, bm25_index, top_k=5):
    if not chunks or faiss_index is None:
        return []
    query_emb = create_embeddings([query])
    faiss.normalize_L2(query_emb)
    distances, indices = faiss_index.search(query_emb, min(top_k * 2, len(chunks)))
    vector_scores = {int(idx): float(distances[0][i]) for i, idx in enumerate(indices[0]) if idx >= 0}
    bm25_results = bm25_index.search(query, top_k * 2)
    max_bm25 = max([s for _, s in bm25_results]) if bm25_results else 1.0
    bm25_scores = {idx: score / max_bm25 for idx, score in bm25_results}
    combined = {}
    for idx in set(vector_scores.keys()) | set(bm25_scores.keys()):
        combined[idx] = 0.6 * vector_scores.get(idx, 0) + 0.4 * bm25_scores.get(idx, 0)
    sorted_idx = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
    return [SearchResult(chunk=chunks[idx], score=combined[idx]) for idx in sorted_idx]

def build_knowledge_graph(chunks):
    G = nx.Graph()
    for c in chunks:
        G.add_node(f"{c.source}_p{c.page}_c{c.chunk_id}")
    return G

def query_llm(query: str, context: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ.get("HF_TOKEN", ""))
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Answer based on context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=512
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)[:100]}"

def build_context(results, max_chars=6000):
    parts = []
    total = 0
    for r in results:
        text = f"[{r.chunk.source} p{r.chunk.page}]: {r.chunk.text}\n"
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n".join(parts)
