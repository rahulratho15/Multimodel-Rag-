"""
Utility functions for Multimodal RAG Application
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import networkx as nx
from huggingface_hub import InferenceClient

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Check for pytesseract
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except:
    print("Tesseract not available")


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int = 0
    chunk_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
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
    source_image: Optional[Image.Image] = None


# Model cache
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


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
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = ' '.join(words[i:i + CHUNK_SIZE])
                if chunk_text.strip():
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        source=filename,
                        page=page_num,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
    except Exception as e:
        print(f"PDF error: {e}")
    return chunks


def pdf_page_to_image(pdf_path: str, page_num: int) -> Optional[Image.Image]:
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc[page_num - 1]
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        doc.close()
    except Exception as e:
        print(f"PDF to image error: {e}")
    return None


def ocr_with_boxes(image_path: str) -> OCRResult:
    try:
        img = Image.open(image_path)
        
        if not TESSERACT_AVAILABLE:
            return OCRResult(text="", words=[], image=img)
        
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        words = []
        full_text_parts = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf > 30:
                words.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'conf': conf
                })
                full_text_parts.append(text)
        
        return OCRResult(text=' '.join(full_text_parts), words=words, image=img)
    except Exception as e:
        print(f"OCR error: {e}")
        try:
            return OCRResult(text="", words=[], image=Image.open(image_path))
        except:
            return OCRResult(text="", words=[], image=None)


def create_image_chunks(ocr_result: OCRResult, image_path: str) -> List[DocumentChunk]:
    chunks = []
    filename = Path(image_path).name
    
    if not ocr_result.text:
        return chunks
    
    word_list = ocr_result.text.split()
    for i in range(0, len(word_list), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = ' '.join(word_list[i:i + CHUNK_SIZE])
        if chunk_text.strip():
            chunks.append(DocumentChunk(
                text=chunk_text,
                source=filename,
                page=1,
                chunk_id=len(chunks)
            ))
    return chunks


def highlight_text_region(image: Image.Image, words: List[Dict[str, Any]], query_terms: List[str]) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    query_lower = [q.lower() for q in query_terms]
    
    for word in words:
        if word['text'].lower() in query_lower:
            x, y, w, h = word['x'], word['y'], word['w'], word['h']
            draw.rectangle([(x-2, y-2), (x+w+2, y+h+2)], fill=(255, 0, 0, 80), outline=(255, 0, 0, 255), width=2)
    
    return img_copy


def create_embeddings(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype('float32')


def build_faiss_index(chunks: List[DocumentChunk]) -> Tuple[faiss.Index, List[DocumentChunk]]:
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
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def hybrid_search(query: str, faiss_index, chunks: List[DocumentChunk], bm25_index: BM25Index, top_k: int = 5, vector_weight: float = 0.6) -> List[SearchResult]:
    if not chunks or faiss_index is None:
        return []
    
    query_embedding = create_embeddings([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = faiss_index.search(query_embedding, min(top_k * 2, len(chunks)))
    
    vector_scores = {int(idx): float(distances[0][i]) for i, idx in enumerate(indices[0]) if idx >= 0}
    
    bm25_results = bm25_index.search(query, top_k * 2)
    max_bm25 = max([s for _, s in bm25_results]) if bm25_results else 1.0
    bm25_scores = {idx: score / max_bm25 for idx, score in bm25_results}
    
    combined_scores = {}
    for idx in set(vector_scores.keys()) | set(bm25_scores.keys()):
        combined_scores[idx] = vector_weight * vector_scores.get(idx, 0) + (1 - vector_weight) * bm25_scores.get(idx, 0)
    
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
    return [SearchResult(chunk=chunks[idx], score=combined_scores[idx]) for idx in sorted_indices]


def build_knowledge_graph(chunks: List[DocumentChunk]) -> nx.Graph:
    G = nx.Graph()
    for chunk in chunks:
        node_id = f"{chunk.source}_p{chunk.page}_c{chunk.chunk_id}"
        G.add_node(node_id, text=chunk.text[:100], source=chunk.source, page=chunk.page)
    
    for i, c1 in enumerate(chunks):
        for c2 in chunks[i+1:]:
            if c1.source == c2.source:
                n1 = f"{c1.source}_p{c1.page}_c{c1.chunk_id}"
                n2 = f"{c2.source}_p{c2.page}_c{c2.chunk_id}"
                G.add_edge(n1, n2, weight=1.0 if c1.page == c2.page else 0.5)
    return G


def query_llm(query: str, context: str) -> str:
    system_prompt = """You are a helpful AI assistant. Answer based on the provided context. 
Cite source and page when available. Be concise."""
    
    user_message = f"""Context:
{context}

Question: {query}

Answer based on the context above."""

    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=512,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Error: {str(e)[:100]}"


def build_context(results: List[SearchResult], max_tokens: int = 2000) -> str:
    context_parts = []
    char_count = 0
    char_limit = max_tokens * 4
    
    for result in results:
        chunk = result.chunk
        chunk_text = f"[{chunk.source}, Page {chunk.page}]\n{chunk.text}\n"
        if char_count + len(chunk_text) > char_limit:
            break
        context_parts.append(chunk_text)
        char_count += len(chunk_text)
    
    return "\n---\n".join(context_parts)
