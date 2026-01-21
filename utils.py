"""
Utility functions for Multimodal RAG Application
Handles: PDF extraction, OCR, Audio processing, Embeddings, and Visual Annotations
"""

import os
import re
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF - faster than pdf2image for page rendering
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import networkx as nx
import edge_tts
from huggingface_hub import InferenceClient

# ============================================================================
# CONFIGURATION
# ============================================================================

# HF Token should be set as environment variable HF_TOKEN
# In Hugging Face Spaces, add it as a Secret in Settings > Repository secrets

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
WHISPER_MODEL = "openai/whisper-tiny"  # HF transformers model
TTS_VOICE = "en-US-AriaNeural"  # Natural sounding voice
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Flag to check if pytesseract is available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    # Test if tesseract binary is available
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    print("Warning: Tesseract OCR not available. Image text extraction will be limited.")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata"""
    text: str
    source: str
    page: int = 0
    chunk_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    embedding: Optional[np.ndarray] = None

@dataclass
class OCRResult:
    """Result from OCR processing with bounding boxes"""
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[Image.Image] = None

@dataclass 
class SearchResult:
    """Result from hybrid search"""
    chunk: DocumentChunk
    score: float
    source_image: Optional[Image.Image] = None

# ============================================================================
# SINGLETON MODEL LOADERS (Lazy Loading for Memory Efficiency)
# ============================================================================

_embedding_model = None
_whisper_model = None
_inference_client = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_whisper_model():
    """Lazy load Whisper model using transformers pipeline"""
    global _whisper_model
    if _whisper_model is None:
        try:
            from transformers import pipeline
            _whisper_model = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL,
                device="cpu"
            )
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            _whisper_model = None
    return _whisper_model

def get_inference_client() -> InferenceClient:
    """Get HF Inference Client"""
    global _inference_client
    if _inference_client is None:
        token = os.environ.get("HF_TOKEN", "")
        _inference_client = InferenceClient(api_key=token)
    return _inference_client

# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_pdf_text(pdf_path: str) -> List[DocumentChunk]:
    """
    Extract text from PDF with page tracking and chunking.
    Returns list of DocumentChunk objects.
    """
    chunks = []
    
    try:
        reader = PdfReader(pdf_path)
        filename = Path(pdf_path).name
        chunk_id = 0
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            
            # Clean and chunk the text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into chunks with overlap
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
        print(f"Error extracting PDF: {e}")
        
    return chunks

def pdf_page_to_image(pdf_path: str, page_num: int) -> Optional[Image.Image]:
    """
    Convert a specific PDF page to PIL Image.
    Uses PyMuPDF (fitz) for better performance on CPU.
    """
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc[page_num - 1]
            # Render at 150 DPI for good quality without being too large
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        doc.close()
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
    return None

# ============================================================================
# OCR PROCESSING
# ============================================================================

def ocr_with_boxes(image_path: str) -> OCRResult:
    """
    Perform OCR on an image and return text with bounding box coordinates.
    Falls back to basic image loading if Tesseract is not available.
    """
    try:
        img = Image.open(image_path)
        
        # Check if tesseract is available
        if not TESSERACT_AVAILABLE:
            print("Tesseract not available, returning image without OCR")
            return OCRResult(text="[Image uploaded - OCR not available]", words=[], image=img)
        
        # Get detailed OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        words = []
        full_text_parts = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf > 30:  # Filter low confidence
                words.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'conf': conf
                })
                full_text_parts.append(text)
        
        full_text = ' '.join(full_text_parts)
        
        return OCRResult(text=full_text, words=words, image=img)
        
    except Exception as e:
        print(f"OCR Error: {e}")
        # Try to at least return the image
        try:
            img = Image.open(image_path)
            return OCRResult(text="", words=[], image=img)
        except:
            return OCRResult(text="", words=[], image=None)

def create_image_chunks(ocr_result: OCRResult, image_path: str) -> List[DocumentChunk]:
    """
    Create document chunks from OCR result with bounding box info.
    """
    chunks = []
    filename = Path(image_path).name
    
    if not ocr_result.text:
        return chunks
    
    # Group words into lines/paragraphs based on y-coordinate
    if ocr_result.words:
        words = ocr_result.words
        text = ocr_result.text
        
        # Create chunks from the full text
        word_list = text.split()
        for i in range(0, len(word_list), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = ' '.join(word_list[i:i + CHUNK_SIZE])
            if chunk_text.strip():
                # Find bounding box for this chunk (approximate)
                chunk_words = chunk_text.split()[:10]  # First 10 words
                matching_boxes = []
                for w in words:
                    if w['text'] in chunk_words:
                        matching_boxes.append(w)
                
                # Calculate encompassing bbox
                if matching_boxes:
                    x = min(w['x'] for w in matching_boxes)
                    y = min(w['y'] for w in matching_boxes)
                    x2 = max(w['x'] + w['w'] for w in matching_boxes)
                    y2 = max(w['y'] + w['h'] for w in matching_boxes)
                    bbox = (x, y, x2 - x, y2 - y)
                else:
                    bbox = None
                
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=filename,
                    page=1,
                    chunk_id=len(chunks),
                    bbox=bbox
                ))
    
    return chunks

# ============================================================================
# VISUAL ANNOTATION
# ============================================================================

def draw_bounding_box(
    image: Image.Image, 
    bbox: Tuple[int, int, int, int],
    color: str = "red",
    width: int = 3,
    label: str = ""
) -> Image.Image:
    """
    Draw a bounding box on the image.
    bbox: (x, y, width, height)
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x, y, w, h = bbox
    # Draw rectangle
    draw.rectangle(
        [(x, y), (x + w, y + h)],
        outline=color,
        width=width
    )
    
    # Add label if provided
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw label background
        text_bbox = draw.textbbox((x, y - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 20), label, fill="white", font=font)
    
    return img_copy

def highlight_text_region(
    image: Image.Image,
    words: List[Dict[str, Any]],
    query_terms: List[str]
) -> Image.Image:
    """
    Highlight regions in image that match query terms.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    query_lower = [q.lower() for q in query_terms]
    
    for word in words:
        if word['text'].lower() in query_lower:
            x, y, w, h = word['x'], word['y'], word['w'], word['h']
            # Semi-transparent highlight
            draw.rectangle(
                [(x - 2, y - 2), (x + w + 2, y + h + 2)],
                fill=(255, 0, 0, 80),
                outline=(255, 0, 0, 255),
                width=2
            )
    
    return img_copy

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using HuggingFace API (no ffmpeg needed).
    """
    try:
        from huggingface_hub import InferenceClient
        import os
        
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            return "[Audio transcription unavailable - HF_TOKEN not set]"
        
        client = InferenceClient(api_key=token)
        
        # Read audio file and send to HF API
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # Use HF automatic speech recognition
        result = client.automatic_speech_recognition(
            audio=audio_data,
            model="openai/whisper-tiny"
        )
        
        if isinstance(result, dict):
            return result.get("text", "").strip()
        elif hasattr(result, 'text'):
            return result.text.strip()
        return str(result).strip()
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[Transcription failed: {str(e)[:100]}]"

async def generate_speech_async(text: str, output_path: str) -> str:
    """
    Generate speech from text using Edge-TTS.
    Returns path to the generated audio file.
    """
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        return ""

def generate_speech(text: str, output_path: str = None) -> str:
    """
    Synchronous wrapper for TTS generation.
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(generate_speech_async(text, output_path))

# ============================================================================
# EMBEDDING & VECTOR STORE
# ============================================================================

def create_embeddings(texts: List[str]) -> np.ndarray:
    """
    Create embeddings for a list of texts.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype('float32')

def build_faiss_index(chunks: List[DocumentChunk]) -> Tuple[faiss.Index, List[DocumentChunk]]:
    """
    Build FAISS index from document chunks.
    Returns the index and the chunks list for retrieval.
    """
    if not chunks:
        return None, []
    
    texts = [c.text for c in chunks]
    embeddings = create_embeddings(texts)
    
    # Store embeddings in chunks
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, chunks

# ============================================================================
# BM25 SEARCH
# ============================================================================

class BM25Index:
    """BM25 keyword search index"""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search and return top-k results with indices and scores"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        return results

# ============================================================================
# HYBRID SEARCH
# ============================================================================

def hybrid_search(
    query: str,
    faiss_index: faiss.Index,
    chunks: List[DocumentChunk],
    bm25_index: BM25Index,
    top_k: int = 5,
    vector_weight: float = 0.6
) -> List[SearchResult]:
    """
    Perform hybrid search combining FAISS (semantic) and BM25 (keyword).
    """
    if not chunks or faiss_index is None:
        return []
    
    # Vector search
    query_embedding = create_embeddings([query])
    faiss.normalize_L2(query_embedding)
    
    distances, indices = faiss_index.search(query_embedding, min(top_k * 2, len(chunks)))
    
    # Create score dict for vector results
    vector_scores = {}
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            vector_scores[int(idx)] = float(distances[0][i])
    
    # BM25 search - now returns (index, score) tuples
    bm25_results = bm25_index.search(query, top_k * 2)
    
    # Normalize BM25 scores
    max_bm25 = max([s for _, s in bm25_results]) if bm25_results else 1.0
    bm25_scores = {}
    for idx, score in bm25_results:
        bm25_scores[idx] = score / max_bm25
    
    # Combine scores
    combined_scores = {}
    all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
    
    for idx in all_indices:
        v_score = vector_scores.get(idx, 0)
        b_score = bm25_scores.get(idx, 0)
        combined_scores[idx] = vector_weight * v_score + (1 - vector_weight) * b_score
    
    # Sort and get top-k
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
    
    results = []
    for idx in sorted_indices:
        results.append(SearchResult(
            chunk=chunks[idx],
            score=combined_scores[idx]
        ))
    
    return results

# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

def build_knowledge_graph(chunks: List[DocumentChunk]) -> nx.Graph:
    """
    Build a simple knowledge graph linking chunks by source document.
    """
    G = nx.Graph()
    
    # Add nodes for each chunk
    for chunk in chunks:
        node_id = f"{chunk.source}_p{chunk.page}_c{chunk.chunk_id}"
        G.add_node(node_id, 
                   text=chunk.text[:100],
                   source=chunk.source,
                   page=chunk.page)
    
    # Link chunks from same document
    for i, chunk1 in enumerate(chunks):
        for j, chunk2 in enumerate(chunks[i+1:], i+1):
            if chunk1.source == chunk2.source:
                node1 = f"{chunk1.source}_p{chunk1.page}_c{chunk1.chunk_id}"
                node2 = f"{chunk2.source}_p{chunk2.page}_c{chunk2.chunk_id}"
                # Same page = stronger connection
                weight = 1.0 if chunk1.page == chunk2.page else 0.5
                G.add_edge(node1, node2, weight=weight)
    
    return G

def find_related_chunks(
    graph: nx.Graph,
    chunk: DocumentChunk,
    max_depth: int = 2
) -> List[str]:
    """
    Find related chunks using graph traversal.
    """
    node_id = f"{chunk.source}_p{chunk.page}_c{chunk.chunk_id}"
    
    if node_id not in graph:
        return []
    
    # BFS to find neighbors
    related = []
    visited = {node_id}
    queue = [(node_id, 0)]
    
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
            
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                related.append(neighbor)
                queue.append((neighbor, depth + 1))
    
    return related

# ============================================================================
# LLM QUERY
# ============================================================================

def query_llm(
    query: str,
    context: str,
    system_prompt: str = None
) -> str:
    """
    Query the LLM using HF Router with OpenAI-compatible API.
    Uses meta-llama/Meta-Llama-3-8B-Instruct via HF Router.
    """
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Always cite the source document and page number when available.
If the context doesn't contain relevant information, say so honestly.
Keep responses concise and informative."""
    
    user_message = f"""Context:
{context}

Question: {query}

Please provide a helpful answer based on the context above. If citing sources, mention the document name and page number."""

    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=512,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I apologize, but I encountered an error processing your request: {str(e)}"

# ============================================================================
# CONTEXT BUILDER
# ============================================================================

def build_context(results: List[SearchResult], max_tokens: int = 2000) -> str:
    """
    Build context string from search results.
    """
    context_parts = []
    char_count = 0
    char_limit = max_tokens * 4  # Rough estimate
    
    for result in results:
        chunk = result.chunk
        source_info = f"[Source: {chunk.source}, Page {chunk.page}]"
        chunk_text = f"{source_info}\n{chunk.text}\n"
        
        if char_count + len(chunk_text) > char_limit:
            break
            
        context_parts.append(chunk_text)
        char_count += len(chunk_text)
    
    return "\n---\n".join(context_parts)
