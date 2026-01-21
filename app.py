import os
from pathlib import Path
from typing import List, Dict, Any
import gradio as gr

from utils import (
    extract_pdf_text, pdf_page_to_image, ocr_with_boxes, create_image_chunks,
    highlight_text_region, build_faiss_index, BM25Index, hybrid_search,
    build_knowledge_graph, query_llm, build_context, DocumentChunk, OCRResult
)

class RAGState:
    def __init__(self):
        self.chunks = []
        self.faiss_index = None
        self.bm25_index = None
        self.knowledge_graph = None
        self.source_images = {}
        self.ocr_data = {}
        self.uploaded_files = []
    
    def reset(self):
        self.__init__()
    
    def add_chunks(self, new_chunks):
        self.chunks.extend(new_chunks)
        if self.chunks:
            self.faiss_index, self.chunks = build_faiss_index(self.chunks)
            self.bm25_index = BM25Index(self.chunks)
            self.knowledge_graph = build_knowledge_graph(self.chunks)

state = RAGState()

def process_files(files):
    if not files:
        return "No files", "None"
    processed = []
    for f in files:
        try:
            name = Path(f).name
            ext = Path(f).suffix.lower()
            if ext == '.pdf':
                chunks = extract_pdf_text(f)
                if chunks:
                    state.add_chunks(chunks)
                    for p in range(1, min(10, len(chunks) + 1)):
                        img = pdf_page_to_image(f, p)
                        if img:
                            state.source_images[f"{name}_page_{p}"] = img
                    processed.append(name)
                    state.uploaded_files.append(name)
            elif ext in ['.txt', '.md']:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                if text.strip():
                    words = text.split()
                    chunks = []
                    for i in range(0, len(words), 450):
                        chunk_text = ' '.join(words[i:i+500])
                        if chunk_text.strip():
                            chunks.append(DocumentChunk(text=chunk_text, source=name, page=1, chunk_id=len(chunks)))
                    state.add_chunks(chunks)
                    processed.append(name)
                    state.uploaded_files.append(name)
            elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
                ocr = ocr_with_boxes(f)
                if ocr.image:
                    state.source_images[name] = ocr.image
                    processed.append(name)
                    state.uploaded_files.append(name)
        except:
            pass
    status = f"Loaded: {', '.join(processed)}" if processed else "Failed"
    files_str = ", ".join(state.uploaded_files) if state.uploaded_files else "None"
    return status, files_str

def clear_docs():
    state.reset()
    return "Cleared", "None"

def ask(question, history):
    if history is None:
        history = []
    if not question or not question.strip():
        return history, None, ""
    if not state.chunks:
        history.append([question, "Please upload documents first."])
        return history, None, "No docs"
    results = hybrid_search(question, state.faiss_index, state.chunks, state.bm25_index, top_k=5)
    if not results:
        history.append([question, "No relevant info found."])
        return history, None, "No results"
    context = build_context(results)
    answer = query_llm(question, context)
    sources = [f"{r.chunk.source} p{r.chunk.page}" for r in results[:3]]
    img = None
    top = results[0]
    key = f"{top.chunk.source}_page_{top.chunk.page}"
    if key in state.source_images:
        img = state.source_images[key]
    elif top.chunk.source in state.source_images:
        img = state.source_images[top.chunk.source]
    history.append([question, answer])
    return history, img, " | ".join(sources)

with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# Multimodal RAG System\nUpload documents and ask questions")
    
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Upload", file_count="multiple", file_types=[".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"])
            with gr.Row():
                proc_btn = gr.Button("Process", variant="primary")
                clr_btn = gr.Button("Clear")
            status = gr.Textbox(label="Status", interactive=False)
            loaded = gr.Textbox(label="Files", interactive=False, value="None")
        
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Chat", height=350)
            question = gr.Textbox(label="Question", placeholder="Ask...")
            ask_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            img_out = gr.Image(label="Source")
            src_info = gr.Textbox(label="Sources", interactive=False)
    
    proc_btn.click(process_files, [files], [status, loaded])
    clr_btn.click(clear_docs, [], [status, loaded])
    
    def submit(q, h):
        h2, im, src = ask(q, h)
        return h2, im, src, ""
    
    ask_btn.click(submit, [question, chat], [chat, img_out, src_info, question])
    question.submit(submit, [question, chat], [chat, img_out, src_info, question])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
