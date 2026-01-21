import os
from pathlib import Path
import gradio as gr

from utils import (
    extract_pdf_text, pdf_page_to_image, ocr_with_boxes,
    build_faiss_index, BM25Index, hybrid_search,
    build_knowledge_graph, query_llm, build_context, DocumentChunk
)

# Global state
chunks = []
faiss_index = None
bm25_index = None
source_images = {}
uploaded_files = []

def process_files(files):
    global chunks, faiss_index, bm25_index, source_images, uploaded_files
    
    if not files:
        return "No files uploaded", "None"
    
    processed = []
    
    for f in files:
        try:
            name = Path(f).name
            ext = Path(f).suffix.lower()
            
            if ext == '.pdf':
                new_chunks = extract_pdf_text(f)
                if new_chunks:
                    chunks.extend(new_chunks)
                    for p in range(1, min(10, len(new_chunks) + 1)):
                        img = pdf_page_to_image(f, p)
                        if img:
                            source_images[f"{name}_page_{p}"] = img
                    processed.append(name)
                    uploaded_files.append(name)
                    
            elif ext in ['.txt', '.md']:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                if text.strip():
                    words = text.split()
                    for i in range(0, len(words), 450):
                        chunk_text = ' '.join(words[i:i+500])
                        if chunk_text.strip():
                            chunks.append(DocumentChunk(text=chunk_text, source=name, page=1, chunk_id=len(chunks)))
                    processed.append(name)
                    uploaded_files.append(name)
                    
            elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
                ocr = ocr_with_boxes(f)
                if ocr.image:
                    source_images[name] = ocr.image
                    processed.append(name)
                    uploaded_files.append(name)
        except Exception as e:
            print(f"Error: {e}")
    
    # Build indices
    if chunks:
        faiss_index, _ = build_faiss_index(chunks)
        bm25_index = BM25Index(chunks)
    
    status = f"Loaded: {', '.join(processed)}" if processed else "No files processed"
    files_str = ", ".join(uploaded_files) if uploaded_files else "None"
    return status, files_str

def clear_all():
    global chunks, faiss_index, bm25_index, source_images, uploaded_files
    chunks = []
    faiss_index = None
    bm25_index = None
    source_images = {}
    uploaded_files = []
    return "Cleared", "None", [], None, ""

def ask_question(question, history):
    global chunks, faiss_index, bm25_index, source_images
    
    if not question or not question.strip():
        return history, None, "Enter a question"
    
    if not chunks:
        history = history or []
        history.append((question, "Please upload documents first."))
        return history, None, "No documents"
    
    results = hybrid_search(question, faiss_index, chunks, bm25_index, top_k=5)
    
    if not results:
        history = history or []
        history.append((question, "No relevant information found."))
        return history, None, "No results"
    
    context = build_context(results)
    answer = query_llm(question, context)
    
    sources = [f"{r.chunk.source} p{r.chunk.page}" for r in results[:3]]
    
    # Get source image
    img = None
    top = results[0]
    key = f"{top.chunk.source}_page_{top.chunk.page}"
    if key in source_images:
        img = source_images[key]
    elif top.chunk.source in source_images:
        img = source_images[top.chunk.source]
    
    history = history or []
    history.append((question, answer))
    
    return history, img, " | ".join(sources)

# Build UI
with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# 🧠 Multimodal RAG System\nUpload documents and ask questions")
    
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Upload Files", file_count="multiple", file_types=[".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"])
            proc_btn = gr.Button("Process", variant="primary")
            clr_btn = gr.Button("Clear All")
            status = gr.Textbox(label="Status", interactive=False)
            loaded = gr.Textbox(label="Loaded Files", interactive=False, value="None")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            question = gr.Textbox(label="Your Question", placeholder="Ask something...")
            ask_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            source_img = gr.Image(label="Source Document")
            sources_txt = gr.Textbox(label="Sources", interactive=False)
    
    # Events
    proc_btn.click(process_files, [files], [status, loaded])
    clr_btn.click(clear_all, [], [status, loaded, chatbot, source_img, sources_txt])
    
    ask_btn.click(ask_question, [question, chatbot], [chatbot, source_img, sources_txt]).then(lambda: "", None, question)
    question.submit(ask_question, [question, chatbot], [chatbot, source_img, sources_txt]).then(lambda: "", None, question)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
