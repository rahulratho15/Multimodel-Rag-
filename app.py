"""
Multimodal RAG Application
A lightweight, Python-only RAG system with visual grounding.
Designed for Hugging Face Spaces (Free CPU Tier)
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import gradio as gr

from utils import (
    extract_pdf_text,
    pdf_page_to_image,
    ocr_with_boxes,
    create_image_chunks,
    highlight_text_region,
    build_faiss_index,
    BM25Index,
    hybrid_search,
    build_knowledge_graph,
    query_llm,
    build_context,
    DocumentChunk,
    OCRResult
)


class RAGState:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self.bm25_index = None
        self.knowledge_graph = None
        self.source_images: Dict[str, Any] = {}
        self.ocr_data: Dict[str, OCRResult] = {}
        self.uploaded_files: List[str] = []
        
    def reset(self):
        self.__init__()
        
    def add_chunks(self, new_chunks: List[DocumentChunk]):
        self.chunks.extend(new_chunks)
        if self.chunks:
            self.faiss_index, self.chunks = build_faiss_index(self.chunks)
            self.bm25_index = BM25Index(self.chunks)
            self.knowledge_graph = build_knowledge_graph(self.chunks)


rag_state = RAGState()


def process_files(files):
    if not files:
        return "No files uploaded", "None"
    
    processed = []
    
    for file_path in files:
        try:
            filename = Path(file_path).name
            suffix = Path(file_path).suffix.lower()
            
            if suffix == '.pdf':
                chunks = extract_pdf_text(file_path)
                if chunks:
                    rag_state.add_chunks(chunks)
                    for page_num in range(1, min(10, len(chunks) + 1)):
                        img = pdf_page_to_image(file_path, page_num)
                        if img:
                            rag_state.source_images[f"{filename}_page_{page_num}"] = img
                    processed.append(filename)
                    rag_state.uploaded_files.append(filename)
                    
            elif suffix in ['.png', '.jpg', '.jpeg', '.webp']:
                ocr_result = ocr_with_boxes(file_path)
                if ocr_result.image:
                    if ocr_result.text and not ocr_result.text.startswith("["):
                        chunks = create_image_chunks(ocr_result, file_path)
                        rag_state.add_chunks(chunks)
                    rag_state.source_images[filename] = ocr_result.image
                    rag_state.ocr_data[filename] = ocr_result
                    processed.append(filename)
                    rag_state.uploaded_files.append(filename)
                    
            elif suffix in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if text.strip():
                    words = text.split()
                    chunks = []
                    for i in range(0, len(words), 450):
                        chunk_text = ' '.join(words[i:i + 500])
                        if chunk_text.strip():
                            chunks.append(DocumentChunk(
                                text=chunk_text, source=filename, page=1, chunk_id=len(chunks)
                            ))
                    rag_state.add_chunks(chunks)
                    processed.append(filename)
                    rag_state.uploaded_files.append(filename)
                    
            elif suffix in ['.docx']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = '\n'.join([p.text for p in doc.paragraphs])
                    if text.strip():
                        words = text.split()
                        chunks = []
                        for i in range(0, len(words), 450):
                            chunk_text = ' '.join(words[i:i + 500])
                            if chunk_text.strip():
                                chunks.append(DocumentChunk(
                                    text=chunk_text, source=filename, page=1, chunk_id=len(chunks)
                                ))
                        rag_state.add_chunks(chunks)
                        processed.append(filename)
                        rag_state.uploaded_files.append(filename)
                except:
                    pass
        except:
            pass
    
    status = f"Loaded: {', '.join(processed)}" if processed else "No files processed"
    files_list = ", ".join(rag_state.uploaded_files) if rag_state.uploaded_files else "None"
    return status, files_list


def clear_docs():
    rag_state.reset()
    return "Cleared", "None"


def ask_question(question, history):
    if history is None:
        history = []
    
    if not question or not question.strip():
        return history, None, "Type a question"
    
    if not rag_state.chunks:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "Please upload documents first."})
        return history, None, "No docs"
    
    results = hybrid_search(
        query=question,
        faiss_index=rag_state.faiss_index,
        chunks=rag_state.chunks,
        bm25_index=rag_state.bm25_index,
        top_k=5
    )
    
    if not results:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "No relevant info found."})
        return history, None, "No results"
    
    context = build_context(results)
    answer = query_llm(question, context)
    
    sources = [f"{r.chunk.source} p{r.chunk.page}" for r in results[:3]]
    
    # Visual proof
    img = None
    top = results[0]
    key = f"{top.chunk.source}_page_{top.chunk.page}"
    if key in rag_state.source_images:
        img = rag_state.source_images[key].copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([(3, 3), (img.width-3, img.height-3)], outline="red", width=3)
    elif top.chunk.source in rag_state.source_images:
        src_img = rag_state.source_images[top.chunk.source]
        if src_img and top.chunk.source in rag_state.ocr_data:
            img = highlight_text_region(src_img, rag_state.ocr_data[top.chunk.source].words, question.lower().split())
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    return history, img, " | ".join(sources)


# Simple responsive CSS
CSS = """
* { box-sizing: border-box; }
.gradio-container { max-width: 100% !important; padding: 10px !important; }
#header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 15px; text-align: center; }
#header h1 { color: white; margin: 0; font-size: 1.5em; }
#header p { color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 0.9em; }
.contain { display: flex !important; flex-direction: column !important; }
#chat-col { min-height: 400px; }
@media (max-width: 768px) {
    .wrap { flex-direction: column !important; }
    #header h1 { font-size: 1.2em; }
}
"""

with gr.Blocks(css=CSS, title="RAG System") as demo:
    
    gr.HTML('<div id="header"><h1>Multimodal RAG</h1><p>Upload docs and ask questions</p></div>')
    
    with gr.Row(elem_classes="wrap"):
        # Left column
        with gr.Column(scale=1, min_width=250):
            files = gr.File(label="Upload Files", file_count="multiple", 
                           file_types=[".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".webp"])
            with gr.Row():
                proc_btn = gr.Button("Process", variant="primary", size="sm")
                clr_btn = gr.Button("Clear", size="sm")
            status = gr.Textbox(label="Status", lines=2, interactive=False)
            loaded = gr.Textbox(label="Files", lines=1, interactive=False, value="None")
        
        # Middle column - Chat
        with gr.Column(scale=2, min_width=300, elem_id="chat-col"):
            chat = gr.Chatbot(label="Chat", height=350)
            with gr.Row():
                question = gr.Textbox(label="Your Question", placeholder="Ask something...", scale=4)
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
        
        # Right column
        with gr.Column(scale=1, min_width=200):
            img_out = gr.Image(label="Source", height=200)
            src_info = gr.Textbox(label="Sources", lines=1, interactive=False)
    
    gr.HTML('<p style="text-align:center;color:#666;font-size:0.8em;margin-top:10px;">Powered by Llama 3</p>')
    
    # Events
    proc_btn.click(process_files, [files], [status, loaded])
    clr_btn.click(clear_docs, [], [status, loaded])
    
    def submit(q, h):
        hist, im, src = ask_question(q, h)
        return hist, im, src, ""
    
    ask_btn.click(submit, [question, chat], [chat, img_out, src_info, question])
    question.submit(submit, [question, chat], [chat, img_out, src_info, question])


if __name__ == "__main__":
    Path("uploads").mkdir(exist_ok=True)
    if not os.environ.get("HF_TOKEN"):
        print("Set HF_TOKEN: $env:HF_TOKEN='your_token'")
    demo.launch(server_name="127.0.0.1", server_port=7860)
