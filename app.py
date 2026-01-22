import streamlit as st
from pathlib import Path
import tempfile
from utils import extract_pdf, pdf_to_image, build_index, BM25, search, ask_llm, get_context, Chunk

st.set_page_config(page_title="Multimodal RAG", page_icon="🧠", layout="wide")

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.bm25 = None
    st.session_state.images = {}
    st.session_state.files = []
    st.session_state.messages = []

st.title("🧠 Multimodal RAG System")
st.markdown("Upload documents and ask questions")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("📁 Upload")
    files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if st.button("Process Files", type="primary"):
        if files:
            st.session_state.chunks = []
            st.session_state.images = {}
            st.session_state.files = []
            
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                
                if f.name.endswith('.pdf'):
                    chunks = extract_pdf(tmp_path)
                    st.session_state.chunks.extend(chunks)
                    for i in range(1, min(10, len(chunks)+1)):
                        img = pdf_to_image(tmp_path, i)
                        if img:
                            st.session_state.images[f"{f.name}_page_{i}"] = img
                else:
                    text = f.getvalue().decode('utf-8', errors='ignore')
                    words = text.split()
                    for i in range(0, len(words), 400):
                        t = ' '.join(words[i:i+450])
                        if t.strip():
                            st.session_state.chunks.append(Chunk(text=t, source=f.name, page=1))
                
                st.session_state.files.append(f.name)
            
            if st.session_state.chunks:
                st.session_state.index, st.session_state.chunks = build_index(st.session_state.chunks)
                st.session_state.bm25 = BM25(st.session_state.chunks)
            
            st.success(f"Loaded {len(st.session_state.files)} files, {len(st.session_state.chunks)} chunks")
    
    if st.button("Clear All"):
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.bm25 = None
        st.session_state.images = {}
        st.session_state.files = []
        st.session_state.messages = []
        st.rerun()
    
    st.subheader("📄 Loaded Files")
    if st.session_state.files:
        for f in st.session_state.files:
            st.text(f"• {f}")
    else:
        st.text("No files loaded")

with col2:
    st.subheader("💬 Chat")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        if not st.session_state.chunks:
            response = "Please upload documents first."
            st.session_state.current_image = None
            st.session_state.current_sources = ""
        else:
            results = search(prompt, st.session_state.index, st.session_state.chunks, st.session_state.bm25)
            if results:
                context = get_context(results)
                response = ask_llm(prompt, context)
                st.session_state.current_sources = " | ".join([f"{r.chunk.source} p{r.chunk.page}" for r in results[:3]])
                
                # Get source image
                top = results[0]
                key = f"{top.chunk.source}_page_{top.chunk.page}"
                st.session_state.current_image = st.session_state.images.get(key)
            else:
                response = "No relevant information found."
                st.session_state.current_image = None
                st.session_state.current_sources = ""
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.write(response)

with col3:
    st.subheader("📷 Source")
    
    if hasattr(st.session_state, 'current_image') and st.session_state.current_image:
        st.image(st.session_state.current_image, caption="Source Document")
    else:
        st.info("Source will appear here")
    
    st.subheader("📚 Sources")
    if hasattr(st.session_state, 'current_sources') and st.session_state.current_sources:
        st.text(st.session_state.current_sources)
