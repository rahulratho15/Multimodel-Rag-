---
title: Multimodal RAG System
emoji: brain
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Multimodal RAG System

A lightweight multimodal RAG system with visual grounding for document Q&A.

## Features

- PDF, TXT, DOCX, and Markdown document processing
- Image OCR with visual grounding
- Hybrid search (FAISS + BM25)
- LLM-powered answers via Llama 3

## Supported Files

| Type | Formats |
|------|---------|
| Documents | PDF, TXT, MD, DOCX |
| Images | PNG, JPG, JPEG, WebP |

## Setup

Add `HF_TOKEN` as a secret in Space Settings.

## Local Dev

```bash
pip install -r requirements.txt
export HF_TOKEN="your_token"
python app.py
```
