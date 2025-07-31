# Simple RAG Pipeline

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using Python. It retrieves relevant documents from a local corpus and generates answers using a language model (stubbed for simplicity).

## Structure
- `src/rag_pipeline.py`: Main pipeline implementation
- `src/corpus.txt`: Sample corpus for retrieval
- `requirements.txt`: Dependencies

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python src/rag_pipeline.py`

## Features
- Simple TF-IDF retrieval
- Basic LLM stub for generation
- Extensible for real LLMs and vector DBs
