# CareerMatch AI

CareerMatch AI is a local resume and job-description assistant built with Streamlit, LangChain, ChromaDB, Hugging Face sentence-transformer embeddings, and Ollama.

## Features

- Upload a resume PDF and a job-description PDF.
- Ask questions grounded in retrieved document text.
- Compare documented resume evidence with job requirements.
- Generate role-specific interview questions.
- Keep separate saved conversations locally.

## Requirements

- Python 3.10 or newer (developed with Python 3.12).
- Git.
- Ollama installed locally.
- The `llama3.2:1b` Ollama model.

## Setup (Windows PowerShell)

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Download the model:

   ```powershell
   ollama pull llama3.2:1b
   ```

3. Open a terminal in the project directory and create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install Python dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

6. Open the local URL shown by Streamlit, upload a resume and job-description PDF, then try **Ask AI**, **Skill Gap**, and **Interview Prep**.

## Data and Privacy

Uploaded PDFs, session history, and the local Chroma index may be stored under `data/` or other local runtime folders. These should remain local and must not be committed to GitHub.

Do not commit private resumes, job descriptions, `.env` files, database files, virtual environments, or local vector indexes.

## Current Limitations

- The app currently accepts PDF uploads.
- AI-generated analysis can make mistakes. Verify extracted facts and skill-gap results against the original documents before relying on them.
- The local Ollama model must be installed and running.
- Dependency versions are not pinned yet; pin tested versions before a production deployment.

## Project Structure

```text
CareerMatch-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── chunking.py
    ├── document_loader.py
    ├── embeddings.py
    ├── history.py
    ├── rag_pipeline.py
    ├── session_manager.py
    └── vector_store.py
```