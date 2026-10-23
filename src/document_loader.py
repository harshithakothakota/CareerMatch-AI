
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    filename = Path(file_path).name

    for document in documents:
        # PyPDFLoader normally numbers pages starting at zero.
        # Convert to human-friendly page numbers starting at one.
        page_index = document.metadata.get("page", 0)
        document.metadata["page_number"] = page_index + 1
        document.metadata["source_filename"] = filename

    return documents
