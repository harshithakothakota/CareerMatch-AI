from langchain_chroma import Chroma

COLLECTION_NAME = "careermatch_documents"


def create_vector_store(chunks, embeddings, persist_directory="chroma_db"):
    """Create a fresh, isolated Chroma index for one conversation."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Replace this collection only inside the specified conversation directory.
    try:
        vector_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    if chunks:
        vector_store.add_documents(chunks)
    return vector_store


def load_vector_store(embeddings, persist_directory="chroma_db"):
    """Load the Chroma index from the requested directory."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
