import os
import tempfile
from pathlib import Path

import streamlit as st

from src.document_loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, load_vector_store
from src.rag_pipeline import (
    retrieve_documents,
    generate_answer,
    generate_skill_gap_analysis,
    generate_interview_questions,
)
from src.session_manager import (
    init_db,
    create_session,
    save_document,
    get_session,
    list_sessions,
    add_message,
    get_messages,
)

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"

st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 38px; font-weight: 750; margin-bottom: 0;}
    .subtitle {font-size: 17px; color: #888; margin-bottom: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()



def get_active_session_id():
    return st.query_params.get("session_id")


def set_active_session(session_id):
    st.query_params["session_id"] = session_id


def start_new_chat():
    st.query_params.clear()
    st.session_state.pop("skill_gap_result", None)
    st.session_state.pop("interview_result", None)
    st.rerun()


def session_index_path(session_id):
    return str(SESSIONS_DIR / session_id / "chroma_db")


def load_active_store(session_id):
    index_path = Path(session_index_path(session_id))
    if not index_path.exists():
        return None
    try:
        embeddings=get_embeddings()
        store = load_vector_store(embeddings, persist_directory=str(index_path))
        if store._collection.count() > 0:
            return store
    except Exception:
        return None
    return None


# Query parameters make the active conversation recoverable after a refresh.
active_id = get_active_session_id()
active_session = get_session(active_id) if active_id else None
if active_id and active_session is None:
    st.query_params.clear()
    active_id = None

if "active_store" not in st.session_state:
    st.session_state.active_store = None
if "loaded_store_session_id" not in st.session_state:
    st.session_state.loaded_store_session_id = None

if active_id and st.session_state.loaded_store_session_id != active_id:
    st.session_state.active_store = load_active_store(active_id)
    st.session_state.loaded_store_session_id = active_id
elif not active_id:
    st.session_state.active_store = None
    st.session_state.loaded_store_session_id = None

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🚀 CareerMatch AI")
    st.caption("Your resume and job-description assistant.")

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        start_new_chat()

    st.divider()
    st.markdown("### Previous chats")

    sessions = list_sessions()
    if not sessions:
        st.caption("Your saved conversations will appear here.")

    for item in sessions:
        title = item.get("title") or "Career conversation"
        if len(title) > 38:
            title = title[:35] + "..."
        files_line = f"{item.get('resume_name') or 'No resume'} · {item.get('jd_name') or 'No job description'}"
        is_active = item["id"] == active_id
        label = ("● " if is_active else "") + title
        if st.button(label, key=f"open_{item['id']}", use_container_width=True):
            set_active_session(item["id"])
            st.session_state.pop("skill_gap_result", None)
            st.session_state.pop("interview_result", None)
            st.rerun()
        st.caption(files_line)

# ----------------------------- Header ------------------------------
st.markdown('<div class="main-title">🚀 CareerMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Resume & Job Description Assistant</div>', unsafe_allow_html=True)

if active_session:
    st.markdown(f"### 💬 {active_session.get('title', 'Previous conversation')}")
    names = []
    if active_session.get("resume_name"):
        names.append(f"**Resume:** {active_session['resume_name']}")
    if active_session.get("jd_name"):
        names.append(f"**Job description:** {active_session['jd_name']}")
    if names:
        st.caption("  ·  ".join(names))
else:
    st.markdown("### ✨ New Chat")
    st.write("Upload a resume and a job description to start a separate, saved conversation.")

st.divider()

# ----------------------------- New upload --------------------------
if not active_session:
    st.subheader("📄 Upload documents")
    col1, col2 = st.columns(2)
    with col1:
        resume_upload = st.file_uploader(
            "Upload resume (PDF)", type=["pdf"], key="new_resume_upload"
        )
    with col2:
        jd_upload = st.file_uploader(
            "Upload job description (PDF)", type=["pdf"], key="new_jd_upload"
        )

    if resume_upload:
        st.success(f"Resume selected: {resume_upload.name}")
    if jd_upload:
        st.success(f"Job description selected: {jd_upload.name}")

    if st.button(
        "🚀 Start conversation",
        type="primary",
        use_container_width=True,
        disabled=not (resume_upload and jd_upload),
    ):
        with st.spinner("Saving PDFs and preparing a separate document index..."):
            new_id = create_session(resume_upload.name, jd_upload.name)
            save_document(new_id, "resume", resume_upload.name, resume_upload.getvalue())
            save_document(new_id, "jd", jd_upload.name, jd_upload.getvalue())

            session_folder = SESSIONS_DIR / new_id
            resume_path = session_folder / "resume.pdf"
            jd_path = session_folder / "jd.pdf"
            resume_documents = load_pdf(str(resume_path))
            jd_documents = load_pdf(str(jd_path))

            resume_chunks = split_documents(resume_documents)
            jd_chunks = split_documents(jd_documents)

            for chunk in resume_chunks:
                chunk.metadata["document_type"] = "Resume"
                chunk.metadata["source_filename"] = resume_upload.name

            for chunk in jd_chunks:
                chunk.metadata["document_type"] = "Job Description"
                chunk.metadata["source_filename"] = jd_upload.name

            all_chunks = resume_chunks + jd_chunks
            index_path = session_index_path(new_id)
            os.makedirs(index_path, exist_ok=True)
            embeddings=get_embeddings()
            create_vector_store(
                all_chunks,
                embeddings,
                persist_directory=index_path,
            )

        set_active_session(new_id)
        st.session_state.active_store = load_active_store(new_id)
        st.session_state.loaded_store_session_id = new_id
        st.session_state.pop("skill_gap_result", None)
        st.session_state.pop("interview_result", None)
        st.success("Your new conversation and PDFs have been saved.")
        st.rerun()

# -------------------------- Existing conversation -----------------
else:
    session_folder = SESSIONS_DIR / active_id
    resume_path = BASE_DIR / active_session["resume_path"] if active_session.get("resume_path") else session_folder / "resume.pdf"
    jd_path = BASE_DIR / active_session["jd_path"] if active_session.get("jd_path") else session_folder / "jd.pdf"

    with st.expander("📎 Saved files for this conversation", expanded=False):
        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.markdown("**Resume**")
            st.write(active_session.get("resume_name") or "Not saved")
            if resume_path.exists():
                st.download_button(
                    "Download resume",
                    data=resume_path.read_bytes(),
                    file_name=active_session.get("resume_name") or "resume.pdf",
                    mime="application/pdf",
                    key=f"download_resume_{active_id}",
                )
        with file_col2:
            st.markdown("**Job description**")
            st.write(active_session.get("jd_name") or "Not saved")
            if jd_path.exists():
                st.download_button(
                    "Download job description",
                    data=jd_path.read_bytes(),
                    file_name=active_session.get("jd_name") or "job_description.pdf",
                    mime="application/pdf",
                    key=f"download_jd_{active_id}",
                )

# ------------------------------ AI tools ---------------------------
vector_store = st.session_state.active_store

if active_session and vector_store is None:
    st.warning(
        "This conversation is saved, but its document index could not be loaded. "
        "Check that the files under data/sessions and their chroma_db folder still exist."
    )

if active_session and vector_store is not None:
    tab_chat, tab_gap, tab_interview = st.tabs(
        ["💬 Ask AI", "🎯 Skill Gap", "🎤 Interview Prep"]
    )

    with tab_chat:
        messages = get_messages(active_id)
        if messages:
            for message in messages:
                with st.chat_message("user"):
                    st.markdown(message["question"])
                with st.chat_message("assistant"):
                    st.markdown(message["answer"])
                    sources = message.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                page = source.get("page")
                                page_label = f" · Page {page}" if page is not None else ""
                                st.markdown(f"**{source.get('type', 'Document')}** · {source.get('filename', 'Uploaded document')}{page_label}")
                                st.write(source.get("content", ""))
        else:
            st.info("Ask your first question about the saved resume and job description.")

        query = st.chat_input("Ask about the resume, job, skills, or interview...")
        if query and query.strip():
            with st.spinner("Thinking..."):
                results = retrieve_documents(vector_store, query)
                answer = generate_answer(query, results)
                sources = [
                    {
                        "type": result.metadata.get("document_type", "Document"),
                        "filename": result.metadata.get("source_filename") or result.metadata.get("source") or "Uploaded document",
                        "page": result.metadata.get("page_number", (result.metadata.get("page") + 1) if result.metadata.get("page") is not None else None),
                        "content": result.page_content,
                    }
                    for result in results
                ]
                add_message(active_id, query.strip(), answer, sources)
            st.rerun()

    
    with tab_gap:
        st.write(
            "Compare the candidate's skills and experience "
            "with the requirements in this job description."
        )

        if st.button(
            "🎯 Analyze Skill Gap",
            key=f"skill_gap_{active_id}",
            use_container_width=True,
        ):
            with st.spinner("Analyzing this conversation's documents..."):
                st.session_state.skill_gap_result = None

                st.session_state.skill_gap_result = (
                    generate_skill_gap_analysis(vector_store)
                )

        if st.session_state.get("skill_gap_result"):
            st.markdown(st.session_state.skill_gap_result)

    with tab_interview:
        st.write("Generate interview questions based on this candidate and role.")
        if st.button("🎤 Generate Questions", key=f"interview_{active_id}", use_container_width=True):
            with st.spinner("Preparing interview questions..."):
                st.session_state.interview_result = generate_interview_questions(vector_store)
        if st.session_state.get("interview_result"):
            st.markdown(st.session_state.interview_result)

elif not active_session:
    st.info("Your workspace is empty. Upload both PDFs above to start a new conversation.")
