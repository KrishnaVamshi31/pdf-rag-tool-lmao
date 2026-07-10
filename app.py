import streamlit as st
import tempfile
import os

from ingestion import load_pdf, split_documents
from embeddings_store import build_vector_store
from rag_chain import answer_question

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 PDF Q&A")
st.caption("Upload a PDF and ask questions about it — answers are grounded in the document, with page citations.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Use filename + size as a simple cache key -- good enough to detect
    # "is this a new file" without hashing the whole file content.
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("processed_file_key") != file_key:
        # New file (or first upload this session) -- process it.
        with st.spinner("Reading PDF and building index... (first time only, ~10-30s)"):
            # Streamlit's uploaded_file is an in-memory buffer, but our
            # load_pdf() expects a file path -- write it to a temp file first.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            docs = load_pdf(tmp_path)
            chunks = split_documents(docs)
            store = build_vector_store(chunks)

            os.unlink(tmp_path)  # clean up temp file, we don't need it anymore

            st.session_state["vector_store"] = store
            st.session_state["processed_file_key"] = file_key
            st.session_state["chat_history"] = []  # reset Q&A history for new file

        st.success(f"Processed {len(docs)} pages -> {len(chunks)} chunks. Ready to answer questions.")
    else:
        st.info(f"Using already-processed: **{uploaded_file.name}**")

    # --- Question input ---
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = answer_question(query, st.session_state["vector_store"])

        st.session_state.setdefault("chat_history", []).append(
            {"query": query, "answer": answer, "sources": sources}
        )

    # --- Display Q&A history (most recent first) ---
    for entry in reversed(st.session_state.get("chat_history", [])):
        st.markdown(f"**Q: {entry['query']}**")
        st.write(entry["answer"])
        with st.expander(f"Sources: pages {entry['sources']}"):
            results = st.session_state["vector_store"].similarity_search(entry["query"], k=4)
            for doc, score in results:
                st.markdown(f"**Page {doc.metadata['page']}** (relevance: {score:.2f})")
                st.text(doc.page_content[:300] + "...")
        st.divider()

else:
    st.info("Upload a PDF to get started.")