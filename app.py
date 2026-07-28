import os
import tempfile
import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


st.set_page_config(page_title="Multi-Doc Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Multi-Document Research Assistant")
st.caption("Upload multiple PDFs/txt files and ask questions across all of them, with source citations.")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading LLM (flan-t5-base)...")
def load_llm():
    model_id = "google/flan-t5-base"  # CPU-friendly: ~250M params, fast on regular hardware
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
    "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)



PROMPT_TEMPLATE = """Answer the question using only the context below. Context comes from
multiple documents. Mention which document(s) support your answer.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files (multiple allowed)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, step=50)
    top_k = st.slider("Chunks retrieved per query", 2, 10, 5)

    process_btn = st.button("Process Documents", type="primary", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Reading and indexing documents..."):
                all_docs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uf in uploaded_files:
                        path = os.path.join(tmpdir, uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.getbuffer())

                        if uf.name.lower().endswith(".pdf"):
                            loader = PyPDFLoader(path)
                        else:
                            loader = TextLoader(path)

                        docs = loader.load()
                        st.write(f"Loaded {len(docs)} pages from{uf.name},total chars: {sum(len(d.page_content) for d in docs)}")
                        for d in docs:
                            d.metadata["source"] = uf.name
                        all_docs.extend(docs)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    chunks = splitter.split_documents(all_docs)

                    embeddings = load_embeddings()
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

                    llm = load_llm()

                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": PROMPT},
                        return_source_documents=True,
                    )
                    st.session_state.processed_files = [uf.name for uf in uploaded_files]
                    st.session_state.chat_history = []

            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")

    if st.session_state.processed_files:
        st.divider()
        st.subheader("Indexed documents")
        for fname in st.session_state.processed_files:
            st.write(f"📄 {fname}")

        if st.button("Clear all", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.rerun()


if st.session_state.qa_chain is None:
    st.info("👈 Upload documents and click **Process Documents** in the sidebar to get started.")
else:

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry["sources"]:
                with st.expander("Sources used"):
                    for src in entry["sources"]:
                        st.write(f"- {src}")


    question = st.chat_input("Ask a question across your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = sorted(set(
                    doc.metadata.get("source", "unknown")
                    for doc in result["source_documents"]
                ))

            st.write(answer)
            if sources:
                with st.expander("Sources used"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
        })