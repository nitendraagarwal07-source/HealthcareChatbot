import os
import logging
import textwrap
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# -------------------- Logging --------------------
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------- Guardrails --------------------
DISALLOWED_CATEGORIES = ["politics", "religion", "personal advice"]
REFUSAL_TEXT = "This question is outside the scope of my knowledge base."

def is_disallowed(query: str) -> bool:
    q = (query or "").lower()
    return any(cat in q for cat in DISALLOWED_CATEGORIES)

# -------------------- Build Local LLM (no API keys) --------------------
@st.cache_resource(show_spinner=False)
def load_local_llm(model_name: str = "google/flan-t5-small"):
    """
    Loads a small, CPU-friendly instruction model for free.
    Options:
      - google/flan-t5-small  (~80M)  -> fastest, lower quality
      - google/flan-t5-base   (~250M) -> balanced
    """
    gen = pipeline(
        "text2text-generation",
        model=model_name,
        device_map="auto",         # works fine on CPU
    )
    return gen

# -------------------- Embeddings (free) --------------------
@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# -------------------- Vector store build --------------------
def load_and_embed(pdf_path: str, splitter_cfg: dict) -> FAISS:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_cfg.get("chunk_size", 1000),
        chunk_overlap=splitter_cfg.get("chunk_overlap", 200),
        separators=splitter_cfg.get("separators", None),
    )
    docs = text_splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# -------------------- Prompt construction --------------------
def build_prompt(context: str, question: str) -> str:
    # Keep it compact for small models
    prompt = f"""
You are a strict assistant that MUST answer ONLY using the context below.
If the answer is not present in the context, reply exactly:
"{REFUSAL_TEXT}"

Context:
{context}

Question:
{question}

Answer briefly and only from the context:
"""
    # T5 models prefer concise prompts (trim excess whitespace)
    return textwrap.dedent(prompt).strip()

# -------------------- Generate answer with guardrails --------------------
def answer_with_context(llm_pipe, context: str, question: str) -> str:
    if not context.strip():
        return REFUSAL_TEXT

    prompt = build_prompt(context, question)
    out = llm_pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,     # deterministic
        num_beams=4
    )
    text = out[0]["generated_text"].strip()

    # Safety: if model ignored instructions, enforce refusal
    if not text or text.lower().startswith("as an ai") or "i don't have information" in text.lower():
        return REFUSAL_TEXT
    # Prevent repetitive looping: truncate overly long repeats
    return text[:2000]

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="🏥 HCL Healthcare Sales Chatbot (Free)", layout="wide")

    # HCL blue/white theme
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #0033A0; }
        .stButton button { background-color: #0033A0; color: #ffffff; border-radius: 8px; }
        .stTextInput > div > div > input { border: 1px solid #0033A0; }
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color:#0033A0;'>🏥 HCL Healthcare Sales Chatbot (Free, No API Keys)</h1>", unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Local model",
            ["google/flan-t5-small", "google/flan-t5-base"],
            index=0,
            help="Smaller = faster; Base = better quality, a bit slower."
        )
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 10)
        st.caption("Tip: Larger chunks improve context, but too large may slow retrieval.")

        st.markdown("---")
        if st.button("🔄 Reset Conversation"):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.loaded_file_name = None
            st.success("Conversation reset!")

    # Session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "loaded_file_name" not in st.session_state:
        st.session_state.loaded_file_name = None

    # File upload at top
    uploaded = st.file_uploader("📂 Upload a PDF document", type="pdf")
    if uploaded:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            st.session_state.vectorstore = load_and_embed(
                "temp.pdf",
                splitter_cfg={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            )
            st.session_state.loaded_file_name = uploaded.name
            st.success(f"✅ Indexed: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            return

    # Load local model (cached)
    llm_pipe = load_local_llm(model_choice)

    # Chat window
    st.subheader("💬 Chat")
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")

    query = st.text_input("Ask a question about the uploaded document:")

    if st.button("Send"):
        if not query:
            st.info("Please enter a question.")
        elif is_disallowed(query):
            response = REFUSAL_TEXT
        elif not st.session_state.vectorstore:
            response = "Please upload a PDF first."
        else:
            # Retrieve top-3 most relevant chunks
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)

            # Build context from top chunks (trim each to keep prompt small)
            def _trim(txt, cap=1200):
                txt = txt.replace("\n", " ").strip()
                return txt[:cap]

            context = "\n\n".join(_trim(d.page_content) for d in docs)

            # Generate answer from the local model using ONLY the context
            response = answer_with_context(llm_pipe, context, query)

        # Save + display
        st.session_state.chat_history.append((query, response))
        logging.info(f"Query: {query}")
        logging.info(f"Response: {response}")
        st.experimental_rerun()

    # Footer info
    if st.session_state.loaded_file_name:
        st.caption(f"Indexed document: {st.session_state.loaded_file_name} · Retrieval: FAISS · Embeddings: all-MiniLM-L6-v2 · Model: {model_choice}")
    else:
        st.caption("Upload a PDF to begin. All processing is local & free (no API keys).")

if __name__ == "__main__":
    main()
