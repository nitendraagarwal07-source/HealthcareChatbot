import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -------------------- Logging --------------------
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------- Guardrails --------------------
DISALLOWED_CATEGORIES = ["politics", "religion", "personal advice"]

def is_disallowed(query: str) -> bool:
    query_lower = query.lower()
    return any(category in query_lower for category in DISALLOWED_CATEGORIES)

# -------------------- Document Ingestion --------------------
def load_and_embed(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    documents = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="Corporate Healthcare Sales Chatbot", layout="wide")
    st.markdown(
        "<h1 style='color:#0047AB;'>🏥 Corporate Healthcare Sales Chatbot</h1>",
        unsafe_allow_html=True
    )

    # Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # File Upload
    uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.vectorstore = load_and_embed("temp.pdf")
        st.success("✅ Document uploaded, chunked, embedded and indexed successfully!")

    # Reset Conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.success("Conversation reset!")

    # Chat UI
    query = st.text_input("💬 Ask a question about the document:")
    if query:
        if is_disallowed(query):
            response = "🚫 This question is outside the scope of my knowledge base."
        elif not st.session_state.vectorstore:
            response = "⚠️ Please upload a document first."
        else:
            # Retrieve relevant chunks
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Build prompt
            prompt = f"""
            You are a strict assistant that ONLY answers using the provided context.
            If the answer is not in the context, say:
            "This question is outside the scope of my knowledge base."

            Context: {context}
            Question: {query}

            Answer strictly from the context:
            """

            llm = ChatOpenAI(
                model="gpt-4o-mini",  # free/cheaper tier model
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

            response = llm.predict(prompt)

        # Save to chat history
        st.session_state.chat_history.append((query, response))

        # Log query + response
        logging.info(f"Query: {query}")
        logging.info(f"Response: {response}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, (q, r) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"<span style='color:#0047AB;'>**A{i}:** {r}</span>", unsafe_allow_html=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
