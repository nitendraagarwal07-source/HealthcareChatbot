import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Free embeddings with SentenceTransformers
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [embedder.encode(doc.page_content) for doc in docs]

    # Store in FAISS
    vectorstore = FAISS.from_embeddings([(docs[i], embeddings[i]) for i in range(len(docs))], embedder)
    return vectorstore

# -------------------- QA Chain --------------------
def build_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Change model here if needed
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt_template = """
    You are a strict assistant that answers ONLY based on the provided context.
    If the answer is not present in the context, respond with:
    "This question is outside the scope of my knowledge base."

    Context: {context}
    Question: {question}

    Answer strictly from the context:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="🏥 HCL Healthcare Sales Chatbot", layout="wide")

    st.markdown(
        "<h1 style='text-align: center; color: #0047AB;'>🏥 HCL Healthcare Sales Chatbot</h1>",
        unsafe_allow_html=True
    )

    # Upload section
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type="pdf")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        vectorstore = load_and_embed("temp.pdf")
        qa_chain = build_qa_chain(vectorstore)

        st.success("✅ Document uploaded, chunked, embedded, and indexed successfully!")

        query = st.text_input("💬 Ask a question about the document:")
        if st.button("Send"):
            if query:
                if is_disallowed(query):
                    response = "🚫 This question is outside the scope of my knowledge base."
                else:
                    result = qa_chain.invoke({"query": query})
                    response = result["result"]

                # Log query + response
                logging.info(f"Query: {query}")
                logging.info(f"Response: {response}")

                # Save to history
                st.session_state.chat_history.append((query, response))

        # Chat history display
        st.markdown("### 🗂 Chat History")
        for q, r in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {r}")

        # Reset conversation
        if st.button("🔄 Reset Conversation"):
            st.session_state.chat_history = []
            st.success("Conversation reset!")

if __name__ == "__main__":
    main()
