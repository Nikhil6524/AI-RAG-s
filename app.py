import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
import tempfile

# ---------------------------
# UI SETUP
# ---------------------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("🚀 LangChain RAG App")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
model_name = st.sidebar.selectbox(
    "Model",
    ["nvidia/nemotron-3-nano-30b-a3b:free"]
)

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# ---------------------------
# LOAD LLM
# ---------------------------
def load_llm(openrouter_api_key: str, model: str):
    return ChatOpenAI(
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=model,
        temperature=0.3
    )

# ---------------------------
# PROCESS DOCUMENT
# ---------------------------
@st.cache_resource
def process_file(file):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore

# ---------------------------
# PROMPT
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the given context.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------
# MAIN LOGIC
# ---------------------------
if uploaded_file and api_key:

    st.success("File uploaded successfully!")

    vectorstore = process_file(uploaded_file)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(api_key, model_name)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
            st.write("### Answer:")
            st.write(response)

else:
    st.info("Upload a file and enter API key to start.")