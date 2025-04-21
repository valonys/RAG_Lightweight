import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
import requests
from tqdm import tqdm
import time
import os

# Download and load the model if not already present
model_path = "llama-2-7b-chat.Q4_K_M.gguf"
if not os.path.exists(model_path):
    print(f"Downloading {model_path}...")
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=1024), total=total_size//1024):
            f.write(data)
    print("Download complete!")

# Load documents
documents = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("docs", file))
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(os.path.join("docs", file))
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Initialize LLM
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=2000,
    n_ctx=4096,
    verbose=False
)

# Define prompt template
template = """
Answer the question based on the following context:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create RAG pipeline
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Function to ask a question and stream the output
def ask_question(question):
    start_time = time.time()
    result = rag_pipeline({"query": question})
    end_time = time.time()
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\nSource documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Document {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...\n")

# Streamlit app
st.title("RAG Model Question Answering App")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    with open(os.path.join("docs", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"File {uploaded_file.name} uploaded successfully!")

# Question input
question = st.text_input("Ask a question:", "")

if st.button("Get Answer"):
    if question:
        answer = ""
        for char in rag_pipeline({"query": question})["result"]:
            answer += char
            st.markdown(answer, unsafe_allow_html=True)
            st.experimental_rerun()
    else:
        st.warning("Please enter a question.")