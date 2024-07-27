import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain.llms import CTransformers
import tempfile
import os
from llama_cpp import Llama
from langchain.llms import LlamaCpp

# Function to load document based on file type
def load_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            loader = UnstructuredFileLoader(temp_file_path)
        
        document = loader.load()
        return document
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None
    finally:
        os.remove(temp_file_path)

# Process documents
def process_documents(documents):
    if documents is None:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

# Create vector store
def create_vector_store(texts):
    if texts is None:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# Load LLM
@st.cache_resource
def load_llm():
    try:
        #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")
#         llm = Llama(
#     model_path='Meta-Llama-3.1-8B-Instruct-Q8_0.gguf',
#     n_ctx=16000,  # Context length to use
#     n_threads=32,            # Number of CPU threads to use
#     n_gpu_layers=0        # Number of model layers to offload to GPU
# )
        llm = LlamaCpp(
    streaming = True,
    model_path="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# Main function
def main():
    st.title("Local RAG Application with File Upload")

    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])

    if uploaded_file is not None:
        documents = load_document(uploaded_file)
        if documents is None:
            return

        texts = process_documents(documents)
        if texts is None:
            return

        vectorstore = create_vector_store(texts)
        if vectorstore is None:
            return

        llm = load_llm()
        if llm is None:
            return

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        )

        # User input
        query = st.text_input("Enter your question:")

        if query:
            try:
                # Get answer
                answer = qa_chain.run(query)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()