from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    doc=loader.load()
    return doc

def test_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    chunks = text_splitter.split_documents(data)
    return chunks

def download_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings