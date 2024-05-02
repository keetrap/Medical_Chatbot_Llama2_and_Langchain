from src.helper import load_pdf, test_split, download_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
index_name="chatbot"
spec=ServerlessSpec(
        cloud='aws', 
        region='us-east-1')

def create_index():
    if index_name not in pc.list_indexes().names():
    # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=384,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
    # connect to index
    index = pc.Index(index_name)
    # view index stats
    index.describe_index_stats()
    return index

data=load_pdf("data")
text_chunks=test_split(data)
embeddings=download_embedding()

index=create_index()
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embeddings,index_name=index_name)
# Uncomment the above line to create the index
# It will take some time to create the index 