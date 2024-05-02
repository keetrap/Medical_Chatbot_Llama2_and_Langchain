from flask import Flask, render_template,request
from src.helper import download_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

embeddings = download_embeddings()

#Initializing the Pinecone
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

index_name="chatbot"

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)


Prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": Prompt}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':1024,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    result=qa.invoke(msg)
    # print(result)
    # print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug= True)