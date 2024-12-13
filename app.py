### End to end medical chatbot

from flask import Flask, request, jsonify, render_template
from src.helper import embedding
from langchain_pinecone import Pinecone
from src.prompt import *
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Chunks operation
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load the API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Embedding loading
embedding = embedding()

# Pinecone setup
index_name = "chatbotwebsite"
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)

# Prompt setup
prompt = ChatPromptTemplate(
    [
        ('system', system_prompt),
        ('human', "{input}")
    ]
)

# Chaining setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask setup
@app.route("/")
def index(): 
    return render_template('index.html')

@app.route('/get', methods=["POST"])
def chat():
    msg = request.json.get('msg')  # Receive JSON data from the front-end
    if not msg:
        return jsonify({"error": "Message is required"}), 400
    
    response = rag_chain.invoke({"input": msg})
    answer = response.get('answer', 'I am sorry, I could not understand your question.')
    return jsonify({"answer": answer})  # Respond with JSON data


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
