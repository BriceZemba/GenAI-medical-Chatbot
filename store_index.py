from src.helper import load_pdf_file, text_splitter, embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file('data/')
text_split = text_splitter(extracted_data)
embedding = embedding()

## Loading of the existing index
from langchain_pinecone import Pinecone
index_name = 'chatbotwebsite'

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
