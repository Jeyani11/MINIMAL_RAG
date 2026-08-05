import os
import sys
import glob # What does all this import used for ?
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain.evaluation.qa import QAEvalChain

#=======================DOCUMENT LOADING===================

loader = PyPDFLoader("CORPUS/paper.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} documents")

def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'Page \d+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text.strip()

# Apply to documents
for doc in documents:
    doc.page_content = preprocess_text(doc.page_content)

#=========================== CHUNKING =============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Max characters per chunk
    chunk_overlap=50 # Overlap between chunks
)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

#========================== EMBEDDING =====================
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2",
    model_kwargs = {'device':'cpu'}
)

#=========================== VECTOR DATABASE =========================
# Create vector store from documents

db = FAISS.from_documents(docs, embeddings)
print("Vector store created")

#=========================== RETRIEVAL =========================

# Create a retriever
retriever = db.as_retriever(
    search_type =  "similarity",
    search_kwargs = {"k":5} # Return top 5 chunks 
)
    

# Retrieve documents
query = "What does RAG stand for ?"
retrieved_docs = retriever.invoke(query)
print(f"Retrived {len(retrieved_docs)} documents")

#================================ GENERATION ==================================

# Récupère la clé
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vérification optionnelle (à retirer en production)
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

#Initialize LLM

llm = ChatGoogleGenerativeAI(
    model= "gemini-flash-latest",
    temperature = 0.0,
    google_api_key = GOOGLE_API_KEY

)

# Create RAG Chain

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context. 
    If the context does not contain the answer, say "I don't know".
    Always be concise.
    
    Context: {context}
    Question: {input}
    """
)

documents_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
    )

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=documents_chain,
    
)
# Generate response
response = rag_chain.invoke({"input": query})
print(response["answer"])

# =========================== EVALUATION =========================
# Prepare evaluation data
examples = [{ "query": query, "answer": "Ground truth answer" }] 
predictions = [{"query": query, "result": response["answer"] }]

