# Retriever similarity ? MMR ? How does it work

import os
import sys
import glob # Lister fichiers avec pattern
import re

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# from langchain.evaluation.qa import QAEvalChain

#=======================DOCUMENT LOADING===================

loader = glob.glob("CORPUS/*.pdf")
documents = []
for path in loader:
    doc = PyPDFLoader(path)
    documents.extend(doc.load())
print(f"Loaded {len(documents)} documents")

def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'Page \d+', '', text)
    # Remove special characters
    #text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text.strip()

# Apply to documents
for doc in documents:
    doc.page_content = preprocess_text(doc.page_content)

#=========================== CHUNKING =============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Max characters per chunk
    chunk_overlap=300 # Overlap between chunks
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
    search_type =  "similarity", # Maximum Marginal Relevance for diversity
    search_kwargs = {"k":45} # Return top 5 chunks 
)
    


#=========================== RERANKER =========================

cross_encoder = HuggingFaceCrossEncoder(model_name = "cross-encoder/ms-marco-MiniLM-L6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n= 5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = reranker,
    base_retriever = retriever,
)

# Retrieve documents
query = "How is Wikipedia used for RAG in a wide range of knowledge-intensive tasks ?"
retrieved_compressed_docs = compression_retriever.invoke(query)
for i, doc in enumerate(retrieved_compressed_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


"""
print("\n--- RETRIEVED CHUNKS ---")
# Cherche le chunk contenant la définition, indépendamment du retriever
for i, doc in enumerate(docs):
    if "retrieval-augmented generation" in doc.page_content.lower():
        print(f"Chunk index {i}: {doc.page_content[:400]}")

all_scored = db.similarity_search_with_score(query, k=len(docs))  # score sur TOUS les chunks
for rank, (doc, score) in enumerate(all_scored):
    if "retrieval-augmented generation" in doc.page_content.lower():
        print(f"Rang réel du bon chunk : {rank} (score={score})")
        break

# Verifie si le chunk contient "Retrieval-Augmented"
for i, doc in enumerate(retrieved_docs):
    if "Retrieval-Augmented" in doc.page_content or "RAG" in doc.page_content:
        print(f"✓ Chunk {i+1} contains RAG related content!")
        print(f"Full content: {doc.page_content}")
"""
#================================ GENERATION ==================================

# Récupère la clé
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vérification optionnelle (à retirer en production)
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
else:
    print("GOOGLE_API_KEY found in .env file")
    print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY[:4]}...{GOOGLE_API_KEY[-4:]}")  # Affiche les 4 premiers et derniers caractères
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

    Important: If the question asks for an acronym, look for the full form
    
    Context: {context}
    Question: {input}
    """
)

documents_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
    )

rag_chain = create_retrieval_chain(
    retriever= compression_retriever,
    combine_docs_chain=documents_chain,
    
)
# Generate response
response = rag_chain.invoke({"input": query})
print(response["answer"])

# =========================== EVALUATION =========================
# Prepare evaluation data
examples = [{ "query": query, "answer": "Ground truth answer" }] 
predictions = [{"query": query, "result": response["answer"] }]

