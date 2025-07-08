# Vector store
from langchain_community.vectorstores import FAISS 
# Embedding models
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import env_loader  # This will load the .env file
import time
import os

# Set your API key
openai_key = os.getenv("OPENAI_API_KEY")

# Path to PDF folder
pdf_folder = "C:/Users/Hassan Muhammad Khan/Downloads/RAG Comparision/PDF/"

# Load all PDFs: each page is a chunk
all_docs = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(filepath)
        print(f'PDF-loaded: {filename}')
        pages = loader.load()  # returns one Document per page
        for page_num, doc in enumerate(pages):
            doc.metadata["source"] = filename
            doc.metadata["page"] = page_num + 1  # optional
        all_docs.extend(pages)

# Now all_docs has 1 chunk per PDF page
print(len(all_docs)) #3037 pages

# Embed using text-embedding-ada-002
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
# Text splitter to break long pages (optional)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#initially
vectorstore = None
for i, doc in enumerate(all_docs):
    
    chunks = splitter.split_documents([doc])  # splits a single page if needed

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        time.sleep(0.1)
    else:
        vectorstore.add_documents(chunks)
        time.sleep(0.1)

    if i % 100 == 0:
        print(f"Embedded {i}/{len(all_docs)} pages...")
        #vectorstore.save_local("faiss_openai_3-large")


vectorstore.save_local("faiss_openai_3-large")