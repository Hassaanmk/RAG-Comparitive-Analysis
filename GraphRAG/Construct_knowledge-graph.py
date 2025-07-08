import os
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_neo4j import Neo4jGraph
import json
import time
import random


# Load API keys and other configurations
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Path to PDF folder
pdf_folder = "C:/Users/HMK/Desktop/RAG Comparision/PDF/"

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

print(len(all_docs)) #3037 pages

#Use LLMGraphTransformer to convert documents into graph triples
llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
transformer = LLMGraphTransformer(llm=llm)

#Connect to Neo4j instance
graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_password,
)
print("Connected to graph")


#Create a knowledge graph from all the PDF texts
local_graph_data = []
for i in range(0,len(all_docs),10):
    if i<=3010:
        continue 
    #if i==2980:
        #continue

    graph_docs = transformer.convert_to_graph_documents(all_docs[i:i+10])  # Convert 10 pages (10 chunks)at a time
    #Append to the local graph data
    for g_doc in graph_docs:
        local_graph_data.append(g_doc)
    
    graph.add_graph_documents(graph_docs)
    print("Added to neo4j till Page no.: ", i )

    if i % 100 == 0:
        print(f"done {i}/{len(all_docs)} pages...")


print(f"Knowledge graph has been created and pushed to Neo4j!")

print("Sample KG query result:")
result = graph.query("MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 10")
print(result)

# Testing Query using a specific node and get related documents or entities
print("Querying specific node results:")
specific_node_query = graph.query("MATCH (n:Entity {name: 'Jim Hawkins'})-[:TRANSFORMS]->(m) RETURN m LIMIT 5")
print(specific_node_query)

# Save the knowledge graph locally as a JSON file (optional)
local_graph_file_path = "C:/Users/HMK/Desktop/RAG Comparision/knowledge_graph.json"
with open(local_graph_file_path, "w") as json_file:
    json.dump(local_graph_data, json_file, indent=4)

print(f"Knowledge graph has been saved locally at: {local_graph_file_path}")




