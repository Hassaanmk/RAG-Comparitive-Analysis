import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# Load API keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
print(neo4j_uri)
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# 1. Load FAISS vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local("C:/Users/Hassan Muhammad Khan/Downloads/RAG Comparision/faiss_openai_3-large", embedding_model, allow_dangerous_deserialization=True)

# 2. Use LLMGraphTransformer to transform documents into graph triples
llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
transformer = LLMGraphTransformer(llm=llm)

# 3. Connect to Neo4j instance
graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_password,
)

# 4. Iterate over all documents in FAISS vector store and convert to graph format
# Assuming that vectorstore.get_all_documents() exists, otherwise you need to implement iteration

all_docs = vectorstore.get_all_documents()

# Prepare list to store graph data locally
local_graph_data = []

# Transform each document to a graph representation
for doc in all_docs:
    graph_docs = transformer.convert_to_graph_documents([doc])  # Convert one document at a time
    
    # Save to local storage (JSON format)
    for g_doc in graph_docs:
        local_graph_data.append(g_doc)
    
    # Add the graph data to Neo4j
    graph.add_graph_documents(graph_docs)

# 5. Save the knowledge graph locally as a JSON file
local_graph_file_path = "knowledge_graph.json"
with open(local_graph_file_path, "w") as json_file:
    json.dump(local_graph_data, json_file, indent=4)

print(f"✅ Knowledge graph has been created and pushed to Neo4j!")
print(f"✅ Knowledge graph has been saved locally at: {local_graph_file_path}")

# Optional: test a query to ensure that the graph has been populated correctly
print("📌 Sample KG query result:")
result = graph.query("MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 10")
print(result)

# Optional: Query using a specific node and get related documents or entities
print("📌 Querying specific node results:")
specific_node_query = graph.query("MATCH (n:Entity {name: 'Jim Hawkins'})-[:TRANSFORMS]->(m) RETURN m LIMIT 5")
print(specific_node_query)
