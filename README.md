# Comparative Analysis of RAG Architectures

## Overview

This project presents a comparative analysis of three Retrieval-Augmented Generation (RAG) architectures: **Basic RAG**, **Corrective RAG**, and **Graph RAG**. The goal is to demonstrate how each RAG architecture handles data retrieval and generation, and their respective performance improvements. 

The architectures were implemented using **LangChain**, an open-source framework that simplifies the development of applications powered by language models. For the **Graph RAG** architecture, we utilized a **Neo4j Knowledge Graph** to enrich the retrieval process by capturing relationships within the data.

## Architectures Implemented

### 1. **Basic RAG**

The **Basic RAG** architecture follows the traditional method of combining a retriever (to fetch relevant documents) with a generative model. The retrieved documents are used to augment the model's input, which leads to more accurate and context-aware responses.

- **Implementation Link**: [Basic RAG](https://python.langchain.com/docs/concepts/rag/)

### 2. **Corrective RAG**

The **Corrective RAG** model improves upon the basic RAG architecture by introducing a feedback loop to correct the information retrieved during the retrieval process. This method enhances the overall output quality by refining the retrieved documents before they are fed into the generative model.

- **Implementation Link**: [Corrective RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)

### 3. **Graph RAG**

The **Graph RAG** model uses the principles of graph databases to improve the retrieval process. It makes use of a **Neo4j Knowledge Graph** to store and retrieve data relationships. By leveraging the graph structure, it can more effectively find connections between various pieces of information and integrate them into the final response.

- **Implementation Link**: [Neo4j](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)

## Architecture Workflow

### LangChain Integration

All three RAG architectures were implemented using **LangChain**, which provided the framework for seamlessly integrating retrievers with generative models. LangChain made the entire end-to-end process, from data retrieval to response generation, easy to manage and optimize.

### Neo4j Knowledge Graph for Graph RAG

For the **Graph RAG** implementation, we used **Neo4j** to construct a knowledge graph based on the same dataset used by the basic and corrective RAG models. Neo4j allowed us to store entities and their relationships in a graph format, which could then be queried to retrieve the most relevant relationships for generating a response.

- Neo4j allowed us to:
  - Construct the knowledge graph from structured data.
  - Retrieve relevant entities and relationships to augment the generative model’s input.
  - Enhance the accuracy and contextual relevance of responses by leveraging the graph structure.

## Evaluation

For the evaluation of the models, each RAG architecture is stored in its respective folder. Inside each folder, you will find an **Excel file** that contains the results of the evaluation for that specific RAG type. These files offer detailed insights into the performance of each model, including response accuracy and retrieval efficiency.

## Conclusion

This project explores and compares the performance of three RAG architectures using LangChain, offering insights into the strengths of each model. The **Graph RAG** architecture, enhanced with a Neo4j Knowledge Graph, demonstrated a significant advantage in handling complex relationships within the data compared to the **Basic RAG** and **Corrective RAG**.
