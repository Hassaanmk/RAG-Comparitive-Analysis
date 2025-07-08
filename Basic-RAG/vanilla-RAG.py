import os
import cohere
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import env_loader  # This will load the .env file
import random
import pandas as pd
import math
from typing import List,Tuple
from langchain_core.documents import Document

#openai_key = os.getenv("OPENAI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")
co = cohere.ClientV2(cohere_key)

# Load FAISS vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local("faiss_openai_3-large", embedding_model, allow_dangerous_deserialization=True)

#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # This returns just documents

def retrieve_with_scores(query: str) -> Tuple[List[Document], List[float]]:
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    docs, scores = zip(*docs_with_scores) if docs_with_scores else ([], [])
    scores = [round(float(score) * 10, 2) for score in scores]
    return list(docs),scores

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Prompts
answer_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question:
{context}
Question: {question}
""")

def cohere_rerank(question, documents, top_n=3):
    docs = [doc.page_content for doc in documents]
    response = co.rerank(model="rerank-v3.5", query=question, documents=docs, top_n=top_n)
    # Preserve original documents in same order
    reranked_docs = [documents[r.index] for r in response.results]
    #reranked_score= [r.relevance_score*10 for r in response.results]
    reranked_score = [round(r.relevance_score * 10, 2) for r in response.results]
    return response,reranked_docs,reranked_score

# Full RAG pipeline
def rag_pipeline(question):
    #retrieved_docs = retriever.invoke(question)
    #rank_response,top_docs,rank_score = cohere_rerank(question, retrieved_docs, top_n=3)
    retrieved_docs , doc_scores = retrieve_with_scores(question)

    response = (answer_prompt | llm).invoke({
        "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
        "question": question
    })
    return response,retrieved_docs,doc_scores


def response_evaluator(question, reranked_docs, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant evaluating how relevant all documents are to a users question. "
         "You must give a combined numeric score between 0 (not related at all) and 10 (perfectly relevant). "
         "Give only the score as a number."),
        ("human", "Question:\n{question}\n\nDocuments:\n{documents}\n\nScore (0-10):")
    ])
    
    chain = prompt | llm | StrOutputParser()
    #for i, doc in enumerate(reranked_docs):
    response = chain.invoke({
            "question": question,
            "documents": reranked_docs})
    try:
        score = int(response.strip())
    except:
        score = -1  # If parsing fails

    return score


rag_test_questions = [
    # Treasure Island
    "How does Long John Silver's duality challenge traditional notions of villainy and morality in *Treasure Island*?",
    "In what ways does Jim Hawkins' transformation embody a critique of colonial adventure narratives in *Treasure Island*?",
    "How do shifting loyalties and betrayal shape the moral landscape of *Treasure Island*?",

    # Pride and Prejudice
    "How does the evolution of Elizabeth Bennet's character reflect broader feminist undercurrents in *Pride and Prejudice*?",
    "In what ways does *Pride and Prejudice* satirize the performativity of social class and gender norms?",
    "How does Austen use narrative irony to subvert readers expectations of romantic tropes in *Pride and Prejudice*?",

    # A Tale of Two Cities
    "How does Dickens use the motif of doubles and contrasts to critique the nature of justice and revolution in *A Tale of Two Cities*?",
    "What role does memory and sacrifice play in the shaping of identity in *A Tale of Two Cities*?",
    "How does Sydney Carton's character arc embody existential redemption in the face of deterministic societal forces?",

    # Dangerous Liaisons
    "How does the correspondence format in *Dangerous Liaisons* enhance the novel's themes of manipulation and performative virtue?",
    "What does the rivalry between Merteuil and Valmont reveal about gender, power, and hypocrisy in aristocratic French society?",
    "How does *Dangerous Liaisons* reflect Enlightenment tensions between reason and passion?",

    # The Three Musketeers
    "How do Duma's portrayals of duels and honor rituals critique absolutist political authority in *The Three Musketeers*?",
    "In what ways do the Musketeers embody competing notions of masculinity and loyalty during a time of political instability?",
    "How does *The Three Musketeers* reflect tensions between personal ambition and state service in 17th-century France?"
]

# Initialize an empty DataFrame
results_df = pd.DataFrame(columns=[
    'question',  
    'doc-1',
    'doc-2',
    'doc-3',
    'semantic-score',
    'llm-score'
])

j=0
for question in rag_test_questions:
    print("\n" + "="*80)
    print("question:", question)
    j=j+1
    
    final_response,top_docs,rank_score= rag_pipeline(question)
    llm_score= response_evaluator(question, top_docs, ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0))

    content_list=[]
    score=rank_score
    doc_1=0
    doc_2=0
    doc_3=0
    print(f"\nDocs are scored {score}")
    #for i, content in results:  # Display top 3
        #print("Content Preview:\n", content[:300], "...\n")
    content_list=top_docs

    doc_1=content_list[0].page_content
    doc_2=content_list[1].page_content
    doc_3=content_list[2].page_content
    row ={
        'question': question,
        'doc-1': doc_1,
        'doc-2': doc_2,
        'doc-3': doc_3,
        'semantic-score': score,
        'llm-score' : llm_score
        }
    
    print("\n question response:", final_response)
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    #if j==1:
        #break

results_df.to_excel("new_basic-rag_test_results.xlsx", index=False)
print("df saved")

