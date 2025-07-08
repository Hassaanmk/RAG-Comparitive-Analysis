from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain import hub
import os
import cohere
import pandas as pd
import random
import env_loader  # This will load the .env file
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List,Tuple
from langchain_core.documents import Document


openai_key = os.getenv("OPENAI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")
#co = cohere.ClientV2(cohere_key)

# Load FAISS vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local("faiss_openai_3-large", embedding_model, allow_dangerous_deserialization=True)

#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # This returns just documents

def retrieve_with_scores(query: str) -> Tuple[List[Document], List[float]]:
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    docs, scores = zip(*docs_with_scores) if docs_with_scores else ([], [])
    scores = [round(float(score) * 10, 2) for score in scores]
    return list(docs),scores


rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the retrieved *context* to answer the *question*. 
CONTEXT:
{context}
QUESTION:
{question}
Provide a corrected, complete, and accurate answer:
""")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
rag_chain = rag_prompt | llm | StrOutputParser()


#Document evaluator
class RetrievalEvaluator(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

retrieval_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
structured_llm_evaluator = retrieval_evaluator_llm.with_structured_output(RetrievalEvaluator)
retrieval_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a document retrieval evaluator responsible for checking if the retrieved documents contain the answer to the user question.
        If the documents contains the answer related to the question, grade them as relevant.
        Output a binary score 'yes' if present or 'no' if not present."""),
    ("human", "Retrieved documents: \n\n{documents}\n\n User question: {question}")
])
retrieval_grader = retrieval_eval_prompt | structured_llm_evaluator


# Question Rewriter
question_rewriter_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a question re-writer that converts an input question to a better version optimized for retrieval. The relevant retrieved documents of initial question are also provided. " 
     " Use the relevant docs and change the initial questions to formulate an improved question ."),
    ("human", "Retrieved documents: \n\n{documents}\n\n initial question: {question}""\n\nFormulate an improved question."),
])
question_rewriter = rewrite_prompt | question_rewriter_llm | StrOutputParser()

def semantic_rank(question, documents, top_n=3):
    # Load a suitable SentenceTransformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2") 
    # Convert docs to list of texts
    doc_texts = [doc.page_content for doc in documents]
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    doc_embeddings = sentence_model.encode(doc_texts, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, doc_embeddings).squeeze()

    # Sort by similarity
    sorted_indices = torch.argsort(similarities, descending=True)
    #top_indices = sorted_indices[:top_n]
    top_docs = [documents[i] for i in sorted_indices]
    rank_scores = [round(float(similarities[i]), 4) for i in sorted_indices]
    return top_docs, rank_scores


def crag_pipeline(question: str, min_docs: int = 3, max_rewrites: int = 5):
    attempt = 0
    rewritten_question = question
    retrieved_docs , doc_scores = retrieve_with_scores(question)

    while attempt < max_rewrites:
        #print(f"\n Retrieval attempt {attempt + 1} → Using question: {repr(rewritten_question)}")
        retrieved_docs, doc_scores = retrieve_with_scores(rewritten_question)
        if len(retrieved_docs) == 0:
            print("No documents retrieved. Trying rewrite...")
            rewritten_question = question_rewriter.invoke({"question": rewritten_question, "documents": retrieved_docs})
            attempt += 1
            continue
        # Evaluate relevance of retrieved documents
        relevant_docs = []
        #for doc in retrieved_docs:  # only evaluate top n docs
        grade = retrieval_grader.invoke({"documents": retrieved_docs, "question": rewritten_question})
        if grade.binary_score.strip().lower() == "yes" or "Yes":
            relevant_docs=retrieved_docs

        if len(relevant_docs) == min_docs:
            print("got 3 docs with answers in them")
            break

        rewritten_question = question_rewriter.invoke({"question": rewritten_question, "documents": relevant_docs}) # Otherwise, rewrite the question and try again
        print("Rewriting question")        
        attempt += 1

    # Get 3 reranked top docs to generate the final answer
    print("lenght of relevant docs is:", len(relevant_docs))
    #top_docs,rank_score = semantic_rank(question, relevant_docs)
    #context = format_docs(top_docs)
    #print("lenght of reranked docs is: ", len(top_docs))
    answer = rag_chain.invoke({"context": relevant_docs, "question": question})
    return answer,relevant_docs,doc_scores


def response_evaluator(question, reranked_docs, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant evaluating how relevant all documents arre to a users question. "
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
    
    final_response,top_docs,rank_score= crag_pipeline(question)
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

results_df.to_excel("new_corrective_rag_test_results-2.xlsx", index=False)
print("df saved")

