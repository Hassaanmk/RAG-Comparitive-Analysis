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

os.environ["USER_AGENT"] = "Corrective-RAG/0.1"

os.environ["CO_API_KEY"] = "rJ4bDc2oYF2Qd8JZOuNgcrW6mKu5MySIKORMoFOZ"
# Set your API key
os.environ["OPENAI_API_KEY"] = "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"
co = cohere.ClientV2()


#openai_key = os.getenv("OPENAI_API_KEY")
#cohere_key = os.getenv("CO_API_KEY")


# Load FAISS vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local("faiss_openai_3-large", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 40})

#LLM
rag_prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
rag_chain = rag_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Document evaluator
class RetrievalEvaluator(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

retrieval_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
structured_llm_evaluator = retrieval_evaluator_llm.with_structured_output(RetrievalEvaluator)
retrieval_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a document retrieval evaluator responsible for checking the relevancy of a retrieved document to the user's question. 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Output a binary score 'yes' or 'no'."""),
    ("human", "Retrieved document: \n\n{document}\n\n User question: {question}")
])

retrieval_grader = retrieval_eval_prompt | structured_llm_evaluator

# Question Rewriter
question_rewriter_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a question re-writer that converts an input question to a better version optimized for retrieval. " 
     "Try to reason about the intent of the original question."),
    ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved question."),
])
question_rewriter = rewrite_prompt | question_rewriter_llm | StrOutputParser()

def cohere_rerank(question, documents, top_n=3):
    docs = [doc.page_content for doc in documents]
    response = co.rerank(model="rerank-v3.5", query=question, documents=docs, top_n=top_n)
    # Preserve original documents in same order
    reranked_docs = [documents[r.index] for r in response.results]
    #reranked_score= [r.relevance_score*10 for r in response.results]
    reranked_score = [round(r.relevance_score * 10, 2) for r in response.results]
    return response,reranked_docs,reranked_score

def crag_pipeline(question: str, min_docs: int = 10):
    # Step 1: Initial retrieval using raw question
    initial_docs = retriever.invoke(question)

    # Step 2: Check if enough relevant documents were retrieved
    if len(initial_docs) < min_docs:
        print("Not enough relevant docs. Rewriting question...")
        question = question_rewriter.invoke({"question": question})
        retrieved_docs = retriever.invoke(question)
    else:
        retrieved_docs = initial_docs

    # Step 3: Evaluate relevance of documents
    relevant_docs = []
    print("lenght of retrieved docs are:", len(retrieved_docs))
    for doc in retrieved_docs:
        grade = retrieval_grader.invoke({"document": doc.page_content, "question": question})
        if grade.binary_score.lower() == "yes" or "Yes":
            relevant_docs.append(doc)

    # Step 4: Get 3 reranked top docs to generate the final answer
    print("lenght of relevant docs is:", len(relevant_docs))
    rank_response,top_docs,rank_score = cohere_rerank(question, retrieved_docs, top_n=3)
    #context = format_docs(top_docs)
    print("lenght of reranked docs is: ", len(top_docs))
    answer = rag_chain.invoke({"context": top_docs, "question": question})
    return answer,rank_response,top_docs,rank_score

def response_evaluator(question, reranked_docs, llm):
    # Prompt to evaluate each document for relevance
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant evaluating how relevant a document is to a users question. "
         "You must give a numeric score between 0 (not related at all) and 10 (perfectly relevant). "
         "Give only the score as a number."),
        ("human", "Question:\n{question}\n\nDocument:\n{document}\n\nScore (0-10):")
    ])
    
    chain = prompt | llm | StrOutputParser()
    results = []
    for i, doc in enumerate(reranked_docs):
        response = chain.invoke({
            "question": question,
            "document": doc
        })
        try:
            score = int(response.strip())
        except:
            score = -1  # If parsing fails
        results.append((i, score, doc))

    return results

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
    'cohere-rank',
    'llm-rank'
]) 

j=0
for question in rag_test_questions:
    print("\n" + "="*80)
    print("question:", question)
    j=j+1
    
    final_response,cohere_response,top_docs,rank_score= crag_pipeline(question)
    results = response_evaluator(question, top_docs, ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0))

    content_list=[]
    score_list=[]
    doc_1=0
    doc_2=0
    doc_3=0
    for i, score, content in results:  # Display top 3
        print(f"\nDoc {i+1} scored {score}")
        #print("Content Preview:\n", content[:300], "...\n")
        score_list.append(score)
        content_list.append(content)

    doc_1=content_list[0]
    doc_2=content_list[1]
    doc_3=content_list[2]
    row ={
        'question': question,
        'doc-1': doc_1,
        'doc-2': doc_2,
        'doc-3': doc_3,
        'cohere-rank': rank_score,
        'llm-rank' : score_list
        }
    
    print("\n question response:", final_response)
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    #if j==1:
       #break

results_df.to_excel("corrective_rag_test_results.xlsx", index=False)
print("df saved")

