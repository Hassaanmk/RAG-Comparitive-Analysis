import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Dict
from pydantic import BaseModel,Field
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph 
from typing import List
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import json

# Load environment variables and set up OpenAI API key
from dotenv import load_dotenv
load_dotenv()

# Set up the Neo4j Graph connection
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
#graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_password,
)

#before running first time always run on Neo4j webpage thr following query for converting each node name as a id: CREATE INDEX FOR (n:Entity) ON (n.id)
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]") #double checks if id index is not present


class Entities(BaseModel):
    """Identifying information about entities with their types."""
    entities: List[str] = Field(
        ...,
        description="List of entities along with their types. Each entity should be given with it's type as For e.g: 'Treasure Island': 'Location'"
                    "with 'name' as the entity's name and 'type' as the entity's type (as, Person, Organization, Clothing, Concept, Food, Location, Object, Occupation, Organization,Quantity)."
    )
#Define LLM for generating responses based on entities
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Set up a prompt template for entity extraction
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting entities and there types from the Question.",
        ),
        (
            "human",
            "Use the given format and example to extract from the following input: {question}",
        ),
    ]
)
#Define the entity extraction chain
entity_chain = prompt | llm.with_structured_output(Entities)

def relationship_refiner(graph, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant that converts each entity and relationship present in the Graph into a single sentence."
         "Do for all entities with id and relationship r and give a single sentence for each, precise and accurate in a seperate line."),
        ("human", "Graph:\n{graph}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "graph": graph,
        })
    return response

def extract_entities_and_types(entities: Entities):
    entities_dict = {}
    # Extract the entities list from the Entities object
    entities_list = entities.entities
    # Iterate through the entities list (should contain entity data in a specific format)
    for entity_str in entities_list:
        try:
            name, entity_type = entity_str.split(": ", 1)  # Split at the first occurrence of ": "
            # Add the entity and its type to the dictionary
            if name and entity_type:
                entities_dict[name] = entity_type
        except ValueError:
            print(f"Error processing entity string: {entity_str}")
    
    return entities_dict

def structured_retriever(question: str) -> str:
    result = ""
    # Extract entities from the question using the entity_chain
    entities = entity_chain.invoke({"question": question})
    print("Extracted entities:", entities)  # Debugging extracted entities
    entities_dict = extract_entities_and_types(entities)
    entity_names = list(entities_dict.keys())

    for entity in entity_names:
        try:
            print(f"Generating query for entity (id): {entity}")
            # Query Neo4j to retrieve related nodes and relationships for each entity
            response = graph.query(
                """
                MATCH (n {id: $entity})  // Match based on the 'id' property
                OPTIONAL MATCH (n)-[r]->(m)  // Outgoing relationships
                OPTIONAL MATCH (n)<-[r2]-(m2)  // Incoming relationships
                RETURN n, r, m, r2, m2
                LIMIT 100
                """,
                {"entity": entity}  # Pass the entity id as a parameter to the query
            )
        except:
            print(f"\n error while extracting entity(id): entity (id): {entity}")

        # Combine the results into a string for use as context
        if response:
            result += "\n".join([str(el) for el in response])  # Convert response to string
        else:
            print(f"\n No results found for entity (id): {entity}\n")

    relationship_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
    refined_relationship=relationship_refiner(result, relationship_llm) #Refining relationships through llm
    #print("normal graph: ", result)
    print("refined-relationship: ",refined_relationship)
    return result,refined_relationship,entity_names

def response_evaluator(question, answer, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant evaluating how relevant a response is to a users question. "
         "You must give a numeric score between 0 (not related at all) and 10 (perfectly relevant). "
         "Give only the score as a number."),
        ("human", "Question:\n{question}\n\nResponse:\n{response}\n\nScore (0-10):")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "question": question,
        "response": answer
        })
    try:
        score = int(response.strip())
    except:
        score = -1  # If parsing fails
    return score

def relationships_evaluator(question,relationship,llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant evaluating how relevant the relationships of entities extracted from knowledge graph are to the question provided by user"
         "You must give a combined numeric score between 0 (not related at all) and 10 (perfectly relevant). "
         "Give only the score as a number."),
        ("human", "Question:\n{question}\n\nrelationships:\n{relationship}\n\nScore (0-10):")
    ])
    chain = prompt | llm | StrOutputParser()
    #for i, doc in enumerate(reranked_docs):
    response = chain.invoke({
            "question": question,
            "relationship": relationship})
    try:
        score = int(response.strip())
    except:
        score = -1  # If parsing fails
    return score

def full_process(question: str) -> str:
    # Retrieve relationships and relevant information from Neo4j based on the entities
    raw_relationships,refined_relationships,entities = structured_retriever(question)
    # Set up a prompt for the LLM to generate a response
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in extracting information from entities and relationships shared and giving to the point answers."),
            ("human", "Here is the extracted context: {context}. Please generate a straightforward and precise response based on this context: {question}"),
        ]
    )
    # Create a prompt using the retrieved context and the original question
    full_prompt = prompt.invoke({"context": refined_relationships, "question": question})
    # Use the LLM chain to generate the response
    response= llm.invoke(full_prompt)
    response_text = response.content if hasattr(response, 'content') else "No response content found"
    return response_text,refined_relationships,entities

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
    'No. ',
    'question',  
    'entities',
    'relationships',
    'final-response',
    'relationship-score',
    'response-score'
]) 

j=0
for question in rag_test_questions:
    print("\n" + "="*80)
    print("question:", question)
    j=j+1
    
    final_response,relationships,entities= full_process(question)
    relation_score=relationships_evaluator(question,relationships,ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0))
    llm_score= response_evaluator(question, final_response, ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0))

    row ={
        'No. ': j,
        'question': question,  
        'entities': entities,
        'relationships': relationships,
        'final-response': final_response,
        'relationship-score': relation_score,
        'response-score': llm_score
        }
    
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df.to_excel("GraphRAG_test_results.xlsx", index=False)
    #if j==1:
       #break

results_df.to_excel("GraphRAG_test_results.xlsx", index=False)
print("df saved")



