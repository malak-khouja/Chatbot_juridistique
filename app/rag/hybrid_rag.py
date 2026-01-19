import os
import re
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from app.config import NEO4J_URI as CFG_NEO4J_URI, NEO4J_USER as CFG_NEO4J_USER, NEO4J_PASSWORD as CFG_NEO4J_PASSWORD, EMBEDDING_MODEL, LLM_MODEL


CHROMA_DIR = "app/data/chroma"

# Chargement de la base vectorielle (Chroma)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Connexion au graphe Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", CFG_NEO4J_URI)
NEO4J_USER = os.getenv("NEO4J_USER", CFG_NEO4J_USER)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", CFG_NEO4J_PASSWORD)

try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        timeout=60
    )
except Exception as exc:
    print(f"[WARN] Connexion Neo4j impossible ({exc}); le contexte graphe sera vide.")
    graph = None

# Modèle LLM (Ollama) avec configuration pour réponses complètes
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.1,
    num_predict=500,
    base_url="http://127.0.0.1:11434"
)

# Cypher Prompt Template for LLM-based query generation
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are an expert Neo4j Cypher query generator for a Tunisian legal knowledge graph.

Graph Schema:
{schema}

Rules:
- Output ONLY a valid Cypher query
- DO NOT use ``` or language tags
- DO NOT add explanations or comments
- The query MUST start with MATCH, CALL, or RETURN
- Focus on finding Articles, Concepts, and their relationships
- Return results that directly answer the question
- Limit results to 5 nodes maximum

Question:
{question}

Cypher Query:
"""
)

# Initialize GraphCypherQAChain for dynamic context retrieval
if graph is not None:
    try:
        graph_qa_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=CYPHER_PROMPT,
            verbose=False,
            allow_dangerous_requests=True
        )
    except Exception as e:
        logging.warning(f"Failed to initialize GraphCypherQAChain: {e}")
        graph_qa_chain = None
else:
    graph_qa_chain = None


def get_graph_context(question: str) -> str:
    if graph is None or graph_qa_chain is None:
        return ""
    
    try:
        # Generate and execute Cypher query using LLM
        return graph_qa_chain.run(question)
        
    except Exception as e:
        logging.warning(f"Graph context retrieval failed: {e}")
        return ""

# Prompt HYBRIDE
prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé en droit tunisien.

RÈGLES STRICTES :
- Réponds UNIQUEMENT à partir des textes fournis ci-dessous.
- Donne une réponse COURTE, PRÉCISE et DIRECTE (3-5 points maximum).
- Cite UNIQUEMENT les articles les plus pertinents (pas tous les articles).
- NE RÉPÈTE JAMAIS le même article ou point.
- N'invente JAMAIS de loi ou d'article.
- Ne cite JAMAIS le droit français ou d'autres pays.
- Si l'information n'est pas présente, dis :
  "Cette information n'est pas disponible dans les documents fournis."

FORMATAGE OBLIGATOIRE :
- Commence par une phrase de réponse directe
- Utilise des tirets (-) pour lister les points clés (3-5 maximum)
- Mentionne l'article entre parenthèses : (Article XXX)
- Une ligne vide entre chaque point
- Sois CONCIS - maximum 5 phrases au total

CONTEXTE JURIDIQUE STRUCTUREL (Neo4j) :
{graph_context}

CONTEXTE TEXTUEL (Recherche sémantique) :
{vector_context}

QUESTION :
{question}

RÉPONSE (courte et précise) :
""")

def deduplicate_context(text: str) -> str:
    """Supprime les chunks dupliqués du contexte"""
    lines = text.split("\n\n")
    seen = set()
    unique_lines = []
    for line in lines:
        # Normaliser pour comparaison (minuscules, espaces supprimés)
        normalized = line.lower().strip()[:250]
        if normalized not in seen and len(line.strip()) > 20:
            seen.add(normalized)
            unique_lines.append(line)
    return "\n\n".join(unique_lines)


def format_answer(text: str) -> str:
    """Améliore le formatage de la réponse et supprime les redondances"""
    import re
    
    # Ajouter un saut de ligne après les phrases (. ! ?)
    text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
    
    # Ajouter des sauts de ligne avant et après les tirets de liste
    text = re.sub(r'\n-\s+', '\n\n- ', text)
    text = re.sub(r'-\s+', '- ', text)
    
    # Supprimer les listes dupliquées (ex: même article listé plusieurs fois)
    lines = text.split("\n")
    unique_lines = []
    seen_bullets = set()
    for line in lines:
        if line.strip().startswith("-"):
            # Extraire le texte du bullet
            bullet_text = line.strip()[1:].strip().lower()[:60]
            if bullet_text not in seen_bullets and len(line.strip()) > 5:
                seen_bullets.add(bullet_text)
                unique_lines.append(line)
        else:
            unique_lines.append(line)
    text = "\n".join(unique_lines)
    
    # Nettoyer les sauts de ligne multiples
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    # Supprimer les espaces en fin de ligne
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text.strip()

# Chaîne RAG HYBRIDE
def hybrid_rag_answer(question: str) -> str:
    # Contexte vectoriel
    docs = retriever.invoke(question)
    vector_context = deduplicate_context("\n\n".join(d.page_content for d in docs))

    # Contexte graphe
    graph_context = get_graph_context(question)

    # Prompt final
    formatted_prompt = prompt.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question
    )

    # Appel LLM
    answer = llm.invoke(formatted_prompt)
    
    # Post-traitement : améliorer le formatage et supprimer les redondances
    answer = format_answer(answer)
    
    return answer