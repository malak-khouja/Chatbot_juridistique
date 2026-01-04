import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from app.config import NEO4J_URI as CFG_NEO4J_URI, NEO4J_USER as CFG_NEO4J_USER, NEO4J_PASSWORD as CFG_NEO4J_PASSWORD


CHROMA_DIR = "data/chroma"

# Chargement de la base vectorielle (Chroma)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Connexion au graphe Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", CFG_NEO4J_URI)
NEO4J_USER = os.getenv("NEO4J_USER", CFG_NEO4J_USER)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", CFG_NEO4J_PASSWORD)

try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
except Exception as exc:
    print(f"[WARN] Connexion Neo4j impossible ({exc}); le contexte graphe sera vide.")
    graph = None

# Modèle LLM (Ollama)
llm = OllamaLLM(model="mistral")
    
# contexte STRUCTUREL depuis Neo4j
def get_graph_context(question, limit=5):
    if graph is None:
        return ""
    keywords = question.lower().split()
    cypher = """
    MATCH (a:Article)
    WHERE any(k IN $keys WHERE toLower(a.content) CONTAINS k)
    OPTIONAL MATCH (a)-[:FAIT_PARTIE_DE]->(c:Chapitre)
    OPTIONAL MATCH (c)-[:FAIT_PARTIE_DE]->(t:Titre)
    RETURN a.name AS article,
           a.content AS content,
           c.name AS chapitre,
           t.name AS titre
    LIMIT $limit
    """

    try:
        results = graph.query(
            cypher,
            {"keys": keywords, "limit": limit}
        )
    except Exception as exc:
        print(f"[WARN] Requête Neo4j échouée ({exc}); contexte graphe ignoré.")
        return ""

    context = ""
    for r in results:
        context += f"""
        {r.get("titre","")}
        {r.get("chapitre","")}
        {r["article"]} :
        {r["content"]}
        """
    return context.strip()

# Prompt HYBRIDE
prompt = ChatPromptTemplate.from_template("""
Tu es un assistant juridique spécialisé en droit tunisien.

RÈGLES STRICTES :
- Réponds UNIQUEMENT à partir des textes fournis.
- N'invente JAMAIS de loi ou d'article.
- Ne cite JAMAIS le droit français ou d'autres pays.
- Si l'information n'est pas présente, dis :
  "Cette information n'est pas disponible dans les documents fournis."

CONTEXTE JURIDIQUE STRUCTUREL (Neo4j) :
{graph_context}

CONTEXTE TEXTUEL (Recherche sémantique) :
{vector_context}

QUESTION :
{question}

RÉPONSE :
""")

# Chaîne RAG HYBRIDE
def hybrid_rag_answer(question: str) -> str:
    # Contexte vectoriel
    docs = retriever.invoke(question)
    vector_context = "\n\n".join(d.page_content for d in docs)

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
    return answer

