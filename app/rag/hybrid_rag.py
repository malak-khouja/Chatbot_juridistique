import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
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


# contexte STRUCTUREL depuis Neo4j
def get_graph_context(question, limit=5):
    if graph is None:
        return ""
    
    # Extraire mots-clés pertinents (filtrer mots courts)
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    
    cypher = """
    MATCH (a:Article)
    WHERE any(k IN $keys WHERE toLower(a.name) CONTAINS k OR toLower(a.content) CONTAINS k)
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