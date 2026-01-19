import sys
import os
from pathlib import Path

from langchain_experimental.graph_transformers.llm import PromptTemplate

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from app.config import (
    NEO4J_URI as CFG_NEO4J_URI,
    NEO4J_USER as CFG_NEO4J_USER,
    NEO4J_PASSWORD as CFG_NEO4J_PASSWORD,
    LLM_MODEL
)

CHUNKS_DIR = "app/data/chunks"
PROGRESS_FILE = "progress.txt"

# progression

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_progress(i):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(i))

# Charger les chunks

all_docs = []

for pdf_folder in os.listdir(CHUNKS_DIR):
    pdf_path = os.path.join(CHUNKS_DIR, pdf_folder)
    if not os.path.isdir(pdf_path):
        continue

    for chunk_file in sorted(os.listdir(pdf_path)):
        if not chunk_file.endswith(".txt"):
            continue

        with open(os.path.join(pdf_path, chunk_file), "r", encoding="utf-8") as f:
            text = f.read()
            chunk_id = f"{pdf_folder}_{chunk_file}"

            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_folder,
                    "chunk_id": chunk_id
                }
            )
            all_docs.append(doc)

print(f"✔ {len(all_docs)} chunks chargés depuis {CHUNKS_DIR}")


# Connexion Neo4j

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", CFG_NEO4J_URI),
    username=os.getenv("NEO4J_USER", CFG_NEO4J_USER),
    password=os.getenv("NEO4J_PASSWORD", CFG_NEO4J_PASSWORD),
    timeout=60
)

#LLM pour transformer les chunks en graphe

llm_graph = Ollama(model=LLM_MODEL)

transformer = LLMGraphTransformer(
    llm=llm_graph,
    allowed_nodes=[
        "Loi", "Code", "Article", "Chapitre", "Titre",
        "Sous-titre", "Concept", "Personne",
        "Entreprise", "Organisation", "Action",
        "Droit", "Obligation"
    ],
    allowed_relationships=[
        "APPARTIENT_A", "TRAITE_DE", "REGIT",
        "CONCERNE", "OBLIGE", "PERMET",
        "FAIT_PARTIE_DE", "CITE", "EXEMPLE_DE"
    ]
)

# Ingestion Graphe

start_index = load_progress()

for i, doc in enumerate(all_docs[start_index:], start=start_index):

    # Skip si source déjà traitée
    existing = graph.query(
        "MATCH (n {chunk_id: $id}) RETURN count(n) AS c",
        {"id": doc.metadata["chunk_id"]}
    )

    if existing and existing[0]["c"] > 0:
        print(f"⏭ Chunk déjà présent : {doc.metadata['chunk_id']}")
        continue

    print(f"⏳ Chunk {i+1}/{len(all_docs)} : {doc.metadata['chunk_id']}")

    # Transformer → GraphDocuments
    graph_docs = transformer.convert_to_graph_documents([doc])

    #  Insertion Neo4j 
    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True
    )

    # CRÉATION DES RELATIONS 
    graph.query(
        """
        MATCH (n {chunk_id: $id})
        MERGE (n)-[:EXTRACTED_FROM]->(:Chunk {source: $source})
        """,
        {"id": doc.metadata["chunk_id"], "source": doc.metadata["source"]}
    )

    save_progress(i + 1)

print(" Graphe créé dans Neo4j à partir des chunks")

# Vérification

node_count = graph.query("MATCH 👎 RETURN count(n) AS c")[0]["c"]
rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

print(f" Nœuds créés : {node_count}")
print(f" Relations créées : {rel_count}")

if node_count == 0:
    raise ValueError(" Aucun nœud créé → extraction échouée")

# Test Article (debug)

sample = graph.query("""
MATCH 👎
WHERE n.text IS NOT NULL
RETURN labels(n), n.text
LIMIT 1
""")

if sample:
    print(" Exemple de nœud trouvé :")
    print(sample[0]["labels(n)"], sample[0]["n.text"][:300], "...")
else:
    print(" Aucun texte trouvé")

# Graph QA (FIX sécurité)

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are an expert Neo4j Cypher generator.

Schema:
{schema}

Rules:
- Output ONLY a Cypher query
- DO NOT use ``` or any language tags
- DO NOT add explanations
- The query MUST start with MATCH, CALL, or RETURN

Question:
{question}

Cypher:
"""
)

graph_qa = GraphCypherQAChain.from_llm(
    llm=llm_graph,
    graph=graph,
    cypher_prompt=CYPHER_PROMPT,
    verbose=True,
    allow_dangerous_requests=True
)

# Interface interactive

print("\n Pose tes questions sur le graphe juridique (exit pour quitter)\n")

while True:
    q = input(" Question : ")
    if q.lower() in ["exit", "quit"]:
        break

    answer = graph_qa.run(q)
    print(" Réponse :", answer)
    
    
# # contexte STRUCTUREL depuis Neo4j
# def get_graph_context(question: str, limit: int = 5) -> str:
#     """
#     Retrieve structured legal context from Neo4j graph built with LLMGraphTransformer.
    
#     Supports multiple entity types:
#     - Articles, Chapitres, Titres
#     - Loi, Code, Concept
#     - Personne, Entreprise, Organisation
#     - Droit, Obligation, Action
    
#     Args:
#         question: User's question
#         limit: Maximum number of results to return
    
#     Returns:
#         Formatted context string with entities and relationships
#     """
#     if graph is None:
#         return ""
    
#     try:
#         # Extract keywords (length > 3 to filter common words)
#         keywords = [w.lower() for w in question.split() if len(w) > 3]
        
#         if not keywords:
#             return ""
        
#         # 1. MULTI-ENTITY SEARCH: Find all relevant legal entities
#         cypher_entities = """
#         MATCH (n)
#         WHERE labels(n)[0] IN ['Article', 'Chapitre', 'Titre', 'Loi', 'Code', 
#                                'Concept', 'Droit', 'Obligation', 'Action',
#                                'Personne', 'Entreprise', 'Organisation']
#         AND (n.text IS NOT NULL OR n.id IS NOT NULL)
#         AND any(k IN $keys WHERE 
#             toLower(coalesce(n.text, '')) CONTAINS k OR 
#             toLower(coalesce(n.id, '')) CONTAINS k)
#         RETURN labels(n)[0] AS entity_type, 
#                coalesce(n.text, n.id) AS entity_text,
#                n.chunk_id AS chunk_id
#         LIMIT $limit
#         """
        
#         # 2. RELATIONSHIP SEARCH: Find related entities
#         cypher_relationships = """
#         MATCH (n)-[r]->(m)
#         WHERE (any(k IN $keys WHERE 
#                 toLower(coalesce(n.text, '')) CONTAINS k OR 
#                 toLower(coalesce(n.id, '')) CONTAINS k))
#         AND labels(n)[0] IN ['Article', 'Chapitre', 'Titre', 'Loi', 'Code', 
#                              'Concept', 'Droit', 'Obligation']
#         RETURN labels(n)[0] AS source_type,
#                coalesce(n.text, n.id) AS source_text,
#                type(r) AS rel_type,
#                labels(m)[0] AS target_type,
#                coalesce(m.text, m.id) AS target_text
#         LIMIT $limit
#         """
        
#         # Execute queries
#         entities = []
#         relationships = []
        
#         try:
#             entities = graph.query(cypher_entities, {"keys": keywords, "limit": limit})
#         except Exception as e:
#             logging.debug(f"Entity search failed: {e}")
        
#         try:
#             relationships = graph.query(cypher_relationships, {"keys": keywords, "limit": limit})
#         except Exception as e:
#             logging.debug(f"Relationship search failed: {e}")
        
#         # Format results
#         context_parts = []
        
#         # Format entities
#         for entity in entities:
#             entity_type = entity.get("entity_type", "Entity")
#             entity_text = entity.get("entity_text", "")
            
#             if entity_text:
#                 # Add emoji based on type
#                 emoji = {
#                     "Article": "📋",
#                     "Chapitre": "📖",
#                     "Titre": "📕",
#                     "Loi": "⚖️",
#                     "Code": "📚",
#                     "Concept": "💡",
#                     "Droit": "✅",
#                     "Obligation": "⚠️",
#                     "Action": "🔨"
#                 }.get(entity_type, "🔹")
                
#                 context_parts.append(f"{emoji} [{entity_type}] {entity_text[:200]}")
        
#         # Format relationships
#         for rel in relationships:
#             source_type = rel.get("source_type", "")
#             source_text = rel.get("source_text", "")[:80]
#             rel_type = rel.get("rel_type", "LIEN")
#             target_type = rel.get("target_type", "")
#             target_text = rel.get("target_text", "")[:80]
            
#             if source_text and target_text:
#                 context_parts.append(
#                     f"🔗 [{source_type}] {source_text} --{rel_type}--> [{target_type}] {target_text}"
#                 )
        
#         return "\n\n".join(context_parts[:10])  # Limit to 10 results
        
#     except Exception as e:
#         logging.warning(f"Graph context retrieval failed: {e}")
#         return ""