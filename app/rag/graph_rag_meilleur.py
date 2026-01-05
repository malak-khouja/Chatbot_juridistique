import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from app.config import NEO4J_URI as CFG_NEO4J_URI, NEO4J_USER as CFG_NEO4J_USER, NEO4J_PASSWORD as CFG_NEO4J_PASSWORD, EMBEDDING_MODEL, LLM_MODEL

CHUNKS_DIR = "app/data/chunks"


# Charger tous les chunks TXT et créer des Documents

all_docs = []

for pdf_folder in os.listdir(CHUNKS_DIR):
    pdf_path = os.path.join(CHUNKS_DIR, pdf_folder)
    if not os.path.isdir(pdf_path):
        continue

    for chunk_file in sorted(os.listdir(pdf_path)):
        if not chunk_file.endswith(".txt"):
            continue
        chunk_path = os.path.join(pdf_path, chunk_file)
        with open(chunk_path, "r", encoding="utf-8") as f:
            text = f.read()
            doc = Document(page_content=text, metadata={"source": pdf_folder})
            all_docs.append(doc)

print(f"✔ {len(all_docs)} chunks chargés depuis {CHUNKS_DIR}")



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

# vider le graphe pour test propre
graph.query("MATCH (n) DETACH DELETE n")

# LLM pour transformer les chunks en graphe

llm_graph = Ollama(model=LLM_MODEL)

transformer = LLMGraphTransformer(
    llm=llm_graph,
    allowed_nodes=[ "Loi", "code", "Article", "Chapitre", "Titre", "Sous-titre", "Concept", "Personne", "Entreprise", "Organisation", "Action", "Droit", "Obligation" ],
    allowed_relationships=[ "APPARTIENT_A", "TRAITE_DE", "REGIT", "CONCERNE", "OBLIGE", "PERMET", "FAIT_PARTIE_DE", "CITE", "EXEMPLE_DE" ]
)

graph_docs = []

for i, doc in enumerate(all_docs):
    print(f"⏳ Chunk {i+1}/{len(all_docs)}")
    docs = transformer.convert_to_graph_documents([doc])
    graph_docs.extend(docs)

print(" Graphe créé dans Neo4j à partir des chunks")

# Vérification du graphe

node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

print(f" Nœuds créés : {node_count}")
print(f"relations créées : {rel_count}")

if node_count == 0:
    raise ValueError(" Aucun nœud créé → le LLMGraphTransformer n’a rien extrait")

# Test Cypher direct (debug)

sample = graph.query("""
MATCH (a:Article)
RETURN a.text AS text
LIMIT 1
""")

if not sample:
    print(" Aucun Article trouvé dans le graphe")
else:
    print(" Exemple d’Article trouvé dans Neo4j :")
    print(sample[0]["text"][:300], "...")
    

# Création de la chaîne Graph-QA

graph_qa = GraphCypherQAChain.from_llm(
    llm=llm_graph,
    graph=graph,
    verbose=True
)


# Interface interactive pour tester le Graph-RAG

print("\n Pose tes questions sur le graphe juridique (tape 'exit' pour quitter)\n")

while True:
    question = input(" Question : ")
    if question.lower() in ["exit", "quit"]:
        break

    answer = graph_qa.run(question)
    print(" Réponse :", answer)
