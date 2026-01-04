import os
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

CHUNKS_DIR = "data/chunks"

# -----------------------------
# Charger tous les chunks TXT et créer des Documents
# -----------------------------
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

# -----------------------------
# Connexion Neo4j
# -----------------------------
graph = Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="bd_juridistique"
)

# vider le graphe pour test propre
#graph.query("MATCH (n) DETACH DELETE n")
#print("✔ Graphe Neo4j vidé pour test")

# -----------------------------
# LLM pour transformer les chunks en graphe
# -----------------------------
llm_graph = Ollama(model="deepseek-r1:8b")

transformer = LLMGraphTransformer(
    llm=llm_graph,
    allowed_nodes=["Article", "Loi", "Concept"],
    allowed_relationships=["APPARTIENT_A", "PERMET", "OBLIGE"]
)
""" allowed_nodes=[ "Loi", "code", "Article", "Chapitre", "Titre", "Sous-titre", "Concept", "Personne", "Entreprise", "Organisation", "Action", "Droit", "Obligation" ],
    allowed_relationships=[ "APPARTIENT_A", "TRAITE_DE", "REGIT", "CONCERNE", "OBLIGE", "PERMET", "FAIT_PARTIE_DE", "CITE", "EXEMPLE_DE" ] """
    
# -----------------------------
# Transformer les chunks en graphe
# -----------------------------

# Limiter les chunks
all_docs = all_docs[:10]
graph_docs = []

for i, doc in enumerate(all_docs):
    print(f"⏳ Chunk {i+1}/{len(all_docs)}")
    docs = transformer.convert_to_graph_documents([doc])
    graph_docs.extend(docs)

print("✔ Graphe créé dans Neo4j à partir des chunks")

# -----------------------------
# Vérification du graphe
# -----------------------------
node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

print(f"📊 Nœuds créés : {node_count}")
print(f"🔗 Relations créées : {rel_count}")

if node_count == 0:
    raise ValueError("❌ Aucun nœud créé → le LLMGraphTransformer n’a rien extrait")

# -----------------------------
# Test Cypher direct (debug)
# -----------------------------
sample = graph.query("""
MATCH (a:Article)
RETURN a.text AS text
LIMIT 1
""")

if not sample:
    print("⚠️ Aucun Article trouvé dans le graphe")
else:
    print("🧪 Exemple d’Article trouvé dans Neo4j :")
    print(sample[0]["text"][:300], "...")
    
# -----------------------------
# Création de la chaîne Graph-QA
# -----------------------------
graph_qa = GraphCypherQAChain.from_llm(
    llm=llm_graph,
    graph=graph,
    verbose=True
)

# -----------------------------
# Interface interactive pour tester le Graph-RAG
# -----------------------------
print("\n💡 Pose tes questions sur le graphe juridique (tape 'exit' pour quitter)\n")

while True:
    question = input("❓ Question : ")
    if question.lower() in ["exit", "quit"]:
        break

    answer = graph_qa.run(question)
    print("🤖 Réponse :", answer)
