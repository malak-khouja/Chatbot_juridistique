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

# Utils progression


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


# graph.query("MATCH 👎 DETACH DELETE n")


# LLM + Transformer


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