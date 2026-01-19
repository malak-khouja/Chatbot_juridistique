
# Fichier à exécuter une seule fois ou si tu ajoutes/modifies des PDF

import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_MODEL, LLM_MODEL
import re
import os
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama

# Chemins
PDF_DIR = "../data/pdfs"
CHROMA_DIR = "../data/chroma"
CHUNKS_DIR = "../data/chunks"
PROGRESS_FILE = "progress.txt"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# -----------------------------
# Nettoyage et Chunking
# -----------------------------

# Nettoyage simple du texte
def clean_text(text: str) -> str:
    # 1. Supprimer retours à la ligne inutiles et espaces multiples
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    # 2. Supprimer le sommaire (table des matières) AVANT les points
    lines = text.splitlines()
    cleaned_lines = []
    in_toc = False  # Commence par inclure par défaut
    
    for line in lines:
        l = line.strip()
        
        # Début sommaire
        if re.match(r"(Annexe|sommaire|table\s+des\s+mati[eè]res|sujet\s+articles\s+pages|^page$)", l, re.IGNORECASE):
            in_toc = True
            continue
        
        # Fin sommaire : détection par formules tunisiennes standards
        if in_toc and re.search(r"(Au nom du peuple|La Chambre des Députés ayant adopté|Le Président de la République)", l, re.IGNORECASE):
            in_toc = False
            # N'inclure que si ce n'est pas juste la formule
            if not re.fullmatch(r"^(Au nom du peuple|La Chambre des Députés ayant adopté|Le Président de la République[^.]*)$", l, re.IGNORECASE):
                cleaned_lines.append(line)
            continue
        
        # Si on est dans le sommaire, skip certaines lignes
        if in_toc:
            # Ligne de type index : numéro seul
            if re.fullmatch(r"\d+", l):
                continue
            # Plages de numéros
            if re.search(r"\d+\s*(à|-|et)\s*\d+", l):
                continue
            # Points de remplissage (dotted leaders) - LÀ on les utilise pour détecter
            if re.search(r"\.{2,}|…{2,}", l):
                continue
            # Lignes tout en majuscules (titres de sommaire)
            if len(l) < 50 and re.match(r"^[A-ZÉÈÀÙÇ'\s\-]{3,}:?$", l):
                continue
            # Si ce n'est pas une ligne à skipper, on sort du sommaire
            in_toc = False
        
        # Ajouter la ligne au résultat
        if not in_toc or not re.fullmatch(r"\d+", l):
            cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # 3. Supprimer points de remplissage ou traits (APRÈS suppression sommaire)
    text = re.sub(r"[\.…]{3,}", "", text)
    
    # 4. Supprimer numéros de page simples
    # Amélioration : supprimer les numéros tout seuls sur une ligne
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Variantes avec tirets
    text = re.sub(r"\n\s*[-–—]\s*\d+\s*[-–—]\s*\n", "\n", text)
    # Lignes contenant UNIQUEMENT un numéro
    lines = text.split("\n")
    lines = [line for line in lines if not re.fullmatch(r"\s*\d+\s*", line)]
    text = "\n".join(lines)
    
    # 5. Supprimer en-têtes éditoriaux répétitifs
    headers = [
        r"REPUBLIQUE TUNISIENNE",
        r"Imprimerie Officielle de la République Tunisienne",
        r"Publications de l'Imprimerie Officielle de la République Tunisienne"
    ]
    for h in headers:
        text = re.sub(h, "", text, flags=re.IGNORECASE)
    
    # 6. Standardiser les articles : Article, Art., article → Article
    text = re.sub(r"\b(Art\.|article)\b", "Article", text, flags=re.IGNORECASE)
    
    return text.strip()


# Chunking adaptatif

def get_chunk_params(num_pages):
    if num_pages < 10:
        return 600, 100
    elif num_pages < 50:
        return 1000, 150
    else:
        return 1500, 200

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Extraction, nettoyage et chunking

all_docs = []

for pdf in os.listdir(PDF_DIR):
    if not pdf.endswith(".pdf"):
        continue

    path = os.path.join(PDF_DIR, pdf)
    loader = PyPDFLoader(path)
    documents = loader.load()

    chunk_size, overlap = get_chunk_params(len(documents))

    # Chunking hiérarchique : Loi → Article → Ligne → Mot
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            r"(Loi\s+n°\s*\d+-\d+|Code\s+des\s+[A-Za-zÀ-ÿ\s]+)",
            r"Livre",
            r"Titre",#section
            r"Sous[- ]?Titre",
            "Article",
            "\n\n",
            "\n",
            " "
        ]
    )

    chunks = splitter.split_documents(documents)

    # Créer un sous-dossier pour ce PDF
    pdf_name = os.path.splitext(pdf)[0]
    pdf_chunk_dir = os.path.join(CHUNKS_DIR, pdf_name)
    os.makedirs(pdf_chunk_dir, exist_ok=True)

    for i, c in enumerate(chunks):
        c.page_content = clean_text(c.page_content)
        c.metadata["source"] = pdf
        all_docs.append(c)

        # Sauvegarder chaque chunk dans un fichier
        chunk_file = os.path.join(pdf_chunk_dir, f"chunk_{i}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(c.page_content)

    print(f"✔ {len(chunks)} chunks sauvegardés pour {pdf}")


# -----------------------------
#  Rag
# -----------------------------

# Indexation avec Chroma

vectordb = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
vectordb.persist()
print(f"✔ {len(all_docs)} chunks indexés")

# -----------------------------
#Graph RAG
# -----------------------------

# Progress tracking functions
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_progress(i):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(i))

# Connexion Neo4j

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    timeout=60
)

# LLM pour transformer les chunks en graphe
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

# Charger les chunks depuis les fichiers sauvegardés
chunk_docs = []
for pdf_folder in sorted(os.listdir(CHUNKS_DIR)):
    pdf_path = os.path.join(CHUNKS_DIR, pdf_folder)
    if not os.path.isdir(pdf_path):
        continue
    
    for chunk_file in sorted(os.listdir(pdf_path)):
        if not chunk_file.endswith(".txt"):
            continue
        
        chunk_path = os.path.join(pdf_path, chunk_file)
        with open(chunk_path, "r", encoding="utf-8") as f:
            text = f.read()
            chunk_id = f"{pdf_folder}_{chunk_file}"
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_folder,
                    "chunk_id": chunk_id
                }
            )
            chunk_docs.append(doc)

print(f"✔ {len(chunk_docs)} chunks loaded for graph construction")

# Ingestion avec progression
start_index = load_progress()

for i, doc in enumerate(chunk_docs[start_index:], start=start_index):
    # Skip si déjà traité
    existing = graph.query(
        "MATCH (n {chunk_id: $id}) RETURN count(n) AS c",
        {"id": doc.metadata["chunk_id"]}
    )
    
    if existing and existing[0]["c"] > 0:
        print(f"⏭ Chunk déjà présent : {doc.metadata['chunk_id']}")
        continue
    
    print(f"⏳ Processing chunk {i+1}/{len(chunk_docs)} : {doc.metadata['chunk_id']}")
    
    try:
        # Transformer → GraphDocuments
        graph_docs = transformer.convert_to_graph_documents([doc])
        
        # Insertion Neo4j
        graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )
        
        # CRÉATION DES RELATIONS CHUNK
        graph.query(
            """
            MATCH (n {chunk_id: $id})
            MERGE (n)-[:EXTRACTED_FROM]->(:Chunk {source: $source})
            """,
            {"id": doc.metadata["chunk_id"], "source": doc.metadata["source"]}
        )
        
        save_progress(i + 1)
    
    except Exception as e:
        print(f"❌ Error processing chunk {i+1}: {e}")
        continue

print("\n✅ Graph construction complete!")

# Vérification
node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

print(f"\n📈 Graph Statistics:")
print(f"   • Total Nodes: {node_count:,}")
print(f"   • Total Relationships: {rel_count:,}")

if node_count == 0:
    print("⚠️  Warning: No nodes created - extraction may have failed")
else:
    # Sample node
    sample = graph.query("""
    MATCH (n)
    WHERE n.text IS NOT NULL
    RETURN labels(n), n.text
    LIMIT 1
    """)
    
    if sample:
        print(f"\n📋 Sample node:")
        print(f"   Type: {sample[0]['labels(n)']}")
        print(f"   Text: {sample[0]['n.text'][:150]}...")

