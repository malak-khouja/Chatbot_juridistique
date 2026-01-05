#---------------------------------------------------------------------
# Fichier à exécuter une seule fois ou si tu ajoutes/modifies des PDF
#---------------------------------------------------------------------

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Chemins
# -----------------------------
PDF_DIR = "app/data/pdfs"
CHROMA_DIR = "app/data/chroma"
CHUNKS_DIR = "app/data/chunks"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# -----------------------------
# Nettoyage simple du texte
# -----------------------------
def clean_text(text: str) -> str:
    # 1. Supprimer retours à la ligne inutiles et espaces multiples
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    # 2. Supprimer points de remplissage ou traits
    text = re.sub(r"[\.…]{3,}", "", text)
    
    # 3. Supprimer numéros de page simples
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    
    # 4. Supprimer en-têtes éditoriaux répétitifs
    headers = [
        r"REPUBLIQUE TUNISIENNE",
        r"Imprimerie Officielle de la République Tunisienne",
        r"Publications de l’Imprimerie Officielle de la République Tunisienne"
    ]
    for h in headers:
        text = re.sub(h, "", text, flags=re.IGNORECASE)
    
    # 5. Supprimer le sommaire (table des matières)
    lines = text.splitlines()
    cleaned_lines = []
    in_toc = False
    for line in lines:
        l = line.strip()
        # Début sommaire
        if re.match(r"(Annexe|sommaire|table\s+des\s+mati[eè]res|sujet\s+articles\s+pages|^page$)", l, re.IGNORECASE):
            in_toc = True
            continue
        if in_toc:
            # Ligne de type index : numéro seul, plage, points, majuscules
            if re.fullmatch(r"\d+", l) or re.search(r"\d+\s*(à|-|et)\s*\d+", l):
                continue
            if re.search(r"\.{2,}|…{2,}", l):
                continue
            if len(l) < 50 and re.match(r"^[A-ZÉÈÀÙÇ'\s\-]{3,}:?$", l):
                continue
            # Fin sommaire
            in_toc = False
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # 6. Standardiser les articles : Article, Art., article → Article
    text = re.sub(r"\b(Art\.|article)\b", "Article", text, flags=re.IGNORECASE)
    
    return text.strip()

# -----------------------------
# Chunking adaptatif
# -----------------------------
def get_chunk_params(num_pages):
    if num_pages < 10:
        return 600, 100
    elif num_pages < 50:
        return 1000, 150
    else:
        return 1500, 200

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# -----------------------------
# Extraction, nettoyage et chunking
# -----------------------------
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
# Indexation avec Chroma
# -----------------------------
vectordb = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
vectordb.persist()
print(f"✔ {len(all_docs)} chunks indexés")

import re
import os
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# -----------------------------
# Dossier des chunks
# -----------------------------
CHUNKS_DIR = "data/chunks"

# -----------------------------
# Connexion Neo4j
# -----------------------------
graph = Neo4jGraph(
    url="neo4j+s://b8f76dbc.databases.neo4j.io",
    username="neo4j",
    password="N_AQqAk0Lp4pHzsttUOXcVnveCIcOXPsTJLFgCNKB40",
    timeout=60
)

# ⚠️ Vider le graphe avant test
graph.query("MATCH (n) DETACH DELETE n")
print("Graphe Neo4j vidé pour test")

# -----------------------------
# Regex pour extraire la structure
# -----------------------------
ARTICLE_RE = re.compile(r"(Article\s+\d+)", re.IGNORECASE)
CHAPITRE_RE = re.compile(r"(Chapitre\s+[IVXLC]+)", re.IGNORECASE)
TITRE_RE = re.compile(r"(Titre\s+[IVXLC]+)", re.IGNORECASE)

def extract_entities(text):
    titres = set(TITRE_RE.findall(text))
    chapitres = set(CHAPITRE_RE.findall(text))
    articles = set(ARTICLE_RE.findall(text))
    return titres, chapitres, articles

# -----------------------------
# Ajouter les entités et relations dans Neo4j
# -----------------------------
def add_to_graph(titres, chapitres, articles, text_content):
    for t in titres:
        graph.query("MERGE (:Titre {name:$n})", {"n": t})
    for c in chapitres:
        graph.query("MERGE (:Chapitre {name:$n})", {"n": c})
    for a in articles:
        graph.query("MERGE (:Article {name:$n, content:$content})", {"n": a, "content": text_content[:500]})

    # Chapitre → Titre
    for c in chapitres:
        for t in titres:
            graph.query("""
                MATCH (c:Chapitre {name:$c}), (t:Titre {name:$t})
                MERGE (c)-[:FAIT_PARTIE_DE]->(t)
            """, {"c": c, "t": t})

    # Article → Chapitre
    for a in articles:
        for c in chapitres:
            graph.query("""
                MATCH (a:Article {name:$a}), (c:Chapitre {name:$c})
                MERGE (a)-[:FAIT_PARTIE_DE]->(c)
            """, {"a": a, "c": c})

# -----------------------------
# Construction du graphe à partir des chunks existants
# -----------------------------
def build_graph_from_chunks(chunks):
    for i, doc in enumerate(chunks):
        titres, chapitres, articles = extract_entities(doc.page_content)
        add_to_graph(titres, chapitres, articles, doc.page_content)
        if i % 100 == 0:
            print(f"✔ {i} chunks traités")
# -----------------------------
# Peupler le graphe avec les documents
# -----------------------------
print(" Construction du graphe Neo4j...")
build_graph_from_chunks(all_docs)
print(" Graphe Neo4j construit avec succès")
