import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json
import re
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from app.config import (
    NEO4J_URI as CFG_NEO4J_URI,
    NEO4J_USER as CFG_NEO4J_USER,
    NEO4J_PASSWORD as CFG_NEO4J_PASSWORD,
    LLM_MODEL
)

CHUNKS_DIR = "app/data/chunks"
PROGRESS_FILE = "hybrid_progress.json"


@dataclass
class Entity:
    """Extracted entity"""
    id: str
    type: str
    text: str
    chunk_id: str
    
    def neo4j_key(self) -> str:
        return f"{self.type}_{self.id}"


class HybridGraphExtractor:
    """Hybrid extraction: Fast patterns + Selective LLM for relationships"""
    
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", CFG_NEO4J_URI),
            username=os.getenv("NEO4J_USER", CFG_NEO4J_USER),
            password=os.getenv("NEO4J_PASSWORD", CFG_NEO4J_PASSWORD),
            timeout=300  # Augmenté à 5 minutes pour éviter les timeouts
        )
        
        self.llm = Ollama(model=LLM_MODEL, temperature=0, top_p=0.8)
        
        self.processed_chunks = set()
        self.load_progress()
        
        # Compile regex patterns for Tunisian legal documents
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for legal entity extraction"""
        # Articles: Article 123, Article 123-a, Art. 123, etc.
        self.article_pattern = re.compile(
            r'\b(?:Article|Art\.)\s+(\d+(?:[a-z]|(?:\-\d+)?)?)\b',
            re.IGNORECASE
        )
        
        # Codes: Code civil, Code de commerce, etc.
        self.code_pattern = re.compile(
            r'\b(Code\s+(?:civil|pénal|de\s+commerce|des\s+obligations|des\s+sociétés|du\s+travail|du\s+droit\s+international)[^,.\n]*)\b',
            re.IGNORECASE
        )
        
        # Chapters: Chapitre I, Chapter 1, etc.
        self.chapter_pattern = re.compile(
            r'\b(?:Chapitre|Chapter)\s+([IVXivx]+|\d+)\b',
            re.IGNORECASE
        )
        
        # Titles: Titre I, Title 1, etc.
        self.title_pattern = re.compile(
            r'\b(?:Titre|Title)\s+([IVXivx]+|\d+)\b',
            re.IGNORECASE
        )
        
        # Concepts: specific legal concepts
        self.concept_keywords = {
            'Obligation': r'\b(obligation|duty|devoir)\b',
            'Droit': r'\b(droit|right|droits)\b',
            'Contrat': r'\b(contrat|contract|accord)\b',
            'Personne': r'\b(personne|person|individu)\b',
            'Entreprise': r'\b(entreprise|company|société)\b',
            'Action': r'\b(action|act|action juridique)\b',
        }
    
    def load_progress(self):
        """Load extraction progress"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r") as f:
                    data = json.load(f)
                    self.processed_chunks = set(data.get("processed", []))
                    print(f"✔ Loaded: {len(self.processed_chunks)} chunks already processed")
            except:
                pass
    
    def save_progress(self):
        """Save extraction progress"""
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"processed": list(self.processed_chunks)}, f)
    
    def load_all_chunks(self) -> List[Document]:
        """Load unprocessed chunks"""
        all_docs = []
        
        for pdf_folder in sorted(os.listdir(CHUNKS_DIR)):
            pdf_path = os.path.join(CHUNKS_DIR, pdf_folder)
            if not os.path.isdir(pdf_path):
                continue
            
            for chunk_file in sorted(os.listdir(pdf_path)):
                if not chunk_file.endswith(".txt"):
                    continue
                
                chunk_id = f"{pdf_folder}_{chunk_file}"
                
                if chunk_id in self.processed_chunks:
                    continue
                
                chunk_path = os.path.join(pdf_path, chunk_file)
                with open(chunk_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_folder,
                            "chunk_id": chunk_id
                        }
                    )
                    all_docs.append(doc)
        
        return all_docs
    
    def extract_entities_from_text(self, text: str, chunk_id: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        seen = set()  # Avoid duplicates
        
        # Extract Articles
        for match in self.article_pattern.finditer(text):
            article_num = match.group(1)
            entity_id = f"article_{article_num}"
            if entity_id not in seen:
                entities.append(Entity(
                    id=entity_id,
                    type="Article",
                    text=f"Article {article_num}",
                    chunk_id=chunk_id
                ))
                seen.add(entity_id)
        
        # Extract Codes
        for match in self.code_pattern.finditer(text):
            code_name = match.group(1).strip()
            entity_id = f"code_{code_name.lower().replace(' ', '_')}"
            if entity_id not in seen:
                entities.append(Entity(
                    id=entity_id,
                    type="Code",
                    text=code_name,
                    chunk_id=chunk_id
                ))
                seen.add(entity_id)
        
        # Extract Chapters
        for match in self.chapter_pattern.finditer(text):
            chapter_num = match.group(1)
            entity_id = f"chapter_{chapter_num}"
            if entity_id not in seen:
                entities.append(Entity(
                    id=entity_id,
                    type="Chapitre",
                    text=f"Chapitre {chapter_num}",
                    chunk_id=chunk_id
                ))
                seen.add(entity_id)
        
        # Extract Titles
        for match in self.title_pattern.finditer(text):
            title_num = match.group(1)
            entity_id = f"title_{title_num}"
            if entity_id not in seen:
                entities.append(Entity(
                    id=entity_id,
                    type="Titre",
                    text=f"Titre {title_num}",
                    chunk_id=chunk_id
                ))
                seen.add(entity_id)
        
        # Extract Concepts
        for concept_type, pattern in self.concept_keywords.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concept_text = match.group(1)
                entity_id = f"{concept_type.lower()}_{concept_text.lower()}"
                if entity_id not in seen:
                    entities.append(Entity(
                        id=entity_id,
                        type=concept_type,
                        text=concept_text,
                        chunk_id=chunk_id
                    ))
                    seen.add(entity_id)
        
        return entities
    
    def extract_relationships_with_llm(self, text: str, entities: List[Entity]) -> List[Tuple[str, str, str]]:
        """Use LLM to find relationships between extracted entities (FAST)"""
        if len(entities) < 2:
            return []
        
        # Create entity list for LLM context
        entity_list = "\n".join([
            f"- {e.type}: {e.text} (ID: {e.id})"
            for e in entities
        ])
        
        prompt = PromptTemplate(
            input_variables=["entities", "text"],
            template="""Analyze this legal text and find relationships between entities.

Entities:
{entities}

Text:
{text}

Return ONLY a JSON array of relationships (no explanation):
[
  {{"source_id": "article_123", "relationship": "APPARTIENT_A", "target_id": "code_civil"}},
  {{"source_id": "concept1", "relationship": "TRAITE_DE", "target_id": "article_456"}}
]

Only include relationships that are explicitly mentioned. Return empty array [] if none found."""
        )
        
        try:
            response = self.llm.invoke(prompt.format(entities=entity_list, text=text[:1500]))  # Limit text
            
            # Parse JSON response
            import json
            relationships = json.loads(response)
            
            # Validate relationships (both entities must exist)
            entity_ids = {e.id for e in entities}
            valid_rels = []
            for rel in relationships:
                if rel.get("source_id") in entity_ids and rel.get("target_id") in entity_ids:
                    valid_rels.append((
                        rel["source_id"],
                        rel["relationship"],
                        rel["target_id"]
                    ))
            
            return valid_rels
        
        except json.JSONDecodeError as e:
            logging.debug(f"Failed to parse LLM JSON response: {e}")
            return []
        except Exception as e:
            logging.warning(f"LLM relationship extraction failed: {e}")
            return []

    def sanitize_relationship_type(self, rel_type: str) -> str:
        """Sanitize relationship type to a valid Cypher identifier.
        - Uppercase
        - Replace spaces, hyphens, apostrophes with underscores
        - Remove non-alphanumeric characters (keep underscores)
        - Fallback to 'RELATION' if empty
        """
        if not rel_type:
            return "RELATION"
        s = rel_type.strip().upper()
        s = re.sub(r"\s+", "_", s)
        s = s.replace("'", "_")
        s = s.replace("-", "_")
        s = re.sub(r"[^A-Z0-9_]", "_", s)
        s = re.sub(r"_+", "_", s)
        s = s.strip("_")
        return s or "RELATION"
    
    def insert_chunk_to_neo4j(self, doc: Document) -> int:
        """Extract entities and relationships from chunk, insert to Neo4j"""
        chunk_id = doc.metadata["chunk_id"]
        source = doc.metadata["source"]
        text = doc.page_content
        
        try:
            # 1. FAST: Pattern-based entity extraction
            entities = self.extract_entities_from_text(text, chunk_id)
            
            if not entities:
                self.processed_chunks.add(chunk_id)
                self.save_progress()
                return 0
            
            # 2. Insert entities to Neo4j
            for entity in entities:
                try:
                    self.graph.query(
                        f"""
                        MERGE (n:{entity.type} {{id: $id}})
                        ON CREATE SET n.text = $text, n.chunk_id = $chunk_id
                        """,
                        {"id": entity.id, "text": entity.text, "chunk_id": chunk_id}
                    )
                except Exception as e:
                    logging.warning(f"Failed to insert entity {entity.id}: {e}")
                    continue
            
            # 3. SELECTIVE LLM: Extract relationships between these entities
            relationships = self.extract_relationships_with_llm(text, entities)
            
            # 4. Insert relationships
            for source_id, rel_type, target_id in relationships:
                try:
                    clean_rel = self.sanitize_relationship_type(rel_type)
                    self.graph.query(
                        f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:{clean_rel}]->(target)
                        ON CREATE SET r.original_type = $original_type
                        """,
                        {"source_id": source_id, "target_id": target_id, "original_type": rel_type}
                    )
                except Exception as e:
                    logging.warning(f"Failed to insert relationship {source_id}->{target_id}: {e}")
                    continue
            
            # 5. Link to source chunk
            try:
                self.graph.query(
                    """
                    MERGE (chunk:Chunk {chunk_id: $chunk_id, source: $source})
                    """,
                    {"chunk_id": chunk_id, "source": source}
                )
            except Exception as e:
                logging.warning(f"Failed to link chunk {chunk_id}: {e}")
            
            self.processed_chunks.add(chunk_id)
            self.save_progress()
            
            return len(entities)
        
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_id}: {e}\n{traceback.format_exc()}")
            return 0
    
    def process_all_chunks(self):
        """Process all chunks with hybrid extraction"""
        try:
            all_docs = self.load_all_chunks()
            
            if not all_docs:
                print("✔ All chunks already processed!")
                self.print_stats()
                return
            
            print(f"📂 Found {len(all_docs)} chunks to process")
            print(f"⚡ Using Hybrid: Fast patterns + Selective LLM\n")
            
            total_start = time.time()
            total_entities = 0
            errors = 0
            
            # Process chunks with threading for I/O efficiency
            with ThreadPoolExecutor(max_workers=1) as executor:  # Réduit à 1 pour éviter surcharge
                futures = []
                
                for i, doc in enumerate(all_docs):
                    futures.append((i, executor.submit(self.insert_chunk_to_neo4j, doc)))
                
                for idx, (orig_idx, future) in enumerate(futures):
                    try:
                        entity_count = future.result(timeout=300)  # 5 minutes timeout per chunk
                        total_entities += entity_count
                        
                        progress = orig_idx + 1
                        elapsed = time.time() - total_start
                        rate = progress / elapsed if elapsed > 0 else 0
                        eta = (len(all_docs) - progress) / rate if rate > 0 else 0
                        print(f"  ✓ {progress}/{len(all_docs)} ({rate:.1f}/sec | ETA: {eta/60:.1f}m | Entities: {total_entities})", flush=True)
                    
                    except TimeoutError:
                        logging.error(f"❌ Chunk {orig_idx}: Processing timeout")
                        errors += 1
                        print(f"  ❌ Chunk {orig_idx}: Timeout", flush=True)
                    except Exception as e:
                        logging.error(f"❌ Chunk {orig_idx}: {e}\n{traceback.format_exc()}")
                        errors += 1
                        print(f"  ❌ Chunk {orig_idx}: {e}", flush=True)
            
            total_time = time.time() - total_start
            
            self.print_stats()
            print(f"\n ✅ Hybrid Extraction Complete!")
            print(f"   • Processed: {len(all_docs)} chunks")
            print(f"   • Entities extracted: {total_entities}")
            print(f"   • Errors: {errors}")
            print(f"   • Total time: {total_time/60:.2f} minutes ({total_time:.0f}s)")
            print(f"   • Speed: {(len(all_docs)/total_time):.1f} chunks/sec")
            
            if total_time > 0:
                est_full = (len(self.processed_chunks) / len(all_docs)) * total_time if len(all_docs) > 0 else 0
                print(f"   • Estimated full dataset: {est_full/3600:.1f} hours")
        
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
            self.save_progress()
        except Exception as e:
            logging.error(f"Fatal error in process_all_chunks: {e}\n{traceback.format_exc()}")
            print(f"❌ Fatal error: {e}")
            self.save_progress()
    
    def print_stats(self):
        """Print graph statistics"""
        try:
            node_count = self.graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
            
            print(f"\n📈 Graph Statistics:")
            print(f"   • Total Nodes: {node_count:,}")
            print(f"   • Total Relationships: {rel_count:,}")
            
            node_types = self.graph.query("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """)
            
            if node_types:
                print(f"   • Node Types:")
                for row in node_types[:8]:
                    print(f"     - {row['type']}: {row['count']:,}")
        
        except Exception as e:
            print(f"  Stats error: {e}")
    
    def clear_graph(self):
        """Clear graph (caution!)"""
        response = input(" Delete all graph data? Type 'yes': ")
        if response.lower() == "yes":
            self.graph.query("MATCH (n) DETACH DELETE n")
            self.processed_chunks.clear()
            self.save_progress()
            print("✅ Graph cleared")


if __name__ == "__main__":
    try:
        extractor = HybridGraphExtractor()
        
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--clear":
            extractor.clear_graph()
        else:
            extractor.process_all_chunks()
    
    except KeyboardInterrupt:
        print("\n✋ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        print(f"\n❌ Fatal error: {e}")
        print("📋 Check the logs above for details")
        sys.exit(1)
