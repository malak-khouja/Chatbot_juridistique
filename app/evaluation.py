import sys
import os
import time
import json
from typing import Dict, List, Tuple

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from app.rag.hybrid_rag import hybrid_rag_answer
from app.config import EMBEDDING_MODEL

# Questions de test avec réponses attendues
TEST_QUESTIONS = [
    {
        "question": "Quelle est la durée maximale de la période d'essai?",
        "expected_keywords": ["quinze jours", "Article 862", "essai"],
        "category": "Droit du travail"
    },
    {
        "question": "Comment rédiger un contrat de travail?",
        "expected_keywords": ["forme", "Article 68", "conditions"],
        "category": "Contrats"
    },
    {
        "question": "Quelles sont les conditions essentielles d'un contrat?",
        "expected_keywords": ["compensation", "dettes", "Article"],
        "category": "Contrats"
    },
    {
        "question": "Quels sont les droits et obligations de l'employeur?",
        "expected_keywords": ["maître", "serveur", "Article"],
        "category": "Droit du travail"
    },
    {
        "question": "Comment résoudre un contrat de travail?",
        "expected_keywords": ["résolutoire", "Article 863", "engagements"],
        "category": "Droit du travail"
    }
]

# Initialiser les embeddings pour la similarité
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def measure_latency(question: str) -> Tuple[str, float]:
    """Mesure le temps de réponse du chatbot"""
    start_time = time.time()
    response = hybrid_rag_answer(question)
    latency = time.time() - start_time
    return response, latency


def measure_relevance(question: str, response: str) -> float:
    """
    Mesure la pertinence entre la question et la réponse (0-1)
    Utilise la similarité cosinus entre les embeddings
    """
    try:
        question_embedding = embeddings_model.embed_query(question)
        response_embedding = embeddings_model.embed_query(response)
        
        # Calcul de la similarité cosinus
        dot_product = sum(a * b for a, b in zip(question_embedding, response_embedding))
        magnitude_q = sum(a ** 2 for a in question_embedding) ** 0.5
        magnitude_r = sum(b ** 2 for b in response_embedding) ** 0.5
        
        if magnitude_q == 0 or magnitude_r == 0:
            return 0.0
        
        similarity = dot_product / (magnitude_q * magnitude_r)
        return max(0, similarity)  # Assurer que c'est entre 0 et 1
    except Exception as e:
        print(f"Erreur lors du calcul de pertinence: {e}")
        return 0.0


def measure_precision(response: str, expected_keywords: List[str]) -> float:
    """
    Mesure la précision en vérifiants la présence des mots-clés attendus (0-1)
    """
    response_lower = response.lower()
    keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    
    if len(expected_keywords) == 0:
        return 0.0
    
    precision = keywords_found / len(expected_keywords)
    return precision





def evaluate_chatbot(verbose: bool = True) -> Dict:
    """
    Évalue la qualité globale du chatbot
    Retourne un rapport avec toutes les métriques
    """
    results = {
        "total_tests": len(TEST_QUESTIONS),
        "tests": [],
        "averages": {
            "latency": 0.0,
            "relevance": 0.0,
            "precision": 0.0
        }
    }
    
    print("=" * 80)
    print("ÉVALUATION DU CHATBOT JURIDIQUE")
    print("=" * 80)
    
    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        category = test["category"]
        
        print(f"\n[Test {i}/{len(TEST_QUESTIONS)}] {category}")
        print(f"Question: {question}")
        
        # Mesurer les métriques
        response, latency = measure_latency(question)
        relevance = measure_relevance(question, response)
        precision = measure_precision(response, expected_keywords)
        
        test_result = {
            "question": question,
            "category": category,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "metrics": {
                "latency_seconds": round(latency, 2),
                "relevance_score": round(relevance, 3),  # 0-1
                "precision_score": round(precision, 3)   # 0-1
            }
        }
        
        results["tests"].append(test_result)
        
        # Afficher les résultats
        if verbose:
            print(f"  ⏱️  Latence: {latency:.2f}s")
            print(f"  📊 Pertinence: {relevance:.1%}")
            print(f"  ✅ Précision: {precision:.1%}")
    
    # Calculer les moyennes
    if results["tests"]:
        results["averages"]["latency"] = round(
            sum(t["metrics"]["latency_seconds"] for t in results["tests"]) / len(results["tests"]), 2
        )
        results["averages"]["relevance"] = round(
            sum(t["metrics"]["relevance_score"] for t in results["tests"]) / len(results["tests"]), 3
        )
        results["averages"]["precision"] = round(
            sum(t["metrics"]["precision_score"] for t in results["tests"]) / len(results["tests"]), 3
        )
    
    # Afficher le résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES MÉTRIQUES")
    print("=" * 80)
    print(f"⏱️  Latence moyenne: {results['averages']['latency']}s")
    print(f"📊 Pertinence moyenne: {results['averages']['relevance']:.1%}")
    print(f"✅ Précision moyenne: {results['averages']['precision']:.1%}")
    print("=" * 80)
    
    return results


def save_evaluation_report(results: Dict, filepath: str = "evaluation_report.json"):
    """Sauvegarde le rapport d'évaluation en JSON"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport sauvegardé dans: {filepath}")


if __name__ == "__main__":
    # Lancer l'évaluation
    results = evaluate_chatbot(verbose=True)
    
    # Sauvegarder le rapport
    save_evaluation_report(results)
