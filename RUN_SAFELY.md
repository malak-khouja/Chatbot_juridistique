# 🛡️ Guide pour exécuter l'extraction sans planter

## ✅ Améliorations apportées

Le fichier `hybrid_graph_30min.py` a été corrigé avec :
- ✔️ Gestion robuste des erreurs (try/except partout)
- ✔️ Timeouts pour chaque chunk (60s max)
- ✔️ Logging détaillé pour déboguer
- ✔️ Sauvegarde progressive du progrès
- ✔️ Gestion des interruptions (Ctrl+C)

## 🚀 Comment exécuter

### 1. **En environnement sécurisé** (recommandé)
```bash
python -u app/rag/hybrid_graph_30min.py 2>&1 | tee extraction.log
```
- `-u` : Désactive la mise en buffer (affichage en temps réel)
- `2>&1` : Capture les erreurs aussi
- `tee` : Sauvegarde dans un fichier log

### 2. **Avec limite de ressources** (si PC ralentit)
```bash
# Sur Windows - Limiter les workers à 1 (au lieu de 2)
# Ouvrir hybrid_graph_30min.py et remplacer :
# with ThreadPoolExecutor(max_workers=2) as executor:
# par :
# with ThreadPoolExecutor(max_workers=1) as executor:
```

### 3. **Vérifier la connexion avant de lancer**
```bash
# Tester Neo4j
python -c "from app.config import *; from langchain_community.graphs import Neo4jGraph; g = Neo4jGraph(url=CFG_NEO4J_URI, username=CFG_NEO4J_USER, password=CFG_NEO4J_PASSWORD); print('✅ Neo4j OK')"

# Tester Ollama
python -c "from app.config import *; from langchain_community.llms import Ollama; o = Ollama(model=LLM_MODEL); print('✅ Ollama OK')"
```

## 🛑 Si le PC s'éteint toujours

### **Causes possibles** :
1. **Overheating** - Vérifier la temp du CPU (Ctrl+Alt+Del → Task Manager → Performance)
2. **Manque RAM** - Réduire `max_workers=1` dans le code
3. **Neo4j cloud timeout** - Augmenter le timeout à 300s
4. **Ollama pas réactif** - Redémarrer le service Ollama

### **Solutions rapides** :

**A) Exécuter par chunks manuellement :**
```bash
# Traiter juste 10 chunks pour tester
python app/rag/hybrid_graph_30min.py
# Laisser tourner, le progrès se sauvegarde
# Relancer quand vous voulez continuer
```

**B) Augmenter les timeouts** (éditer le fichier) :
```python
# Ligne ~350 : change 60 en 300
future.result(timeout=300)  # 5 minutes au lieu de 1
```

**C) Utiliser le mode 1 worker** (éditer le fichier) :
```python
# Ligne ~330 : change 2 en 1
with ThreadPoolExecutor(max_workers=1) as executor:
```

## 📊 Fichiers générés

- `hybrid_progress.json` - Sauvegarde automatique de l'avancement
- `extraction.log` - Logs détaillés de l'exécution

## ✋ Pour arrêter proprement

Appuyez sur **Ctrl+C** - Le progrès est sauvegardé et vous pouvez relancer plus tard.

---

💡 **Les corrections de code ont déjà été appliquées au fichier !**
