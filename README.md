# MINIMAL_RAG
Minimal RAG that includes : PDF loader → chunking → embeddings → Chroma → retriever → LLM. It will use my personal learning documents to train and test.

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="ta_clé_api"
```

## Utilisation

1. Mets tes PDFs dans le dossier `corpus/` (le papier Lewis et al. 2020, tes notes de cours, etc.)
2. Construis la base vectorielle :
   ```bash
   python rag_minimal.py index
   ```
3. Pose une question :
   ```bash
   python rag_minimal.py ask "Quelle est la différence entre RAG-Sequence et RAG-Token ?"
   ```
4. Ou lance le mode chat interactif :
   ```bash
   python rag_minimal.py chat
   ```

