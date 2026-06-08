# Context Constellation RAG

A creative, portfolio-grade Retrieval-Augmented Generation project that turns a corpus into a **constellation map** of ideas.

It combines:

- Vector embeddings (`sentence-transformers`)
- Fast dense retrieval (`FAISS`)
- Lexical retrieval (`TF-IDF`)
- Hybrid ranking + MMR diversification
- Optional LLM synthesis with grounded citations

## Why this is unique

Instead of only returning nearest chunks, this project groups retrieved evidence into themed **constellations** and answers with an explicit evidence trail. The output reads like an analyst memo, not a black box response.

## Capabilities

- `index`: Build a persistent embedding + lexical index from `.txt`/`.md` files
- `ask`: Query with hybrid retrieval and citation-grounded answer
- `map`: Inspect the discovered constellation clusters and dominant themes

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Build index:

```bash
python main.py index --corpus example_corpus --index-dir artifacts/index
```

Ask a question:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --top-k 6
```

Write the answer and evidence trail to JSON for downstream tooling:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --json-out artifacts/answer.json
```

View constellation map:

```bash
python main.py map --index-dir artifacts/index
```

Export the constellation map as JSON:

```bash
python main.py map --index-dir artifacts/index --json-out artifacts/map.json
```

## Optional LLM mode

If you set `OPENAI_API_KEY`, `ask` can synthesize a more natural answer:

```bash
set OPENAI_API_KEY=your_key_here
python main.py ask --index-dir artifacts/index --query "What should we prioritize next?" --llm auto --model gpt-4.1-mini
```

If no key is present, it automatically falls back to an extractive grounded response.

## Tiny local web UI

Build the index first, then run:

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:7860
```

This gives a lightweight demo surface for live query + evidence trace walkthroughs.

The local UI now also supports switching retrieval depth between Top 4, Top 6, and Top 8 evidence chunks.

## Repository layout

- `main.py`: end-to-end pipeline (ingest, embed, index, retrieve, answer)
- `web_app.py`: tiny local browser UI for query + citation trace
- `example_corpus/`: sample documents for demo
- `artifacts/`: generated index output

## Notes

- Works fully offline for retrieval and extractive answers.
- If the sentence-transformer model cannot be downloaded, the app automatically falls back to local hashing-based vector embeddings so the full RAG flow still runs.
- LLM synthesis is optional and never required to test the core RAG behavior.

## Portfolio Positioning

- Project type: Python RAG tool + optional local web UI
- Verification path: python main.py --help and python web_app.py --help

