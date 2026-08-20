# Context Constellation RAG

A local Retrieval-Augmented Generation tool that turns a document corpus into a queryable evidence map.

It combines:

- Vector embeddings (`sentence-transformers`)
- Fast dense retrieval (`FAISS`)
- Lexical retrieval (`TF-IDF`)
- Hybrid ranking + MMR diversification
- Optional LLM synthesis with grounded citations

## Why it exists

Instead of only returning nearest chunks, this project groups retrieved evidence into themed **constellations** and answers with an explicit evidence trail. The output reads like an analyst memo, not a black box response.

## Capabilities

- `index`: Build a persistent embedding + lexical index from `.txt`/`.md` files
- `ask`: Query with hybrid retrieval and citation-grounded answer
- `map`: Inspect the discovered constellation clusters and dominant themes
- `evaluate`: Run a repeatable query suite, check paraphrase stability, and flag weak evidence patterns before demoing or iterating

Source paths are normalized across operating systems and included as searchable metadata. Hybrid dense/lexical relevance feeds the MMR selection step, rather than only narrowing its candidate pool. Queries asking for the current or latest authority also demote evidence explicitly marked archived, superseded, obsolete, deprecated, or outdated.

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

For a fast, deterministic offline index that never downloads a model:

```bash
python main.py index --corpus example_corpus --index-dir artifacts/index --embedding-model hashing
```

Ask a question:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --top-k 6
```

Restrict retrieval to only matching source paths when you want a narrower evidence trail:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --source-filter "risk|launch"
```

Write the answer and evidence trail to JSON for downstream tooling:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --json-out artifacts/answer.json
```

The exported answer JSON now includes `source_count` so you can tell whether the answer is grounded in one file or spread across multiple sources.
It now also includes `evidence_posture` so you can quickly tell whether the answer is broad, narrow, cross-supported, or concentrated in one source/constellation.
It now also includes an `agreement_signal` so you can tell whether the retrieved evidence is converging on one story or only loosely overlapping.
It now also records the optional `source_filter` used for the query scope.

Write a human-readable Markdown memo instead of only console output:

```bash
python main.py ask --index-dir artifacts/index --query "Where is rollout risk highest?" --report-out artifacts/answer-report.md
```

View constellation map:

```bash
python main.py map --index-dir artifacts/index
```

Export the constellation map as JSON:

```bash
python main.py map --index-dir artifacts/index --json-out artifacts/map.json
```

Export the constellation map as a Markdown scouting brief:

```bash
python main.py map --index-dir artifacts/index --report-out artifacts/map-report.md
```

Run a repeatable query-suite evaluation:

```bash
python main.py evaluate --index-dir artifacts/index --queries example_queries.json --llm off
```

Export the evaluation as JSON + Markdown for a portfolio-ready validation artifact:

```bash
python main.py evaluate --index-dir artifacts/index --queries example_queries.json --llm off --json-out artifacts/test-eval.json --report-out artifacts/test-eval.md
```

Each evaluation query can be a plain string or an object with `query`, optional `label`, optional `source_filter`, optional `expected_sources`, optional `forbidden_sources`, optional `conflict_source_groups`, and optional `variants`.
Use `expected_sources` to declare regex patterns over source paths that should show up in the evidence trail, and `variants` to provide alternate phrasings that stress-test retrieval stability.
Use `forbidden_sources` for known distractor patterns that must stay out of the first three evidence ranks. Use `conflict_source_groups` to require all sides of a documented disagreement to appear together; each group is an array of at least two source regexes.
The evaluation summary now highlights answer-mode mix, coverage posture, agreement mix, expected-source misses, and whether a question stays stable across paraphrases or turns brittle under rewording.
Expected source patterns are also scored with recall at K and mean reciprocal rank (MRR), including every declared paraphrase variant, so retrieval changes can be compared quantitatively.

Example evaluation entry:

```json
{
  "label": "launch-vs-memory",
  "query": "What tensions show up between launch posture and memory quality?",
  "expected_sources": ["rollout_posture", "memory_signals"],
  "variants": [
    {
      "label": "cross-source-tension-angle",
      "query": "Where do launch plans and memory-quality notes conflict with each other?"
    }
  ]
}
```

Evaluation outputs now include:

- expected-source recall and MRR across primary questions and variants
- the best evidence rank and matching source paths for every expected-source pattern
- variant stability averages for source, constellation, and chunk overlap
- per-query `variant-sensitive retrieval` flags when paraphrases pull the evidence trail apart
- per-query expected-source gaps when important document families disappear from the answer
- top-three distractor hit rate for explicitly forbidden source families
- conflict-source recall for questions that require opposing evidence

Use evaluation thresholds as a CI regression gate. The command writes JSON and Markdown outputs before returning a non-zero exit code when any configured threshold fails:

```bash
python main.py evaluate \
  --index-dir artifacts/index \
  --queries example_queries.json \
  --llm off \
  --min-expected-source-recall 1.0 \
  --min-expected-source-mrr 0.30 \
  --min-variant-stability-rate 0.75 \
  --max-flagged-query-rate 0.25 \
  --json-out artifacts/evaluation.json \
  --report-out artifacts/evaluation.md
```

### Operational benchmark

The three-document example corpus is useful for a quick demo, but it is too small to establish retrieval quality. `benchmark_corpus/` contains a nine-document checkout incident fixture with a current decision, corroborating operational evidence, a superseded launch draft, and unrelated documents that deliberately reuse launch and latency vocabulary.

Run the same deterministic gate used in CI:

```bash
python main.py index --corpus benchmark_corpus --index-dir artifacts/benchmark-index --embedding-model hashing
python main.py evaluate \
  --index-dir artifacts/benchmark-index \
  --queries benchmark_queries.json \
  --llm off \
  --top-k 6 \
  --min-expected-source-recall 0.88 \
  --min-expected-source-mrr 0.50 \
  --max-forbidden-source-hit-rate 0.25 \
  --min-conflict-source-recall 0.75 \
  --min-variant-stability-rate 0.40
```

This gate tests source recall, rank quality, near-match distractors, cross-source conflicts, and paraphrase stability without downloading an embedding model or calling an LLM.

All thresholds are optional values from `0` to `1`. The checked-in CI workflow builds a hashing index, runs the unit suite, and enforces the sample corpus gate without network-dependent embeddings.

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
It also surfaces a quick evidence-coverage badge so demo viewers can tell when an answer is narrow vs broadly supported.

## Repository layout

- `main.py`: end-to-end pipeline (ingest, embed, index, retrieve, answer)
- `example_queries.json`: starter evaluation suite with source expectations and paraphrase variants
- `benchmark_corpus/` and `benchmark_queries.json`: heterogeneous offline retrieval regression fixture
- `tests/test_evaluation.py`: deterministic coverage for rank metrics, gates, and offline embeddings
- `web_app.py`: tiny local browser UI for query + citation trace
- `example_corpus/`: sample documents for demo
- `artifacts/`: generated index output

## Notes

- Works fully offline for retrieval and extractive answers.
- `--embedding-model hashing` makes offline behavior explicit and reproducible instead of waiting for model loading to fail over.
- If the sentence-transformer model cannot be downloaded, the app automatically falls back to local hashing-based vector embeddings so the full RAG flow still runs.
- LLM synthesis is optional and never required to test the core RAG behavior.

## Portfolio Positioning

- Project type: Python RAG tool + optional local web UI
- Verification path: python main.py --help, python main.py evaluate --help, and python web_app.py --help

