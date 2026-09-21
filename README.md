# Context Constellation RAG

A local Retrieval-Augmented Generation tool that turns a document corpus into a queryable evidence map.

It combines:

- Vector embeddings (`sentence-transformers`)
- Fast dense retrieval (`FAISS`)
- Lexical retrieval (`TF-IDF`)
- Source-scope-aware hybrid ranking + query-anchor evidence chaining + source-diverse MMR
- Optional LLM synthesis with grounded citations

## Why it exists

Instead of only returning nearest chunks, this project groups retrieved evidence into themed **constellations** and answers with an explicit evidence trail. The output reads like an analyst memo, not a black box response.

## Capabilities

- `index`: Build a persistent embedding + lexical index from `.txt`/`.md` files
- `ask`: Query with hybrid retrieval and citation-grounded answer
- `map`: Inspect the discovered constellation clusters and dominant themes
- `evaluate`: Run a repeatable query suite, check paraphrase stability, and flag weak evidence patterns before demoing or iterating

`ask` now has an absolute grounding gate in addition to relative ranking. If no selected source directly covers enough of the requested subject, it returns `Insufficient evidence` instead of turning the closest topical match into an answer. JSON and Markdown outputs retain the closest-source trace plus the shared-term and query-coverage decision so abstention is auditable.

Explicitly stale, out-of-scope, or non-evidence rows are no longer allowed to satisfy that grounding gate even when they remain visible in the retrieval trace. Requests for concrete amounts, deadlines, shipments, or ticket identifiers must also cover each requested detail family in affirmative evidence; nearby documents cannot be stitched together into a false answer from unrelated matching words.

Source paths are normalized across operating systems and included as searchable metadata. Dense relevance, lexical relevance, and source/title alignment feed the MMR selection step, which favors distinct sources before returning multiple chunks from one document. A narrow synonym bridge makes operational gate questions stable under wording such as `restart`/`resume`, `proof`/`evidence`, and `sign off`/`approval` without rewriting the dense semantic query. Queries asking for a current decision strongly demote archived or superseded evidence, while explicitly historical queries keep it eligible. Documents that disclaim the query's subject or explicitly say they did not validate the requested evidence are demoted only when the disclaimer overlaps the question's subject.

The two strongest source-distinct query matches also act as bounded support anchors. A normalized lexical support score helps linked recovery, reliability, and approval records stay in the evidence budget when a short paraphrase names the decision but omits one underlying failure term. Answerability does not inherit the anchor score: the absolute grounding gate still requires direct subject coverage before an answer is allowed.

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

Each evaluation query can be a plain string or an object with `query`, optional `label`, optional `source_filter`, optional `expected_outcome`, optional `expected_sources`, optional `forbidden_sources`, optional `conflict_source_groups`, and optional `variants`.
Set `expected_outcome` to `answer` or `abstain` to test whether the grounding gate answers supported questions and refuses unsupported ones. Variants inherit the primary query's expected outcome.
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
  --min-answerability-accuracy 1.0 \
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
  --top-k 4 \
  --min-expected-source-recall 1.0 \
  --min-expected-source-mrr 0.55 \
  --max-forbidden-source-hit-rate 0.0 \
  --min-conflict-source-recall 1.0 \
  --min-variant-stability-rate 1.0 \
  --min-answerability-accuracy 1.0
```

This gate tests source recall, rank quality, near-match distractors, cross-source conflicts, and paraphrase stability without downloading an embedding model or calling an LLM. The frozen hashing benchmark now requires full expected-source and conflict recall within four evidence slots, no forbidden source in the first three ranks, and stable retrieval for every checked paraphrase. The smaller evidence budget prevents the fixture from passing by returning most of its corpus.

### Independent grounding benchmark

`grounding_corpus/` is a separately authored, fictional database-recovery fixture. It is original repository content rather than redistributed third-party text. Its suite mixes supported cutover questions with adjacent but unanswerable requests for customer compensation and hardware-delivery details.

Run the second CI promotion gate:

```bash
python main.py index --corpus grounding_corpus --index-dir artifacts/grounding-index --embedding-model hashing
python main.py evaluate \
  --index-dir artifacts/grounding-index \
  --queries grounding_queries.json \
  --llm off \
  --top-k 4 \
  --min-expected-source-recall 1.0 \
  --min-expected-source-mrr 0.57 \
  --max-forbidden-source-hit-rate 0.0 \
  --min-variant-stability-rate 1.0 \
  --min-answerability-accuracy 1.0 \
  --min-abstention-recall 1.0
```

The workflow requires retrieval changes to preserve the checkout evidence contract and this independent answer/abstain contract. A relative top result is no longer sufficient proof that the corpus can answer a question.

At the four-source budget, the independent fixture now freezes 1.0 expected-source recall, 0.5781 MRR, zero top-three distractor hits, 1.0 paraphrase stability, and perfect answerability/abstention classification. These values are regression baselines, not claims of general RAG quality. The support-anchor sources and per-chunk anchor score are exported beside the existing dense, lexical, scope, and penalty diagnostics so the extra retrieval signal remains auditable.

### Adversarial domain-shift benchmark

`adversarial_corpus/` adds a 14-document warehouse-fulfillment incident outside the checkout and database-recovery domains. It mixes a current cutover decision with corroborating quality, operator, and inventory records; a superseded plan; and near-match payroll, search, shipment, hardware, badge, mobile, and service-credit documents. The unsupported questions deliberately distribute tempting amount, deadline, carrier, ticket, and tracking terms across unrelated sources.

Run the fourth CI gate:

```bash
python main.py index --corpus adversarial_corpus --index-dir artifacts/adversarial-index --embedding-model hashing
python main.py evaluate \
  --index-dir artifacts/adversarial-index \
  --queries adversarial_queries.json \
  --llm off \
  --top-k 4 \
  --min-expected-source-recall 1.0 \
  --min-expected-source-mrr 0.57 \
  --max-forbidden-source-hit-rate 0.0 \
  --min-conflict-source-recall 1.0 \
  --min-variant-stability-rate 1.0 \
  --min-answerability-accuracy 1.0 \
  --min-abstention-recall 1.0
```

The frozen top-four contract requires full expected-source and conflict recall, no forbidden top-three source, stable paraphrases, and perfect answer/abstain classification. It is intentionally small enough to stay deterministic in CI but large enough that returning most of the corpus cannot satisfy the gate.

Answer JSON and Markdown reports include the expanded lexical retrieval query plus source-scope, stale-source, scope-mismatch, and non-evidence penalties for each selected chunk. Ranking behavior is therefore inspectable rather than hidden behind one aggregate score.

All thresholds are optional values from `0` to `1`. The checked-in CI workflow builds deterministic hashing indexes, runs the unit suite, and enforces all four corpus gates without network-dependent embeddings.

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
- `grounding_corpus/` and `grounding_queries.json`: independent answerability and abstention fixture
- `adversarial_corpus/` and `adversarial_queries.json`: cross-domain distractor and unsupported-detail fixture
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

