from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

import faiss
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

console = Console()


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    start: int
    end: int


def read_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            docs.append((path.relative_to(corpus_dir).as_posix(), raw.replace("\ufeff", "")))
    return docs


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(source: str, text: str, target_chars: int = 520, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    cursor = 0

    for sentence in sentences:
        if buffer and len(" ".join(buffer + [sentence])) > target_chars:
            chunk_text_value = " ".join(buffer).strip()
            end = cursor + len(chunk_text_value)
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::c{len(chunks)+1}",
                    source=source,
                    text=chunk_text_value,
                    start=cursor,
                    end=end,
                )
            )
            overlap = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
            buffer = overlap + [sentence]
            cursor = max(0, end - len(" ".join(overlap)))
        else:
            buffer.append(sentence)

    if buffer:
        chunk_text_value = " ".join(buffer).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{source}::c{len(chunks)+1}",
                source=source,
                text=chunk_text_value,
                start=cursor,
                end=cursor + len(chunk_text_value),
            )
        )

    return chunks


def build_retrieval_text(chunk: Chunk) -> str:
    source_without_suffix = re.sub(r"\.[^./\\]+$", "", chunk.source)
    source_terms = re.sub(r"[/\\_.-]+", " ", source_without_suffix).strip()
    return f"Source: {source_terms}. {chunk.text}"


class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.mode = "sentence-transformers"
        self.model = None
        self.fallback_vectorizer: HashingVectorizer | None = None
        self.dim = 1024

        if model_name in {"hashing", "hashing-fallback"}:
            self.mode = "hashing"
            self.fallback_vectorizer = HashingVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                n_features=self.dim,
                alternate_sign=False,
                norm=None,
            )
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            sample = self.model.encode(["shape probe"], normalize_embeddings=True, show_progress_bar=False)
            self.dim = int(np.asarray(sample).shape[1])
        except Exception as exc:  # pragma: no cover
            console.print(
                f"[yellow]Embedding fallback enabled[/yellow]: {exc}. "
                "Using local hashing embeddings (no network/model download required)."
            )
            self.mode = "hashing-fallback"
            self.fallback_vectorizer = HashingVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                n_features=self.dim,
                alternate_sign=False,
                norm=None,
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "sentence-transformers" and self.model is not None:
            matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(matrix, dtype=np.float32)

        assert self.fallback_vectorizer is not None
        matrix = self.fallback_vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def compute_cluster_keywords(texts: list[str], labels: np.ndarray, max_terms: int = 4) -> dict[int, list[str]]:
    keywords: dict[int, list[str]] = {}
    unique_labels = sorted({int(x) for x in labels.tolist()})
    for label in unique_labels:
        cluster_docs = [text for text, doc_label in zip(texts, labels.tolist()) if int(doc_label) == label]
        if not cluster_docs:
            keywords[label] = ["mixed"]
            continue
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
        matrix = vectorizer.fit_transform(cluster_docs)
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        order = np.argsort(mean_scores)[::-1]
        picked = [terms[idx] for idx in order[:max_terms] if mean_scores[idx] > 0]
        keywords[label] = picked or ["mixed"]
    return keywords


def save_index(index_dir: Path, *, chunks: list[Chunk], embeddings: np.ndarray, faiss_index: faiss.IndexFlatIP, vectorizer: TfidfVectorizer, tfidf_matrix: Any, cluster_labels: np.ndarray, cluster_keywords: dict[int, list[str]], model_name: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2), encoding="utf-8"
    )
    np.save(index_dir / "embeddings.npy", embeddings)
    faiss.write_index(faiss_index, str(index_dir / "dense.faiss"))

    with (index_dir / "lexical.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, handle)

    (index_dir / "clusters.json").write_text(
        json.dumps(
            {
                "labels": [int(x) for x in cluster_labels.tolist()],
                "keywords": {str(k): v for k, v in cluster_keywords.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "chunk_count": len(chunks),
                "dim": int(embeddings.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_index(index_dir: Path) -> dict[str, Any]:
    chunks_raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**row) for row in chunks_raw]
    embeddings = np.load(index_dir / "embeddings.npy")
    dense = faiss.read_index(str(index_dir / "dense.faiss"))

    with (index_dir / "lexical.pkl").open("rb") as handle:
        lexical = pickle.load(handle)

    clusters = json.loads((index_dir / "clusters.json").read_text(encoding="utf-8"))
    labels = np.asarray(clusters["labels"], dtype=np.int32)
    keywords = {int(k): v for k, v in clusters["keywords"].items()}

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "dense": dense,
        "vectorizer": lexical["vectorizer"],
        "lexical_matrix": lexical["matrix"],
        "cluster_labels": labels,
        "cluster_keywords": keywords,
        "meta": meta,
    }


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_stale_source_penalties(query: str, chunks: list[Chunk]) -> dict[int, float]:
    current_intent = re.search(
        r"\b(current|latest|now|authoritative|authority|go-live)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not current_intent:
        return {}

    stale_pattern = re.compile(
        r"\b(archive|archived|superseded|obsolete|deprecated|outdated)\b",
        flags=re.IGNORECASE,
    )
    return {
        index: 0.45
        for index, chunk in enumerate(chunks)
        if stale_pattern.search(f"{chunk.source} {chunk.text[:400]}")
    }


def mmr_select(
    candidate_indices: list[int],
    query_vec: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.7,
    relevance_scores: dict[int, float] | None = None,
) -> list[int]:
    chosen: list[int] = []
    remaining = candidate_indices.copy()

    while remaining and len(chosen) < top_k:
        best_idx = remaining[0]
        best_score = -1e9
        for idx in remaining:
            rel = (
                relevance_scores[idx]
                if relevance_scores is not None
                else float(np.dot(query_vec, doc_embeddings[idx]))
            )
            div = 0.0
            if chosen:
                div = max(float(np.dot(doc_embeddings[idx], doc_embeddings[c])) for c in chosen)
            score = lambda_mult * rel - (1.0 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return chosen


def build_extractive_answer(query: str, selected_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"Question: {query}",
        "",
        "Grounded take:",
    ]
    for row in selected_rows[:3]:
        text = row["chunk"].text
        trimmed = text[:230].strip()
        if len(text) > 230:
            trimmed += "..."
        lines.append(f"- [{row['citation']}] {trimmed}")

    lines.append("")
    lines.append("Suggested next action: validate the strongest claim against at least one chunk from a different constellation before finalizing decisions.")
    return "\n".join(lines)


def build_llm_answer(query: str, selected_rows: list[dict[str, Any]], model: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package unavailable") from exc

    context_blocks = []
    for row in selected_rows:
        context_blocks.append(
            f"{row['citation']} | source={row['chunk'].source} | constellation={row['constellation']}\n{row['chunk'].text}"
        )

    prompt = (
        "Use only the provided evidence.\n"
        "Return a concise answer with 2-4 bullets and cite chunk ids like [C1], [C2].\n"
        "If evidence is conflicting, say that explicitly.\n\n"
        f"Question: {query}\n\n"
        "Evidence:\n"
        + "\n\n".join(context_blocks)
    )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": "You are a careful RAG analyst."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        temperature=0.2,
    )

    answer = getattr(response, "output_text", "").strip()
    if not answer:
        raise RuntimeError("Empty model response")
    return answer


def build_evidence_posture(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    constellation_counts: dict[str, int] = {}
    for row in selected_rows:
        source = row["chunk"].source
        constellation = row["constellation"]
        source_counts[source] = source_counts.get(source, 0) + 1
        constellation_counts[constellation] = constellation_counts.get(constellation, 0) + 1

    total_rows = max(1, len(selected_rows))
    dominant_source_share = max(source_counts.values(), default=0) / total_rows
    dominant_constellation_share = max(constellation_counts.values(), default=0) / total_rows

    if len(source_counts) >= 3 and len(constellation_counts) >= 3:
        coverage_label = "broad"
    elif len(source_counts) >= 2 and len(constellation_counts) >= 2:
        coverage_label = "moderate"
    else:
        coverage_label = "narrow"

    if dominant_source_share >= 0.67 or dominant_constellation_share >= 0.67:
        tension_label = "concentrated"
    elif dominant_source_share <= 0.5 and dominant_constellation_share <= 0.5:
        tension_label = "cross-supported"
    else:
        tension_label = "mixed"

    return {
        "coverage_label": coverage_label,
        "tension_label": tension_label,
        "source_count": len(source_counts),
        "constellation_count": len(constellation_counts),
        "dominant_source_share": round(float(dominant_source_share), 4),
        "dominant_constellation_share": round(float(dominant_constellation_share), 4),
    }


def build_agreement_signal(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts: dict[str, int] = {}
    for row in selected_rows:
        text = row["chunk"].text.lower()
        terms = {
            term
            for term in re.findall(r"[a-z][a-z0-9_-]{3,}", text)
            if term not in {"that", "with", "from", "this", "have", "were", "their", "into", "they", "will"}
        }
        for term in terms:
            token_counts[term] = token_counts.get(term, 0) + 1

    repeated_terms = sorted(
        (
            {"term": term, "count": count}
            for term, count in token_counts.items()
            if count >= 2
        ),
        key=lambda item: (-item["count"], item["term"]),
    )
    shared_terms = [item["term"] for item in repeated_terms[:6]]
    repeated_count = len(repeated_terms)

    if repeated_count >= 6:
        label = "aligned"
        note = "Top evidence chunks repeat a clear core vocabulary, so the answer is converging on one main story."
    elif repeated_count >= 3:
        label = "partial"
        note = "Evidence overlaps on some important terms, but the support still spans multiple adjacent angles."
    else:
        label = "fragmented"
        note = "Evidence is dispersed across distinct snippets, so the answer should be treated as exploratory rather than settled."

    return {
        "label": label,
        "shared_terms": shared_terms,
        "repeated_term_count": repeated_count,
        "note": note,
    }


def chunk_id_set(result: dict[str, Any]) -> set[str]:
    return {row["chunk"].chunk_id for row in result["evidence"]}


def source_set(result: dict[str, Any]) -> set[str]:
    return set(result["source_breakdown"].keys())


def constellation_set(result: dict[str, Any]) -> set[str]:
    return set(result["constellation_breakdown"].keys())


def compute_set_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 4)


def build_variant_stability_summary(
    primary_result: dict[str, Any],
    variant_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not variant_results:
        return {
            "variant_count": 0,
            "stable": None,
            "avg_source_overlap": None,
            "avg_constellation_overlap": None,
            "avg_chunk_overlap": None,
            "posture_mismatch_count": 0,
            "agreement_mismatch_count": 0,
            "details": [],
        }

    primary_sources = source_set(primary_result)
    primary_constellations = constellation_set(primary_result)
    primary_chunks = chunk_id_set(primary_result)
    primary_posture = primary_result["evidence_posture"]
    primary_agreement = primary_result["agreement_signal"]["label"]
    details = []

    for variant in variant_results:
        variant_sources = source_set(variant["result"])
        variant_constellations = constellation_set(variant["result"])
        variant_chunks = chunk_id_set(variant["result"])
        details.append(
            {
                "label": variant["label"],
                "query": variant["query"],
                "source_filter": variant["source_filter"],
                "source_overlap": compute_set_overlap(primary_sources, variant_sources),
                "constellation_overlap": compute_set_overlap(primary_constellations, variant_constellations),
                "chunk_overlap": compute_set_overlap(primary_chunks, variant_chunks),
                "posture_match": (
                    primary_posture["coverage_label"] == variant["result"]["evidence_posture"]["coverage_label"]
                    and primary_posture["tension_label"] == variant["result"]["evidence_posture"]["tension_label"]
                ),
                "agreement_match": primary_agreement == variant["result"]["agreement_signal"]["label"],
                "source_count": variant["result"]["source_count"],
                "expected_source_recall": variant.get("expected_source_metrics", {}).get("recall_at_k"),
                "missing_expected_sources": [
                    item["pattern"]
                    for item in variant.get("expected_source_metrics", {}).get("details", [])
                    if item["best_rank"] is None
                ],
            }
        )

    avg_source_overlap = round(mean(item["source_overlap"] for item in details), 4)
    avg_constellation_overlap = round(mean(item["constellation_overlap"] for item in details), 4)
    avg_chunk_overlap = round(mean(item["chunk_overlap"] for item in details), 4)
    posture_mismatch_count = sum(1 for item in details if not item["posture_match"])
    agreement_mismatch_count = sum(1 for item in details if not item["agreement_match"])
    stable = (
        avg_source_overlap >= 0.5
        and avg_constellation_overlap >= 0.5
        and posture_mismatch_count == 0
        and agreement_mismatch_count <= 1
    )

    return {
        "variant_count": len(details),
        "stable": stable,
        "avg_source_overlap": avg_source_overlap,
        "avg_constellation_overlap": avg_constellation_overlap,
        "avg_chunk_overlap": avg_chunk_overlap,
        "posture_mismatch_count": posture_mismatch_count,
        "agreement_mismatch_count": agreement_mismatch_count,
        "details": details,
    }


def find_expected_source_matches(
    *,
    expected_sources: list[str],
    observed_sources: set[str],
) -> dict[str, Any]:
    matched: list[str] = []
    missing: list[str] = []
    for pattern_text in expected_sources:
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            raise SystemExit(f"Invalid expected source regex {pattern_text!r}: {exc}") from exc
        pattern_matches = sorted(source for source in observed_sources if pattern.search(source))
        if pattern_matches:
            matched.append(pattern_text)
        else:
            missing.append(pattern_text)
    return {"matched": matched, "missing": missing}


def build_expected_source_metrics(
    *,
    expected_sources: list[str],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    details = []
    for pattern_text in expected_sources:
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            raise SystemExit(f"Invalid expected source regex {pattern_text!r}: {exc}") from exc

        matching_rows = [row for row in evidence if pattern.search(row["chunk"].source)]
        best_rank = min((int(row["rank"]) for row in matching_rows), default=None)
        details.append(
            {
                "pattern": pattern_text,
                "matched_sources": sorted({row["chunk"].source for row in matching_rows}),
                "best_rank": best_rank,
                "reciprocal_rank": round(1.0 / best_rank, 4) if best_rank else 0.0,
            }
        )

    expected_count = len(details)
    matched_count = sum(1 for item in details if item["best_rank"] is not None)
    return {
        "expected_count": expected_count,
        "matched_count": matched_count,
        "recall_at_k": round(matched_count / expected_count, 4) if expected_count else None,
        "mean_reciprocal_rank": (
            round(mean(item["reciprocal_rank"] for item in details), 4) if details else None
        ),
        "details": details,
    }


def build_forbidden_source_metrics(
    *,
    forbidden_sources: list[str],
    evidence: list[dict[str, Any]],
    rank_cutoff: int = 3,
) -> dict[str, Any]:
    details = []
    for pattern_text in forbidden_sources:
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            raise SystemExit(f"Invalid forbidden source regex {pattern_text!r}: {exc}") from exc

        matching_rows = [row for row in evidence if pattern.search(row["chunk"].source)]
        best_rank = min((int(row["rank"]) for row in matching_rows), default=None)
        details.append(
            {
                "pattern": pattern_text,
                "matched_sources": sorted({row["chunk"].source for row in matching_rows}),
                "best_rank": best_rank,
                "violates_cutoff": best_rank is not None and best_rank <= rank_cutoff,
            }
        )

    hit_count = sum(1 for item in details if item["violates_cutoff"])
    return {
        "forbidden_count": len(details),
        "hit_count": hit_count,
        "hit_rate": round(hit_count / len(details), 4) if details else None,
        "rank_cutoff": rank_cutoff,
        "details": details,
    }


def build_conflict_source_metrics(
    *,
    conflict_source_groups: list[list[str]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    details = []
    for group_index, source_patterns in enumerate(conflict_source_groups, start=1):
        if len(source_patterns) < 2:
            raise SystemExit(
                f"Conflict source group {group_index} must declare at least two source patterns."
            )
        source_metrics = build_expected_source_metrics(
            expected_sources=source_patterns,
            evidence=evidence,
        )
        distinct_sources = sorted(
            {
                source
                for item in source_metrics["details"]
                for source in item["matched_sources"]
            }
        )
        covered = (
            source_metrics["matched_count"] == source_metrics["expected_count"]
            and len(distinct_sources) >= len(source_patterns)
        )
        details.append(
            {
                "group_index": group_index,
                "source_patterns": source_patterns,
                "covered": covered,
                "distinct_sources": distinct_sources,
                "source_metrics": source_metrics,
            }
        )

    covered_count = sum(1 for item in details if item["covered"])
    return {
        "group_count": len(details),
        "covered_count": covered_count,
        "recall": round(covered_count / len(details), 4) if details else None,
        "details": details,
    }


def build_quality_gate(
    *,
    summary: dict[str, Any],
    min_expected_source_recall: float | None = None,
    min_expected_source_mrr: float | None = None,
    min_variant_stability_rate: float | None = None,
    max_flagged_query_rate: float | None = None,
    max_forbidden_source_hit_rate: float | None = None,
    min_conflict_source_recall: float | None = None,
) -> dict[str, Any]:
    configured = [
        ("expected source recall", "expected_source_recall", min_expected_source_recall, ">="),
        ("expected source MRR", "expected_source_mrr", min_expected_source_mrr, ">="),
        ("variant stability rate", "variant_stability_rate", min_variant_stability_rate, ">="),
        ("flagged query rate", "flagged_query_rate", max_flagged_query_rate, "<="),
        (
            "forbidden source hit rate in top 3",
            "forbidden_source_hit_rate",
            max_forbidden_source_hit_rate,
            "<=",
        ),
        (
            "conflict source recall",
            "conflict_source_recall",
            min_conflict_source_recall,
            ">=",
        ),
    ]
    checks = []
    for label, metric, threshold, comparator in configured:
        if threshold is None:
            continue
        actual = summary[metric]
        passed = actual >= threshold if comparator == ">=" else actual <= threshold
        checks.append(
            {
                "label": label,
                "metric": metric,
                "actual": actual,
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )

    return {
        "configured": bool(checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def run_query_payload(
    *,
    payload: dict[str, Any],
    query: str,
    top_k: int = 6,
    mmr_lambda: float = 0.7,
    llm_mode: str = "auto",
    model: str = "gpt-4.1-mini",
    source_filter: str | None = None,
    embedder: EmbeddingEngine | None = None,
) -> dict[str, Any]:
    chunks: list[Chunk] = payload["chunks"]
    embeddings: np.ndarray = payload["embeddings"]
    dense_index = payload["dense"]
    vectorizer: TfidfVectorizer = payload["vectorizer"]
    lexical_matrix = payload["lexical_matrix"]
    labels: np.ndarray = payload["cluster_labels"]
    keywords: dict[int, list[str]] = payload["cluster_keywords"]
    model_name: str = payload["meta"]["embedding_model"]

    if embedder is None:
        embedder = EmbeddingEngine(model_name=model_name)
    query_vec = embedder.encode([query])[0]

    filtered_indices = list(range(len(chunks)))
    if source_filter:
        pattern = re.compile(source_filter)
        filtered_indices = [index for index, chunk in enumerate(chunks) if pattern.search(chunk.source)]
        if not filtered_indices:
            raise SystemExit(f"No indexed sources matched source filter: {source_filter}")

    dense_k = min(max(top_k * 4, top_k), len(filtered_indices))
    if source_filter:
        filtered_embeddings = embeddings[filtered_indices]
        filtered_scores = np.dot(filtered_embeddings, query_vec)
        order = np.argsort(filtered_scores)[::-1][:dense_k]
        dense_indices = np.asarray([filtered_indices[int(position)] for position in order], dtype=np.int32)
        dense_scores = np.asarray([filtered_scores[int(position)] for position in order], dtype=np.float32)
    else:
        scores, idxs = dense_index.search(query_vec.reshape(1, -1), dense_k)
        dense_scores = scores[0]
        dense_indices = idxs[0]

    query_lex = vectorizer.transform([query])
    lex_scores_all = (lexical_matrix @ query_lex.T).toarray().ravel()

    dense_norm = normalize_scores(dense_scores)
    lex_subset = np.asarray([lex_scores_all[i] for i in dense_indices], dtype=np.float32)
    lex_norm = normalize_scores(lex_subset)

    hybrid = 0.7 * dense_norm + 0.3 * lex_norm
    stale_penalties = build_stale_source_penalties(query, chunks)
    adjusted_hybrid = np.asarray(
        [
            float(hybrid[position]) - stale_penalties.get(int(dense_indices[position]), 0.0)
            for position in range(len(dense_indices))
        ],
        dtype=np.float32,
    )
    ordering = np.argsort(adjusted_hybrid)[::-1]
    candidates = [int(dense_indices[i]) for i in ordering]
    hybrid_by_index = {
        int(dense_indices[position]): float(adjusted_hybrid[position])
        for position in range(len(dense_indices))
    }

    selected = mmr_select(
        candidates,
        query_vec=query_vec,
        doc_embeddings=embeddings,
        top_k=top_k,
        lambda_mult=mmr_lambda,
        relevance_scores=hybrid_by_index,
    )

    selected_rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected, start=1):
        label = int(labels[idx])
        theme = ", ".join(keywords.get(label, ["mixed"]))
        selected_rows.append(
            {
                "rank": rank,
                "chunk": chunks[idx],
                "citation": f"C{rank}",
                "constellation": f"K{label} ({theme})",
                "dense": float(np.dot(query_vec, embeddings[idx])),
                "lex": float(lex_scores_all[idx]),
                "stale_source_penalty": stale_penalties.get(idx, 0.0),
            }
        )

    use_llm = llm_mode in {"on", "auto"}
    if use_llm:
        try:
            answer = build_llm_answer(query, selected_rows, model=model)
            answer_mode = "llm"
        except Exception as exc:
            if llm_mode == "on":
                raise
            console.print(f"[yellow]LLM fallback:[/yellow] {exc}")
            answer = build_extractive_answer(query, selected_rows)
            answer_mode = "extractive-fallback"
    else:
        answer = build_extractive_answer(query, selected_rows)
        answer_mode = "extractive"

    evidence_posture = build_evidence_posture(selected_rows)
    agreement_signal = build_agreement_signal(selected_rows)

    return {
        "query": query,
        "answer": answer,
        "answer_mode": answer_mode,
        "evidence": selected_rows,
        "source_count": len({row["chunk"].source for row in selected_rows}),
        "source_breakdown": {
            source: sum(1 for row in selected_rows if row["chunk"].source == source)
            for source in sorted({row["chunk"].source for row in selected_rows})
        },
        "constellation_breakdown": {
            constellation: sum(1 for row in selected_rows if row["constellation"] == constellation)
            for constellation in sorted({row["constellation"] for row in selected_rows})
        },
        "evidence_posture": evidence_posture,
        "agreement_signal": agreement_signal,
        "source_filter": source_filter,
        "meta": payload["meta"],
    }


def query_index(
    *,
    index_dir: Path,
    query: str,
    top_k: int = 6,
    mmr_lambda: float = 0.7,
    llm_mode: str = "auto",
    model: str = "gpt-4.1-mini",
    source_filter: str | None = None,
) -> dict[str, Any]:
    payload = load_index(index_dir.resolve())
    return run_query_payload(
        payload=payload,
        query=query,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        llm_mode=llm_mode,
        model=model,
        source_filter=source_filter,
    )


def command_index(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus).resolve()
    index_dir = Path(args.index_dir).resolve()

    docs = read_corpus(corpus_dir)
    if not docs:
        raise SystemExit(f"No .txt/.md files found in {corpus_dir}")

    chunks: list[Chunk] = []
    for source, text in docs:
        chunks.extend(chunk_text(source, text, target_chars=args.chunk_size, overlap_sentences=args.overlap_sentences))

    if len(chunks) < 3:
        raise SystemExit("Need at least 3 chunks to build a useful index.")

    texts = [chunk.text for chunk in chunks]
    retrieval_texts = [build_retrieval_text(chunk) for chunk in chunks]

    console.print(f"[bold]Embedding[/bold] {len(texts)} chunks with model: {args.embedding_model}")
    embedder = EmbeddingEngine(model_name=args.embedding_model)
    embeddings = embedder.encode(retrieval_texts)

    dense = faiss.IndexFlatIP(embeddings.shape[1])
    dense.add(embeddings)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000)
    lexical_matrix = vectorizer.fit_transform(retrieval_texts)

    cluster_count = max(2, min(8, len(chunks) // 4))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    keywords = compute_cluster_keywords(texts, labels)

    save_index(
        index_dir,
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=dense,
        vectorizer=vectorizer,
        tfidf_matrix=lexical_matrix,
        cluster_labels=labels,
        cluster_keywords=keywords,
        model_name=args.embedding_model,
    )

    table = Table(title="Constellation Index Built")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Documents", str(len(docs)))
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Vector Dim", str(embeddings.shape[1]))
    table.add_row("Embedding Mode", embedder.mode)
    table.add_row("Constellations", str(cluster_count))
    table.add_row("Index Dir", str(index_dir))
    console.print(table)


def command_map(args: argparse.Namespace) -> None:
    payload = load_index(Path(args.index_dir).resolve())
    chunks: list[Chunk] = payload["chunks"]
    labels: np.ndarray = payload["cluster_labels"]
    keywords: dict[int, list[str]] = payload["cluster_keywords"]

    counts: dict[int, int] = {}
    for label in labels.tolist():
        counts[int(label)] = counts.get(int(label), 0) + 1

    table = Table(title="Constellation Map")
    table.add_column("Constellation")
    table.add_column("Chunk Count", justify="right")
    table.add_column("Theme")

    for label in sorted(counts):
        theme = ", ".join(keywords.get(label, ["mixed"]))
        table.add_row(f"K{label}", str(counts[label]), theme)

    console.print(table)

    sample = Table(title="Sample Evidence")
    sample.add_column("Chunk")
    sample.add_column("Source")
    sample.add_column("Text")
    for idx, chunk in enumerate(chunks[: min(6, len(chunks))]):
        sample.add_row(chunk.chunk_id, chunk.source, chunk.text[:120] + ("..." if len(chunk.text) > 120 else ""))
    console.print(sample)

    constellation_rows = []
    for label in sorted(counts):
      source_counts: dict[str, int] = {}
      for chunk, chunk_label in zip(chunks, labels.tolist()):
          if int(chunk_label) != label:
              continue
          source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1
      constellation_rows.append(
          {
              "label": f"K{label}",
              "chunk_count": counts[label],
              "theme": keywords.get(label, ["mixed"]),
              "top_sources": [
                  {"source": source, "chunk_count": chunk_count}
                  for source, chunk_count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
              ],
          }
      )

    if args.json_out:
        output_path = Path(args.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "index_dir": str(Path(args.index_dir).resolve()),
                    "constellations": constellation_rows,
                    "sample_chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "text": chunk.text,
                        }
                        for chunk in chunks[: min(6, len(chunks))]
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.report_out:
        write_constellation_report(
            index_dir=Path(args.index_dir).resolve(),
            constellations=constellation_rows,
            sample_chunks=chunks[: min(6, len(chunks))],
            output_path=Path(args.report_out).resolve(),
        )


def write_constellation_report(
    *,
    index_dir: Path,
    constellations: list[dict[str, Any]],
    sample_chunks: list[Chunk],
    output_path: Path,
) -> None:
    report_lines = [
        "# Context Constellation Map Report",
        "",
        f"- Index: {index_dir}",
        f"- Constellation count: {len(constellations)}",
        f"- Sample chunk count: {len(sample_chunks)}",
        "",
        "## Constellations",
        "",
    ]

    for row in constellations:
        theme = ", ".join(row["theme"])
        report_lines.extend(
            [
                f"### {row['label']}",
                f"- Chunk count: {row['chunk_count']}",
                f"- Theme: {theme}",
                f"- Top sources: {', '.join(f'{entry['source']} ({entry['chunk_count']})' for entry in row['top_sources']) or 'none'}",
                "",
            ]
        )

    report_lines.extend(["## Sample Evidence", ""])
    for chunk in sample_chunks:
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = f"{snippet[:217]}..."
        report_lines.extend(
            [
                f"### {chunk.chunk_id}",
                f"- Source: {chunk.source}",
                f"- Snippet: {snippet}",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    evidence_lines = []
    for row in result["evidence"]:
        snippet = row["chunk"].text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = f"{snippet[:217]}..."
        evidence_lines.extend(
            [
                f"### {row['citation']} - {row['chunk'].source}",
                f"- Constellation: {row['constellation']}",
                f"- Dense score: {row['dense']:.4f}",
                f"- Lexical score: {row['lex']:.4f}",
                f"- Snippet: {snippet}",
                "",
            ]
        )

    report_lines = [
        "# Context Constellation Answer Report",
        "",
        f"- Query: {result['query']}",
        f"- Answer mode: {result['answer_mode']}",
        f"- Source coverage: {result['source_count']} unique source(s)",
        f"- Embedding model: {result['meta'].get('embedding_model', 'unknown')}",
        "",
        "## Answer",
        "",
        result["answer"],
        "",
        "## Query Scope",
        "",
        f"- Source filter: {result['source_filter'] or 'none'}",
        "",
        "## Source Breakdown",
        "",
        *[f"- {source}: {count} chunk(s)" for source, count in result["source_breakdown"].items()],
        "",
        "## Constellation Breakdown",
        "",
        *[f"- {constellation}: {count} chunk(s)" for constellation, count in result["constellation_breakdown"].items()],
        "",
        "## Evidence Posture",
        "",
        f"- Coverage: {result['evidence_posture']['coverage_label']}",
        f"- Tension: {result['evidence_posture']['tension_label']}",
        f"- Source count: {result['evidence_posture']['source_count']}",
        f"- Constellation count: {result['evidence_posture']['constellation_count']}",
        f"- Dominant source share: {result['evidence_posture']['dominant_source_share']:.2f}",
        f"- Dominant constellation share: {result['evidence_posture']['dominant_constellation_share']:.2f}",
        "",
        "## Agreement Signal",
        "",
        f"- Label: {result['agreement_signal']['label']}",
        f"- Shared terms: {', '.join(result['agreement_signal']['shared_terms']) or 'none'}",
        f"- Repeated term count: {result['agreement_signal']['repeated_term_count']}",
        f"- Note: {result['agreement_signal']['note']}",
        "",
        "## Evidence",
        "",
        *evidence_lines,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def load_query_suite(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        queries = [line.strip() for line in raw.splitlines() if line.strip()]
        return [{"query": query} for query in queries]

    data = json.loads(raw)
    if isinstance(data, list):
        suite = data
    elif isinstance(data, dict) and isinstance(data.get("queries"), list):
        suite = data["queries"]
    else:
        raise SystemExit("Query suite must be a JSON array, an object with a 'queries' array, or a .txt file.")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(suite, start=1):
        if isinstance(item, str):
            normalized.append({"query": item})
            continue
        if not isinstance(item, dict) or not str(item.get("query", "")).strip():
            raise SystemExit(f"Query suite entry {idx} is missing a non-empty 'query'.")
        normalized.append(
            {
                "query": str(item["query"]).strip(),
                "label": str(item.get("label", "")).strip() or None,
                "source_filter": str(item.get("source_filter", "")).strip() or None,
                "expected_sources": [
                    str(source).strip()
                    for source in item.get("expected_sources", [])
                    if str(source).strip()
                ],
                "forbidden_sources": [
                    str(source).strip()
                    for source in item.get("forbidden_sources", [])
                    if str(source).strip()
                ],
                "conflict_source_groups": [
                    [
                        str(source).strip()
                        for source in group
                        if str(source).strip()
                    ]
                    for group in item.get("conflict_source_groups", [])
                    if isinstance(group, list)
                ],
                "variants": [
                    {
                        "query": str(variant["query"]).strip(),
                        "label": str(variant.get("label", "")).strip() or None,
                        "source_filter": str(variant.get("source_filter", "")).strip() or None,
                    }
                    for variant in item.get("variants", [])
                    if isinstance(variant, dict) and str(variant.get("query", "")).strip()
                ],
            }
        )
    return normalized


def evaluate_query_suite(
    *,
    payload: dict[str, Any],
    queries: list[dict[str, Any]],
    top_k: int,
    mmr_lambda: float,
    llm_mode: str,
    model: str,
    default_source_filter: str | None,
) -> dict[str, Any]:
    embedder = EmbeddingEngine(model_name=payload["meta"]["embedding_model"])
    results = []
    posture_counts: dict[str, int] = {}
    agreement_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    flagged_queries = []
    expected_metric_runs: list[dict[str, Any]] = []
    forbidden_metric_runs: list[dict[str, Any]] = []
    conflict_metric_runs: list[dict[str, Any]] = []

    for idx, entry in enumerate(queries, start=1):
        source_filter = entry.get("source_filter") or default_source_filter
        result = run_query_payload(
            payload=payload,
            query=entry["query"],
            top_k=top_k,
            mmr_lambda=mmr_lambda,
            llm_mode=llm_mode,
            model=model,
            source_filter=source_filter,
            embedder=embedder,
        )
        coverage = result["evidence_posture"]["coverage_label"]
        tension = result["evidence_posture"]["tension_label"]
        agreement = result["agreement_signal"]["label"]
        mode = result["answer_mode"]

        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        posture_key = f"{coverage}/{tension}"
        posture_counts[posture_key] = posture_counts.get(posture_key, 0) + 1
        agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

        risk_flags = []
        if coverage == "narrow":
            risk_flags.append("narrow coverage")
        if tension == "concentrated":
            risk_flags.append("concentrated evidence")
        if agreement == "fragmented":
            risk_flags.append("fragmented agreement")
        if result["source_count"] <= 1:
            risk_flags.append("single-source answer")

        expected_sources = entry.get("expected_sources") or []
        expected_source_matches = find_expected_source_matches(
            expected_sources=expected_sources,
            observed_sources=source_set(result),
        )
        expected_source_metrics = build_expected_source_metrics(
            expected_sources=expected_sources,
            evidence=result["evidence"],
        )
        if expected_sources:
            expected_metric_runs.append(expected_source_metrics)
        if expected_source_matches["missing"]:
            risk_flags.append("missed expected sources")

        forbidden_sources = entry.get("forbidden_sources") or []
        forbidden_source_metrics = build_forbidden_source_metrics(
            forbidden_sources=forbidden_sources,
            evidence=result["evidence"],
        )
        if forbidden_sources:
            forbidden_metric_runs.append(forbidden_source_metrics)
        if forbidden_source_metrics["hit_count"]:
            risk_flags.append("retrieved forbidden sources")

        conflict_source_groups = entry.get("conflict_source_groups") or []
        conflict_source_metrics = build_conflict_source_metrics(
            conflict_source_groups=conflict_source_groups,
            evidence=result["evidence"],
        )
        if conflict_source_groups:
            conflict_metric_runs.append(conflict_source_metrics)
        if (
            conflict_source_metrics["group_count"]
            and conflict_source_metrics["covered_count"] < conflict_source_metrics["group_count"]
        ):
            risk_flags.append("missed conflict evidence")

        variant_results = []
        for variant_idx, variant in enumerate(entry.get("variants") or [], start=1):
            variant_source_filter = variant.get("source_filter") or source_filter
            variant_result = run_query_payload(
                payload=payload,
                query=variant["query"],
                top_k=top_k,
                mmr_lambda=mmr_lambda,
                llm_mode=llm_mode,
                model=model,
                source_filter=variant_source_filter,
                embedder=embedder,
            )
            variant_expected_source_metrics = build_expected_source_metrics(
                expected_sources=expected_sources,
                evidence=variant_result["evidence"],
            )
            if expected_sources:
                expected_metric_runs.append(variant_expected_source_metrics)
            variant_forbidden_source_metrics = build_forbidden_source_metrics(
                forbidden_sources=forbidden_sources,
                evidence=variant_result["evidence"],
            )
            if forbidden_sources:
                forbidden_metric_runs.append(variant_forbidden_source_metrics)
            variant_conflict_source_metrics = build_conflict_source_metrics(
                conflict_source_groups=conflict_source_groups,
                evidence=variant_result["evidence"],
            )
            if conflict_source_groups:
                conflict_metric_runs.append(variant_conflict_source_metrics)
            variant_results.append(
                {
                    "variant_index": variant_idx,
                    "label": variant.get("label") or f"variant-{variant_idx}",
                    "query": variant_result["query"],
                    "source_filter": variant_source_filter,
                    "result": variant_result,
                    "expected_source_metrics": variant_expected_source_metrics,
                    "forbidden_source_metrics": variant_forbidden_source_metrics,
                    "conflict_source_metrics": variant_conflict_source_metrics,
                }
            )

        variant_stability = build_variant_stability_summary(result, variant_results)
        if variant_stability["variant_count"] and not variant_stability["stable"]:
            risk_flags.append("variant-sensitive retrieval")
        if any(item["missing_expected_sources"] for item in variant_stability["details"]):
            risk_flags.append("variant missed expected sources")
        if any(
            variant["forbidden_source_metrics"]["hit_count"]
            for variant in variant_results
        ):
            risk_flags.append("variant retrieved forbidden sources")
        if any(
            metrics["group_count"] and metrics["covered_count"] < metrics["group_count"]
            for metrics in (
                variant["conflict_source_metrics"] for variant in variant_results
            )
        ):
            risk_flags.append("variant missed conflict evidence")

        evaluated = {
            "query_index": idx,
            "label": entry.get("label"),
            "query": result["query"],
            "answer_mode": mode,
            "source_filter": source_filter,
            "expected_sources": expected_sources,
            "expected_source_matches": expected_source_matches,
            "expected_source_metrics": expected_source_metrics,
            "forbidden_sources": forbidden_sources,
            "forbidden_source_metrics": forbidden_source_metrics,
            "conflict_source_groups": conflict_source_groups,
            "conflict_source_metrics": conflict_source_metrics,
            "source_count": result["source_count"],
            "evidence_posture": result["evidence_posture"],
            "agreement_signal": result["agreement_signal"],
            "source_breakdown": result["source_breakdown"],
            "constellation_breakdown": result["constellation_breakdown"],
            "variant_stability": variant_stability,
            "risk_flags": risk_flags,
            "answer_preview": result["answer"][:240].strip(),
        }
        results.append(evaluated)
        if risk_flags:
            flagged_queries.append(evaluated)

    expected_pattern_count = sum(item["expected_count"] for item in expected_metric_runs)
    expected_pattern_match_count = sum(item["matched_count"] for item in expected_metric_runs)
    expected_reciprocal_ranks = [
        detail["reciprocal_rank"]
        for run in expected_metric_runs
        for detail in run["details"]
    ]
    forbidden_pattern_count = sum(item["forbidden_count"] for item in forbidden_metric_runs)
    forbidden_pattern_hit_count = sum(item["hit_count"] for item in forbidden_metric_runs)
    conflict_group_count = sum(item["group_count"] for item in conflict_metric_runs)
    covered_conflict_group_count = sum(item["covered_count"] for item in conflict_metric_runs)
    variant_query_count = sum(1 for item in results if item["variant_stability"]["variant_count"])
    stable_variant_query_count = sum(
        1
        for item in results
        if item["variant_stability"]["variant_count"] and item["variant_stability"]["stable"]
    )

    summary = {
        "query_count": len(results),
        "answer_mode_breakdown": mode_counts,
        "posture_breakdown": posture_counts,
        "agreement_breakdown": agreement_counts,
        "avg_source_count": round(mean(item["source_count"] for item in results), 2) if results else 0.0,
        "avg_dominant_source_share": round(
            mean(item["evidence_posture"]["dominant_source_share"] for item in results), 4
        )
        if results
        else 0.0,
        "avg_dominant_constellation_share": round(
            mean(item["evidence_posture"]["dominant_constellation_share"] for item in results), 4
        )
        if results
        else 0.0,
        "flagged_query_count": len(flagged_queries),
        "flagged_query_rate": round(len(flagged_queries) / len(results), 4) if results else 0.0,
        "expected_source_case_count": len(expected_metric_runs),
        "expected_source_pattern_count": expected_pattern_count,
        "expected_source_pattern_match_count": expected_pattern_match_count,
        "expected_source_recall": (
            round(expected_pattern_match_count / expected_pattern_count, 4)
            if expected_pattern_count
            else 0.0
        ),
        "expected_source_mrr": round(mean(expected_reciprocal_ranks), 4) if expected_reciprocal_ranks else 0.0,
        "forbidden_source_case_count": len(forbidden_metric_runs),
        "forbidden_source_pattern_count": forbidden_pattern_count,
        "forbidden_source_pattern_hit_count": forbidden_pattern_hit_count,
        "forbidden_source_rank_cutoff": 3,
        "forbidden_source_hit_rate": (
            round(forbidden_pattern_hit_count / forbidden_pattern_count, 4)
            if forbidden_pattern_count
            else 0.0
        ),
        "conflict_source_case_count": len(conflict_metric_runs),
        "conflict_source_group_count": conflict_group_count,
        "covered_conflict_source_group_count": covered_conflict_group_count,
        "conflict_source_recall": (
            round(covered_conflict_group_count / conflict_group_count, 4)
            if conflict_group_count
            else 0.0
        ),
        "variant_query_count": variant_query_count,
        "stable_variant_query_count": stable_variant_query_count,
        "variant_stability_rate": (
            round(stable_variant_query_count / variant_query_count, 4) if variant_query_count else 0.0
        ),
        "avg_variant_source_overlap": round(
            mean(
                item["variant_stability"]["avg_source_overlap"]
                for item in results
                if item["variant_stability"]["avg_source_overlap"] is not None
            ),
            4,
        )
        if any(item["variant_stability"]["avg_source_overlap"] is not None for item in results)
        else 0.0,
        "avg_variant_constellation_overlap": round(
            mean(
                item["variant_stability"]["avg_constellation_overlap"]
                for item in results
                if item["variant_stability"]["avg_constellation_overlap"] is not None
            ),
            4,
        )
        if any(item["variant_stability"]["avg_constellation_overlap"] is not None for item in results)
        else 0.0,
    }

    return {
        "summary": summary,
        "results": results,
        "flagged_queries": flagged_queries,
    }


def write_evaluation_report(
    *,
    evaluation: dict[str, Any],
    query_suite_path: Path,
    index_dir: Path,
    output_path: Path,
) -> None:
    summary = evaluation["summary"]
    forbidden_rate = (
        f"{summary['forbidden_source_hit_rate']:.2f}"
        if summary["forbidden_source_pattern_count"]
        else "not evaluated"
    )
    conflict_recall = (
        f"{summary['conflict_source_recall']:.2f}"
        if summary["conflict_source_group_count"]
        else "not evaluated"
    )
    lines = [
        "# Context Constellation Evaluation Report",
        "",
        f"- Query suite: {query_suite_path}",
        f"- Index: {index_dir}",
        f"- Query count: {summary['query_count']}",
        f"- Flagged queries: {summary['flagged_query_count']}",
        f"- Expected source recall: {summary['expected_source_recall']:.2f}",
        f"- Expected source MRR: {summary['expected_source_mrr']:.2f}",
        f"- Forbidden source hit rate (top {summary['forbidden_source_rank_cutoff']}): {forbidden_rate}",
        f"- Conflict source recall: {conflict_recall}",
        f"- Average source count: {summary['avg_source_count']}",
        f"- Average dominant source share: {summary['avg_dominant_source_share']:.2f}",
        f"- Average dominant constellation share: {summary['avg_dominant_constellation_share']:.2f}",
        f"- Queries with variants: {summary['variant_query_count']}",
        f"- Stable variant queries: {summary['stable_variant_query_count']}",
        f"- Variant stability rate: {summary['variant_stability_rate']:.2f}",
        f"- Average variant source overlap: {summary['avg_variant_source_overlap']:.2f}",
        f"- Average variant constellation overlap: {summary['avg_variant_constellation_overlap']:.2f}",
        "",
        "## Answer Modes",
        "",
        *[f"- {mode}: {count}" for mode, count in sorted(summary["answer_mode_breakdown"].items())],
        "",
        "## Evidence Posture Mix",
        "",
        *[f"- {posture}: {count}" for posture, count in sorted(summary["posture_breakdown"].items())],
        "",
        "## Agreement Mix",
        "",
        *[f"- {label}: {count}" for label, count in sorted(summary["agreement_breakdown"].items())],
        "",
    ]

    quality_gate = evaluation.get("quality_gate")
    if quality_gate and quality_gate["configured"]:
        lines.extend(
            [
                "## Quality Gate",
                "",
                f"- Result: {'PASS' if quality_gate['passed'] else 'FAIL'}",
                *[
                    f"- {'PASS' if check['passed'] else 'FAIL'}: {check['label']} "
                    f"{check['actual']:.2f} {check['comparator']} {check['threshold']:.2f}"
                    for check in quality_gate["checks"]
                ],
                "",
            ]
        )

    if evaluation["flagged_queries"]:
        lines.extend(["## Review First", ""])
        for item in evaluation["flagged_queries"]:
            lines.extend(
                [
                    f"### Q{item['query_index']}: {item['label'] or item['query']}",
                    f"- Query: {item['query']}",
                    f"- Flags: {', '.join(item['risk_flags'])}",
                    f"- Posture: {item['evidence_posture']['coverage_label']} / {item['evidence_posture']['tension_label']}",
                    f"- Agreement: {item['agreement_signal']['label']}",
                    f"- Sources: {', '.join(f'{source} ({count})' for source, count in item['source_breakdown'].items())}",
                    f"- Expected source gaps: {', '.join(item['expected_source_matches']['missing']) or 'none'}",
                    f"- Expected source recall: {item['expected_source_metrics']['recall_at_k'] if item['expected_source_metrics']['recall_at_k'] is not None else 'not evaluated'}",
                    f"- Forbidden source hits: {item['forbidden_source_metrics']['hit_count']}",
                    f"- Conflict groups covered: {item['conflict_source_metrics']['covered_count']} / {item['conflict_source_metrics']['group_count']}",
                    (
                        "- Variant stability: "
                        f"{'stable' if item['variant_stability']['stable'] else 'unstable'} "
                        f"(source overlap {item['variant_stability']['avg_source_overlap']:.2f}, "
                        f"constellation overlap {item['variant_stability']['avg_constellation_overlap']:.2f})"
                    )
                    if item["variant_stability"]["variant_count"]
                    else "- Variant stability: not evaluated",
                    "",
                ]
            )

    lines.extend(["## Query Results", ""])
    for item in evaluation["results"]:
        lines.extend(
            [
                f"### Q{item['query_index']}: {item['label'] or item['query']}",
                f"- Query: {item['query']}",
                f"- Answer mode: {item['answer_mode']}",
                f"- Source filter: {item['source_filter'] or 'none'}",
                f"- Expected sources: {', '.join(item['expected_sources']) or 'none'}",
                f"- Forbidden sources: {', '.join(item['forbidden_sources']) or 'none'}",
                f"- Sources: {item['source_count']}",
                f"- Posture: {item['evidence_posture']['coverage_label']} / {item['evidence_posture']['tension_label']}",
                f"- Agreement: {item['agreement_signal']['label']}",
                f"- Risk flags: {', '.join(item['risk_flags']) or 'none'}",
                f"- Expected source gaps: {', '.join(item['expected_source_matches']['missing']) or 'none'}",
                f"- Expected source recall: {item['expected_source_metrics']['recall_at_k'] if item['expected_source_metrics']['recall_at_k'] is not None else 'not evaluated'}",
                f"- Expected source MRR: {item['expected_source_metrics']['mean_reciprocal_rank'] if item['expected_source_metrics']['mean_reciprocal_rank'] is not None else 'not evaluated'}",
                f"- Forbidden source hits: {item['forbidden_source_metrics']['hit_count']}",
                f"- Conflict groups covered: {item['conflict_source_metrics']['covered_count']} / {item['conflict_source_metrics']['group_count']}",
                f"- Answer preview: {item['answer_preview']}",
                "",
            ]
        )
        if item["variant_stability"]["variant_count"]:
            lines.extend(
                [
                    "#### Variant Stability",
                    f"- Stable: {'yes' if item['variant_stability']['stable'] else 'no'}",
                    f"- Average source overlap: {item['variant_stability']['avg_source_overlap']:.2f}",
                    f"- Average constellation overlap: {item['variant_stability']['avg_constellation_overlap']:.2f}",
                    f"- Average chunk overlap: {item['variant_stability']['avg_chunk_overlap']:.2f}",
                    f"- Posture mismatches: {item['variant_stability']['posture_mismatch_count']}",
                    f"- Agreement mismatches: {item['variant_stability']['agreement_mismatch_count']}",
                    "",
                ]
            )
            for variant in item["variant_stability"]["details"]:
                expected_recall = variant["expected_source_recall"]
                lines.extend(
                    [
                        f"- {variant['label']}: source overlap {variant['source_overlap']:.2f}, constellation overlap {variant['constellation_overlap']:.2f}, chunk overlap {variant['chunk_overlap']:.2f}, posture match {'yes' if variant['posture_match'] else 'no'}, agreement match {'yes' if variant['agreement_match'] else 'no'}, expected source recall {f'{expected_recall:.2f}' if expected_recall is not None else 'not evaluated'}",
                    ]
                )
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def command_ask(args: argparse.Namespace) -> None:
    result = query_index(
        index_dir=Path(args.index_dir),
        query=args.query,
        top_k=args.top_k,
        mmr_lambda=args.mmr_lambda,
        llm_mode=args.llm,
        model=args.model,
        source_filter=args.source_filter,
    )

    if args.json_out:
        output_path = Path(args.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "query": result["query"],
            "answer": result["answer"],
            "answer_mode": result["answer_mode"],
            "source_count": result["source_count"],
            "source_breakdown": result["source_breakdown"],
            "constellation_breakdown": result["constellation_breakdown"],
            "evidence_posture": result["evidence_posture"],
            "agreement_signal": result["agreement_signal"],
            "source_filter": result["source_filter"],
            "meta": result["meta"],
            "evidence": [
                {
                    "rank": row["rank"],
                    "citation": row["citation"],
                    "constellation": row["constellation"],
                    "dense": row["dense"],
                    "lex": row["lex"],
                    "stale_source_penalty": row["stale_source_penalty"],
                    "chunk": asdict(row["chunk"]),
                }
                for row in result["evidence"]
            ],
        }
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    if args.report_out:
        write_markdown_report(result, Path(args.report_out).resolve())

    console.print("\n[bold]Answer[/bold]")
    console.print(result["answer"])
    if result["source_filter"]:
        console.print(f"\n[bold]Source filter[/bold] {result['source_filter']}")
    console.print(f"\n[bold]Source coverage[/bold] {result['source_count']} unique source(s)")
    for source, count in result["source_breakdown"].items():
        console.print(f"- {source}: {count} chunk(s)")
    console.print(
        f"\n[bold]Evidence posture[/bold] {result['evidence_posture']['coverage_label']} coverage, "
        f"{result['evidence_posture']['tension_label']} support"
    )
    console.print(
        f"[bold]Agreement signal[/bold] {result['agreement_signal']['label']} "
        f"({', '.join(result['agreement_signal']['shared_terms']) or 'no repeated terms'})"
    )

    trace = Table(title="Evidence Trace")
    trace.add_column("Citation")
    trace.add_column("Source")
    trace.add_column("Constellation")
    trace.add_column("Snippet")
    for row in result["evidence"]:
        snippet = row["chunk"].text[:160] + ("..." if len(row["chunk"].text) > 160 else "")
        trace.add_row(row["citation"], row["chunk"].source, row["constellation"], snippet)
    console.print(trace)


def command_evaluate(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).resolve()
    query_suite_path = Path(args.queries).resolve()
    payload = load_index(index_dir)
    query_suite = load_query_suite(query_suite_path)
    evaluation = evaluate_query_suite(
        payload=payload,
        queries=query_suite,
        top_k=args.top_k,
        mmr_lambda=args.mmr_lambda,
        llm_mode=args.llm,
        model=args.model,
        default_source_filter=args.source_filter,
    )
    evaluation["quality_gate"] = build_quality_gate(
        summary=evaluation["summary"],
        min_expected_source_recall=args.min_expected_source_recall,
        min_expected_source_mrr=args.min_expected_source_mrr,
        min_variant_stability_rate=args.min_variant_stability_rate,
        max_flagged_query_rate=args.max_flagged_query_rate,
        max_forbidden_source_hit_rate=args.max_forbidden_source_hit_rate,
        min_conflict_source_recall=args.min_conflict_source_recall,
    )

    summary = evaluation["summary"]
    forbidden_rate = (
        f"{summary['forbidden_source_hit_rate']:.2f}"
        if summary["forbidden_source_pattern_count"]
        else "not evaluated"
    )
    conflict_recall = (
        f"{summary['conflict_source_recall']:.2f}"
        if summary["conflict_source_group_count"]
        else "not evaluated"
    )
    table = Table(title="Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Queries", str(summary["query_count"]))
    table.add_row("Flagged", str(summary["flagged_query_count"]))
    table.add_row("Expected source recall", f"{summary['expected_source_recall']:.2f}")
    table.add_row("Expected source MRR", f"{summary['expected_source_mrr']:.2f}")
    table.add_row(
        f"Forbidden source hit rate (top {summary['forbidden_source_rank_cutoff']})",
        forbidden_rate,
    )
    table.add_row("Conflict source recall", conflict_recall)
    table.add_row("Avg sources", str(summary["avg_source_count"]))
    table.add_row("Avg dominant source share", f"{summary['avg_dominant_source_share']:.2f}")
    table.add_row("Avg dominant constellation share", f"{summary['avg_dominant_constellation_share']:.2f}")
    table.add_row("Queries with variants", str(summary["variant_query_count"]))
    table.add_row("Stable variant queries", str(summary["stable_variant_query_count"]))
    table.add_row("Variant stability rate", f"{summary['variant_stability_rate']:.2f}")
    table.add_row("Avg variant source overlap", f"{summary['avg_variant_source_overlap']:.2f}")
    table.add_row("Avg variant constellation overlap", f"{summary['avg_variant_constellation_overlap']:.2f}")
    console.print(table)

    console.print("\n[bold]Answer modes[/bold]")
    for mode, count in sorted(summary["answer_mode_breakdown"].items()):
        console.print(f"- {mode}: {count}")

    console.print("\n[bold]Evidence posture mix[/bold]")
    for posture, count in sorted(summary["posture_breakdown"].items()):
        console.print(f"- {posture}: {count}")

    console.print("\n[bold]Agreement mix[/bold]")
    for label, count in sorted(summary["agreement_breakdown"].items()):
        console.print(f"- {label}: {count}")

    if evaluation["flagged_queries"]:
        console.print("\n[bold]Flagged queries[/bold]")
        for item in evaluation["flagged_queries"]:
            console.print(
                f"- Q{item['query_index']} {item['label'] or item['query']}: "
                f"{', '.join(item['risk_flags'])}"
            )

    quality_gate = evaluation["quality_gate"]
    if quality_gate["configured"]:
        gate_label = "PASS" if quality_gate["passed"] else "FAIL"
        gate_style = "green" if quality_gate["passed"] else "red"
        console.print(f"\n[{gate_style}][bold]Quality gate: {gate_label}[/bold][/{gate_style}]")
        for check in quality_gate["checks"]:
            check_label = "PASS" if check["passed"] else "FAIL"
            console.print(
                f"- {check_label}: {check['label']} {check['actual']:.2f} "
                f"{check['comparator']} {check['threshold']:.2f}"
            )

    if args.json_out:
        output_path = Path(args.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    if args.report_out:
        write_evaluation_report(
            evaluation=evaluation,
            query_suite_path=query_suite_path,
            index_dir=index_dir,
            output_path=Path(args.report_out).resolve(),
        )

    if quality_gate["configured"] and not quality_gate["passed"]:
        raise SystemExit(1)


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context Constellation RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build embedding and lexical index")
    p_index.add_argument("--corpus", required=True, help="Path to folder containing .txt/.md files")
    p_index.add_argument("--index-dir", required=True, help="Output directory for index artifacts")
    p_index.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name, or 'hashing' for deterministic offline embeddings.",
    )
    p_index.add_argument("--chunk-size", type=int, default=520)
    p_index.add_argument("--overlap-sentences", type=int, default=1)
    p_index.set_defaults(func=command_index)

    p_ask = sub.add_parser("ask", help="Query the constellation index")
    p_ask.add_argument("--index-dir", required=True)
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--top-k", type=int, default=6)
    p_ask.add_argument("--mmr-lambda", type=float, default=0.7)
    p_ask.add_argument("--llm", choices=["off", "auto", "on"], default="auto")
    p_ask.add_argument("--model", default="gpt-4.1-mini")
    p_ask.add_argument("--source-filter", help="Optional regex applied to source paths before ranking evidence.")
    p_ask.add_argument("--json-out", help="Optional path to save the full answer + evidence payload as JSON")
    p_ask.add_argument("--report-out", help="Optional path to save a markdown analyst report with answer and evidence.")
    p_ask.set_defaults(func=command_ask)

    p_map = sub.add_parser("map", help="Show discovered constellation clusters")
    p_map.add_argument("--index-dir", required=True)
    p_map.add_argument("--json-out", help="Optional path to save the constellation map as JSON")
    p_map.add_argument("--report-out", help="Optional path to save a markdown constellation map report")
    p_map.set_defaults(func=command_map)

    p_eval = sub.add_parser("evaluate", help="Run a repeatable batch evaluation over a query suite")
    p_eval.add_argument("--index-dir", required=True)
    p_eval.add_argument("--queries", required=True, help="Path to a .json/.txt query suite")
    p_eval.add_argument("--top-k", type=int, default=6)
    p_eval.add_argument("--mmr-lambda", type=float, default=0.7)
    p_eval.add_argument("--llm", choices=["off", "auto", "on"], default="auto")
    p_eval.add_argument("--model", default="gpt-4.1-mini")
    p_eval.add_argument("--source-filter", help="Optional default regex applied to source paths for every query.")
    p_eval.add_argument("--json-out", help="Optional path to save evaluation results as JSON")
    p_eval.add_argument("--report-out", help="Optional path to save a markdown evaluation report")
    p_eval.add_argument(
        "--min-expected-source-recall",
        type=unit_interval,
        help="Fail when recall across expected source patterns and query variants is below this value.",
    )
    p_eval.add_argument(
        "--min-expected-source-mrr",
        type=unit_interval,
        help="Fail when expected source mean reciprocal rank is below this value.",
    )
    p_eval.add_argument(
        "--min-variant-stability-rate",
        type=unit_interval,
        help="Fail when the share of stable variant-backed queries is below this value.",
    )
    p_eval.add_argument(
        "--max-flagged-query-rate",
        type=unit_interval,
        help="Fail when the share of queries with evidence risk flags exceeds this value.",
    )
    p_eval.add_argument(
        "--max-forbidden-source-hit-rate",
        type=unit_interval,
        help="Fail when explicitly declared distractor patterns appear in the first three evidence ranks above this rate.",
    )
    p_eval.add_argument(
        "--min-conflict-source-recall",
        type=unit_interval,
        help="Fail when retrieval covers fewer than this share of declared conflict source groups.",
    )
    p_eval.set_defaults(func=command_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
