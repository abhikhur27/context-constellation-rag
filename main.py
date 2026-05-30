from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
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
            docs.append((str(path.relative_to(corpus_dir)), raw.replace("\ufeff", "")))
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


class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.mode = "sentence-transformers"
        self.model = None
        self.fallback_vectorizer: HashingVectorizer | None = None
        self.dim = 1024

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


def mmr_select(candidate_indices: list[int], query_vec: np.ndarray, doc_embeddings: np.ndarray, top_k: int, lambda_mult: float = 0.7) -> list[int]:
    chosen: list[int] = []
    remaining = candidate_indices.copy()

    while remaining and len(chosen) < top_k:
        best_idx = remaining[0]
        best_score = -1e9
        for idx in remaining:
            rel = float(np.dot(query_vec, doc_embeddings[idx]))
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

    console.print(f"[bold]Embedding[/bold] {len(texts)} chunks with model: {args.embedding_model}")
    embedder = EmbeddingEngine(model_name=args.embedding_model)
    embeddings = embedder.encode(texts)

    dense = faiss.IndexFlatIP(embeddings.shape[1])
    dense.add(embeddings)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000)
    lexical_matrix = vectorizer.fit_transform(texts)

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


def command_ask(args: argparse.Namespace) -> None:
    payload = load_index(Path(args.index_dir).resolve())
    chunks: list[Chunk] = payload["chunks"]
    embeddings: np.ndarray = payload["embeddings"]
    dense_index = payload["dense"]
    vectorizer: TfidfVectorizer = payload["vectorizer"]
    lexical_matrix = payload["lexical_matrix"]
    labels: np.ndarray = payload["cluster_labels"]
    keywords: dict[int, list[str]] = payload["cluster_keywords"]
    model_name: str = payload["meta"]["embedding_model"]

    embedder = EmbeddingEngine(model_name=model_name)
    query_vec = embedder.encode([args.query])[0]

    dense_k = min(max(args.top_k * 4, args.top_k), len(chunks))
    scores, idxs = dense_index.search(query_vec.reshape(1, -1), dense_k)
    dense_scores = scores[0]
    dense_indices = idxs[0]

    query_lex = vectorizer.transform([args.query])
    lex_scores_all = (lexical_matrix @ query_lex.T).toarray().ravel()

    dense_norm = normalize_scores(dense_scores)
    lex_subset = np.asarray([lex_scores_all[i] for i in dense_indices], dtype=np.float32)
    lex_norm = normalize_scores(lex_subset)

    hybrid = 0.7 * dense_norm + 0.3 * lex_norm
    ordering = np.argsort(hybrid)[::-1]
    candidates = [int(dense_indices[i]) for i in ordering]

    selected = mmr_select(candidates, query_vec=query_vec, doc_embeddings=embeddings, top_k=args.top_k, lambda_mult=args.mmr_lambda)

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
            }
        )

    answer: str
    use_llm = args.llm in {"on", "auto"}
    if use_llm:
        try:
            answer = build_llm_answer(args.query, selected_rows, model=args.model)
        except Exception as exc:
            if args.llm == "on":
                raise
            console.print(f"[yellow]LLM fallback:[/yellow] {exc}")
            answer = build_extractive_answer(args.query, selected_rows)
    else:
        answer = build_extractive_answer(args.query, selected_rows)

    console.print("\n[bold]Answer[/bold]")
    console.print(answer)

    trace = Table(title="Evidence Trace")
    trace.add_column("Citation")
    trace.add_column("Source")
    trace.add_column("Constellation")
    trace.add_column("Snippet")
    for row in selected_rows:
        snippet = row["chunk"].text[:160] + ("..." if len(row["chunk"].text) > 160 else "")
        trace.add_row(row["citation"], row["chunk"].source, row["constellation"], snippet)
    console.print(trace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context Constellation RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build embedding and lexical index")
    p_index.add_argument("--corpus", required=True, help="Path to folder containing .txt/.md files")
    p_index.add_argument("--index-dir", required=True, help="Output directory for index artifacts")
    p_index.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
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
    p_ask.set_defaults(func=command_ask)

    p_map = sub.add_parser("map", help="Show discovered constellation clusters")
    p_map.add_argument("--index-dir", required=True)
    p_map.set_defaults(func=command_map)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
