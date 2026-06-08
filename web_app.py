from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from main import load_index, query_index

app = Flask(__name__)
DEFAULT_INDEX = Path("artifacts/index")


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Context Constellation RAG</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #111a2e;
      --text: #edf2ff;
      --muted: #a9b4d6;
      --line: rgba(237,242,255,0.12);
      --accent: #66e2c0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 15% 15%, #1a2648 0%, var(--bg) 55%);
      color: var(--text);
      font-family: "Inter", "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    .wrap {
      width: min(1120px, calc(100% - 2rem));
      margin: 2rem auto 3rem;
      display: grid;
      gap: 1rem;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem 1.1rem;
    }
    h1 { margin: 0 0 0.4rem; font-size: 1.45rem; }
    .muted { color: var(--muted); }
    .query-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 0.65rem;
      align-items: center;
    }
    input, select, button {
      min-height: 44px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0d1528;
      color: var(--text);
      padding: 0.65rem 0.8rem;
      font: inherit;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #051514;
      border: none;
      font-weight: 700;
      padding-inline: 1rem;
    }
    pre {
      white-space: pre-wrap;
      margin: 0;
      background: #0d1528;
      border-radius: 10px;
      padding: 0.8rem;
      border: 1px solid var(--line);
      color: #dff8f1;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.93rem;
    }
    th, td {
      border-top: 1px solid var(--line);
      padding: 0.6rem 0.4rem;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); }
    .badge {
      display: inline-block;
      padding: 0.2rem 0.45rem;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.78rem;
    }
    @media (max-width: 760px) {
      .query-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="panel">
      <h1>Context Constellation RAG</h1>
      <p class="muted">Hybrid retrieval + evidence-first answers. Built for practical RAG portfolio demos.</p>
      <div class="query-row">
        <input id="query" value="Where is rollout risk highest and why?" />
        <select id="top-k">
          <option value="4">Top 4</option>
          <option value="6" selected>Top 6</option>
          <option value="8">Top 8</option>
        </select>
        <select id="llm">
          <option value="off">LLM Off (Extractive)</option>
          <option value="auto">LLM Auto</option>
          <option value="on">LLM On</option>
        </select>
        <button id="ask">Ask</button>
      </div>
      <p class="muted" id="status">Ready.</p>
    </section>

    <section class="panel">
      <h2>Answer</h2>
      <p><span class="badge" id="mode">mode: -</span></p>
      <pre id="answer">Run a query to generate a grounded response.</pre>
    </section>

    <section class="panel">
      <h2>Evidence Trace</h2>
      <table>
        <thead>
          <tr><th>Citation</th><th>Source</th><th>Constellation</th><th>Snippet</th></tr>
        </thead>
        <tbody id="evidence"></tbody>
      </table>
    </section>
  </main>

  <script>
    const queryEl = document.getElementById('query');
    const llmEl = document.getElementById('llm');
    const topKEl = document.getElementById('top-k');
    const askBtn = document.getElementById('ask');
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const evidenceEl = document.getElementById('evidence');
    const modeEl = document.getElementById('mode');

    async function runAsk() {
      const query = queryEl.value.trim();
      if (!query) return;
      askBtn.disabled = true;
      statusEl.textContent = 'Running retrieval...';

      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, llm: llmEl.value, top_k: Number(topKEl.value) })
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || 'Request failed');
        }

        answerEl.textContent = data.answer;
        modeEl.textContent = `mode: ${data.answer_mode}`;
        evidenceEl.innerHTML = '';
        data.evidence.forEach((row) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${row.citation}</td>
            <td>${row.source}</td>
            <td>${row.constellation}</td>
            <td>${row.snippet}</td>
          `;
          evidenceEl.appendChild(tr);
        });

        statusEl.textContent = `Retrieved ${data.evidence.length} evidence chunks with Top ${topKEl.value}.`;
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
      } finally {
        askBtn.disabled = false;
      }
    }

    askBtn.addEventListener('click', runAsk);
    queryEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') runAsk();
    });
  </script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    ready = DEFAULT_INDEX.exists()
    return jsonify({"ok": True, "index_ready": ready, "index_dir": str(DEFAULT_INDEX)})


@app.post("/api/ask")
def ask() -> tuple[dict, int] | dict:
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}, 400

    top_k = int(payload.get("top_k", 6))
    llm_mode = str(payload.get("llm", "auto"))

    if not DEFAULT_INDEX.exists():
        return {
            "error": "Index not found. Run: python main.py index --corpus example_corpus --index-dir artifacts/index"
        }, 400

    try:
        result = query_index(
            index_dir=DEFAULT_INDEX,
            query=query,
            top_k=max(1, min(top_k, 12)),
            llm_mode=llm_mode if llm_mode in {"off", "auto", "on"} else "auto",
        )
    except Exception as exc:
        return {"error": str(exc)}, 500

    evidence = []
    for row in result["evidence"]:
        snippet = row["chunk"].text[:200] + ("..." if len(row["chunk"].text) > 200 else "")
        evidence.append(
            {
                "citation": row["citation"],
                "source": row["chunk"].source,
                "constellation": row["constellation"],
                "snippet": snippet,
            }
        )

    return {
        "answer": result["answer"],
        "answer_mode": result["answer_mode"],
        "evidence": evidence,
    }


if __name__ == "__main__":
    app.run(debug=True, port=7860)
