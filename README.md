# Multimodal Submission Integrity & Plagiarism Detection Engine

A working prototype that ingests a corpus of multi-format submissions (**PDF / DOCX / MP4**),
extracts and normalizes their textual content, indexes chunk embeddings, and produces
**explainable** integrity reports combining three signals:

1. **Internal peer-to-peer similarity** — who copied whom in the corpus (with exact passages + page numbers / video timestamps).
2. **External / web-reference similarity** — overlap with an open reference knowledge base (simulated web corpus).
3. **AI-generated content likelihood** — reproducible statistical detector (optional Hugging Face RoBERTa backend).

Built for the AI/ML Engineer assessment (Unstop). Everything runs on commodity CPU; no paid APIs.

---

## Quickstart (single command path)

```bash
# 1. Set up
pip install -r requirements.txt

# 2. (optional, one-time, enables fully offline runs) pre-download models
python scripts/download_models.py

# 3. (optional) regenerate the synthetic evaluation dataset (15 files, ~2 min)
python make_sample_data.py --force

# 4. Run the pipeline end-to-end over the sample corpus
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

Outputs land in `./reports`:

```
reports/
├── per_document/            one .json + one human-readable .md audit report per submission
├── summary.md               batch summary + pairwise peer-coverage matrix
├── similarity_matrix.json   exact pairwise coverage numbers
├── processing_log.json      config snapshot + per-file warnings/errors
└── pipeline.log             full run log
```

CLI options: `--whisper_model tiny|base|small`, `--ai_backend heuristic|hf`,
`--embedding_model <st-model-id>`, `--match_threshold 0.75`, `--fetch-web` (grow the
reference KB from Wikipedia's free REST API at runtime; graceful fallback when offline).

---

## What the pipeline does (methodology per task)

### Task 1 — Multimodal ingestion & normalization
| Modality | Extractor | Metadata preserved |
|---|---|---|
| `.pdf`  | `pypdf` page-by-page text, repeated running headers/footers removed by cross-page frequency analysis | **real page numbers** per sentence |
| `.docx` | `python-docx` paragraphs; page numbers estimated (28 paragraphs/page) since DOCX has no hard pagination | estimated page |
| `.mp4`  | ffmpeg audio extraction (PATH binary or the static binary bundled by `imageio-ffmpeg`) → silence detection → **faster-whisper** (tiny/base, int8, CPU) with **word-level timestamps** | **video timestamps** `00:00:24 - 00:00:36` |

All text passes one normalizer: NFKC unicode folding (ligatures `ﬁ→fi`), hyphenated
line-wrap re-joining, whitespace collapse, page-number-line removal. Text is segmented
into **overlapping sentence-window chunks (~120 words, ~35-word overlap)**, sized below
MiniLM's 256-token context window so nothing is silently truncated (see Assumptions).
Corrupt / empty / silent files return a structured error and are skipped without
crashing the run (Scenario 3).

### Task 2 — Indexing & internal peer-to-peer similarity
- Chunks → **`sentence-transformers/all-MiniLM-L6-v2`** (384-d, L2-normalized, CPU ~1k chunks/s).
- Index: exact **NumPy cosine** (`normalized @ normalized.T`) by default; **FAISS `IndexFlatIP`** auto-selected if installed. Both are exact on prototype scale (see Task 5 for approximate structures at production scale).
- For **every** chunk of a document we compute its best cosine match **per other document**
  (self-matches are excluded by construction). Document-pair score =
  **coverage**: the share of the document's non-boilerplate words whose chunk matches the
  other document at cosine ≥ 0.75. This is far more actionable than a mean similarity:
  *"96% of this document aligns with submission_01.pdf"*.

### Task 3 — Multi-signal integrity detection
- **AI-generated estimation** (`integrity/ai_detector.py`): default is a fully reproducible
  statistical detector — low sentence-length burstiness, discourse-marker density,
  absence of contractions, polished punctuation, absent first-person voice, combined with
  calibrated weights and logistic squashing. `--ai_backend hf` switches to
  `Hello-SimpleAI/chatgpt-detector-roberta` with automatic fallback to the heuristic.
- **External plagiarism** (`integrity/external.py`): submissions are matched against a
  local reference knowledge base of six open Wikipedia-style articles
  (`reference_corpus/*.txt`) — the brief-sanctioned simulation of web search.
  `--fetch-web` optionally extends the KB from Wikipedia's free REST summary API
  (no key; network failures fall back to local-only scoring).
- Signals are combined into `scores`: `peer_similarity_percentage`,
  `external_similarity_percentage`, `ai_generated_probability`.

### Task 4 — Citation & explainability engine
Every flagged overlap emits the **query passage**, the **matched passage**, the matched
file, **page numbers or video timestamp ranges**, cosine similarity, and lexical Jaccard
(so reviewers can see *"semantically similar but lexically different"* at a glance).
Shared institutional boilerplate is **detected at corpus level** (near-duplicate chunk
clusters spanning ≥3 files at cosine ≥ 0.90) and **discounted** from scores and citations
(Scenario 4). Each `.md` report puts the verdict + top citations on one screen —
verifiable in <30 seconds.

### Task 5 — Scaling brief: 15 files → 10,000 submissions
See [**Scaling Architecture (10,000 submissions)**](#scaling-architecture-10000-submissions) below.

---

## Scenario coverage (as specified in the brief)

| Scenario | Where it is handled | Where to verify |
|---|---|---|
| 1. Cross-modal duplication (MP4 reads a PDF's paragraph) | Whisper word-timestamps → transcript chunks match PDF chunks; report cites `query_timestamp` + `matched_page` | `pitch_12.mp4` ↔ `submission_03.pdf`, `multimodal_matches` |
| 2. Paraphrased content | Semantic embeddings match synonym-swapped text that lexical metrics score low | `submission_09.docx` ↔ `submission_04.docx` (high cosine, low Jaccard) |
| 3. Corrupted / empty files | Zero-byte PDF → structured error; silent/no-audio MP4 → structured error; run continues | `submission_11.pdf`, `pitch_13.mp4` + `processing_log.json` |
| 4. Benign common formatting | Cross-corpus near-duplicate cluster detection discounts the shared case prompt | `boilerplate_discount` in every report; only genuine body overlap is scored |
| 5. Multi-document stress batch | 15-file mixed batch → per-doc reports + aggregate summary + matrix | `reports/summary.md` |

---

## Repository layout

```
run_integrity_check.py      # CLI entry point (assignment's required command)
make_sample_data.py         # regenerates the 15-file dataset + dataset_manifest.json
requirements.txt
config/config.json          # thresholds, model ids, risk weights
integrity/
├── models.py               # Chunk / ExtractedDoc dataclasses
├── normalize.py            # cleaning, sentence splitting, overlapping chunker
├── extractors/
│   ├── pdf_extractor.py    # pypdf + header/footer stripping
│   ├── docx_extractor.py   # python-docx + estimated pagination
│   └── mp4_extractor.py    # ffmpeg audio → faster-whisper → timed chunks
├── embeddings.py           # MiniLM backend + documented TF-IDF fallback
├── indexer.py              # NumPy exact cosine index / FAISS IndexFlatIP
├── similarity.py           # per-chunk best-match alignment, boilerplate clustering, coverage
├── ai_detector.py          # burstiness heuristic + optional HF RoBERTa detector
├── external.py             # reference-KB matcher + optional Wikipedia fetcher
├── reporter.py             # JSON (assignment schema) + Markdown reports
└── pipeline.py             # orchestration
reference_corpus/           # external "web" knowledge base (6 open texts)
sample_submissions/         # 15 synthetic files + dataset_manifest.json (ground truth)
scripts/download_models.py  # offline model pre-fetch
scripts/eval_metrics.py     # computes P/R/F1 vs ground truth from generated reports
EVALUATION.md               # metrics vs ground truth + failure analysis
```

---

## Output format

Each `per_document/*.json` follows the assignment schema exactly, with a few
clearly-named additive fields:

```json
{
  "document_id": "submission_02.docx",
  "total_integrity_risk": "HIGH",
  "scores": {
    "peer_similarity_percentage": 96.2,
    "external_similarity_percentage": 0.0,
    "ai_generated_probability": 18.4
  },
  "peer_matches": [
    {
      "matched_file": "submission_01.pdf",
      "similarity_score": 0.99,
      "query_passage": "Rooftop solar paired with neighbourhood batteries ...",
      "matched_passage": "Rooftop solar paired with neighbourhood batteries ...",
      "location": { "query_page": 1, "matched_page": 1 }
    }
  ],
  "multimodal_matches": [
    {
      "matched_file": "pitch_12.mp4",
      "similarity_score": 0.94,
      "query_passage": "Operating margins for the city bus network ...",
      "matched_passage": "operating margins for the city bus network ...",
      "location": { "query_page": 1, "matched_timestamp": "00:00:08 - 00:00:44" }
    }
  ],
  "explanations": "Peer overlap: 96% of this document aligns with submission_01.pdf ...",
  "external_matches": [ ... ],
  "boilerplate_discount": { "chunks": 1, "words": 118, "files_sharing_template": 11 },
  "processing": { "warnings": [ ... ], "error": null, ... }
}
```

Risk labels: composite = `0.5·peer + 0.3·external + 0.2·ai`; **HIGH** ≥ 60 (or any single
pair ≥ 70% coverage), **MEDIUM** ≥ 35, else **LOW**. Floors: AI-likelihood ≥ 60 and
external overlap ≥ 50 always raise a document to at least **MEDIUM** (probable machine
generation or half-copied external content deserves review even when the weighted
composite stays low). Weights live in `config/config.json`.

---

## Assumptions (documented decisions)

1. **Chunk size** — the brief suggests 200–400 tokens; `all-MiniLM-L6-v2` truncates at
   256 tokens, so chunks are built at **~120 words (~160 tokens)** with 35-word overlap.
   This honours the overlap requirement while guaranteeing no chunk is silently
   half-embedded, and finer chunks keep coverage numbers honest. Chunk/overlap sizes
   are config keys, not hard-coded.
2. **DOCX pagination** — DOCX has no fixed pages; pages are estimated at 28 paragraphs
   per page and labelled "estimated". Positions remain stable and verifiable.
3. **Peer score definition** — "peer similarity %" is implemented as **coverage**
   (share of the document's words aligned to another document at cosine ≥ threshold),
   because a bare average hides exactly the copying reviewers need to see. Pairwise
   coverage both directions is exported in `similarity_matrix.json`.
4. **AI detection** — no paid API; the default backend is a documented statistical
   heuristic (reproducible, offline, honest about being a proxy). A real HF
   transformer backend is wired in behind a flag. Neither is treated as proof —
   reports always present it as an *estimate* with feature-level explanations.
5. **External engine** — a local open reference KB simulates web search (brief-sanctioned);
   `--fetch-web` shows the live extension path without making runs non-deterministic.
6. **Boilerplate** — discounting triggers only when a near-duplicate chunk cluster spans
   ≥3 files at cosine ≥ 0.90, so one student quoting a common phrase cannot hide behind it.
7. **English focus** — Whisper pinned to `language="en"` for sample-corpus fidelity;
   language detection is recorded and the pin is a one-line config change.

## AI-assistance disclosure (required by the brief)

Per the assessment rules ("you may use AI coding assistants for boilerplate and
debugging, but you must disclose your usage in your README"): this repository was
**developed with the assistance of an AI coding assistant** used for scaffolding
boilerplate, drafting documentation, and debugging. All architectural decisions,
methodology choices, thresholds and evaluation results were designed/curated by the
candidate and are explained in this README and EVALUATION.md. The candidate must be
(and is prepared to be) able to explain every module and code line in a follow-up
technical interview.

---

# Scaling Architecture (10,000 submissions)

The prototype is single-process and exact; production changes are about **queues,
batching, ANN indexes and storage**, not new ML logic. Concretely:

### 1. Ingestion & asynchronous processing
- Event-driven ingestion: upload → object store (S3/GCS) → **job per file** on a queue
  (SQS/Redis Streams). Workers are stateless containers; extraction and ASR are
  CPU/GPU-bound, so they scale horizontally.
- **Stage A (cheap, CPU fleet):** PDF/DOCX parsing + chunking. ~15 files/s/core in the
  prototype → 10,000 files ≈ 2,000 CPU-seconds ≈ **12 cores × 3 min**. Trivial.
- **Stage B (GPU pool for ASR):** `faster-whisper` base on an L4/T4 transcribes the
  15-file corpus in ~40 s; at 10,000 files with ~45 s audio each ≈ 125 GPU-hours →
  a **2-GPU pool finishes in ~2.6 days**, 8 GPUs in ~16 h. Videos are short
  (assignment cap: 60 s), which keeps this tractable. Autoscale on queue depth.
- Embeddings on the same GPU workers in batches of 256; MiniLM does ~2,000 chunks/s/GPU
  → 10,000 docs × ~40 chunks = **400k chunks ≈ 3–4 GPU-minutes total**. Embedding is
  never the bottleneck; ASR is.

### 2. Vector database & index strategy
| Scale | Choice | Why |
|---|---|---|
| ≤ 50k chunks (≈1,000 docs) | Flat IP (exact) — NumPy/FAISS | 50k×384 fp32 = 76 MB; brute force <10 ms/query. Exactness matters for citations. |
| 50k–5M chunks (10,000+ docs) | **HNSW** (FAISS/pgvector/Chroma/Qdrant) | ~0.95+ recall@10, <5 ms p50, incremental inserts, supports re-ranking with exact search over the candidate set. |
| Memory-bound / >5M | IVF-PQ | 32–64× memory compression; accept ~0.90 recall, restore precision via exact re-score of top-100 candidates. |

The peer-similarity *job* becomes per-document batched top-k (k=8 per chunk) instead of
the prototype's full matrix — 400k query vectors × HNSW is minutes, not hours. The
**coverage aggregation, boilerplate clustering and alignment/citation logic are
unchanged**; only `best_matches_for_chunks` swaps its backend.

### 3. Storage & state
- Object store for binaries; **Postgres** (JSONB) for reports/manifests, with the vector
  index either in-process (FAISS snapshot on S3, rebuilt from a embeddings parquet
  dataset) or a managed vector DB (Qdrant/pgvector) when multi-writer concurrency is needed.
- Idempotent jobs keyed by (file_sha256, pipeline_version) so re-runs never double-charge.

### 4. Cost, throughput & latency bottlenecks
- **ASR = ~90% of compute cost.** Mitigations: batch VAD-trimmed audio, int8 models,
  GPU autoscaling, and result caching (same audio → same transcript). ≈ $0.0005/file GPU cost → **$5 per 10,000-file batch** at L4 spot pricing.
- **Pairwise fan-out** is O(N) per doc with ANN (not O(N²) full-matrix): 10,000 docs is
  40M query-chunk searches ≈ well under an hour on one HNSW node.
- **Latency SLO:** per-file verdict < 90 s p95 (extract 2 s + ASR 45 s + embed 0.2 s +
  match <5 s + report <1 s), measured end-to-end per job.
- Correctness knob: exact re-ranking of top-k candidates keeps citations bit-identical to
  the prototype's math (cosine on L2-normalized vectors), so thresholds tuned on the
  prototype transfer to production.

---

## Evaluation

See **[EVALUATION.md](EVALUATION.md)** for precision/recall against the planted ground
truth, per-scenario results, confusion table, honest failure analysis and trade-offs.
