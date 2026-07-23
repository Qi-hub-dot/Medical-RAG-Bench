# Medical-RAG-Bench

<p align="center">
  <strong>BM25 vs. Semantic Search on Chinese Medical Literature</strong><br>
  <sub>A systematic evaluation benchmark — with a surprising result</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/task-RAG_retrieval-blue" alt="RAG">
  <img src="https://img.shields.io/badge/domain-Chinese_medical-red" alt="Medical">
  <img src="https://img.shields.io/badge/method-BM25_|_FAISS_|_RRF-purple" alt="Methods">
  <img src="https://img.shields.io/badge/queries-15-green" alt="Queries">
  <img src="https://img.shields.io/badge/license-CC_BY_4.0-lightgrey" alt="License">
</p>

---

## The Finding

**BM25 alone (P@3=42.2%, MRR=1.000) significantly outperforms semantic search (P@3=22.2%, MRR=0.622)** on Chinese medical literature retrieval. General-domain embedding models fail on technical medical Chinese — and hybrid RRF fusion makes things worse, not better.

> This challenges the default assumption in RAG system design that "hybrid retrieval is always better."

---

## Results

| Method | P@3 | P@5 | MRR |
|--------|-----:|-----:|-----:|
| **BM25** | **42.2%** | **26.7%** | **1.000** |
| Semantic (all-MiniLM-L6-v2) | 22.2% | 20.0% | 0.622 |
| Hybrid (BM25 + Semantic + RRF) | 37.8% | 25.3% | 0.856 |

- BM25 achieves perfect MRR (1.000) — the first result is always relevant
- Semantic search trails BM25 by **20 percentage points** in P@3
- RRF fusion of noisy semantic results **dilutes** BM25's strong signal

### Ablation: Chunk Size

| Chunk Size | P@5 |
|:----------:|:---:|
| 256 | 20.0% |
| 512 | 20.0% |
| 1024 | 20.0% |

At this document scale (1-2K chars per doc), chunk size has no measurable impact.

---

## Why BM25 Wins in Medicine

### 1. Medical terminology has near-zero paraphrasing
"心肌梗死" is never colloquially called "心脏病发作" in clinical literature. General-domain embeddings are optimized for paraphrasing everyday language — a capability that provides zero advantage when the target domain has no paraphrases.

### 2. Embedding model domain gap
all-MiniLM-L6-v2 was trained on English web data. Its Chinese tokenization and semantic representations are suboptimal for technical medical text. A domain-specific model (e.g., Chinese medical BERT) could close this gap.

### 3. RRF dilution effect
When semantic results are noisy (relevant docs at rank 3-5 while BM25 finds them at rank 1-2), RRF averaging pulls scores down. Fusion helps only when both retrievers are individually competent.

---

## Dataset

- **11 Chinese medical literature excerpts** spanning 10 specialties (cardiology, neurology, oncology, endocrinology, infectious disease, psychiatry, surgery, pediatrics, ophthalmology, nephrology)
- **15 annotated Q&A pairs** with relevance judgments
- Source: publicly available clinical guidelines and disease summaries
- License: CC BY 4.0

---

## Quick Start

```bash
cd medical-rag-bench
python evaluate.py
```

Requires the RAG-Studio backend on the same machine:
```bash
git clone https://github.com/Qi-hub-dot/RAG-Studio.git
```

Output:
```
Medical-RAG-Bench: 中文医学文献 RAG 评测
============================================================
[1/4] 导入 11 篇医学文献...
[2/4] 构建检索索引...
[3/4] 运行检索实验...
[4/4] 消融实验：chunk_size 对检索精度的影响...

============================================================
实验结果汇总
============================================================
  bm25      : P@3=42.22%  P@5=26.67%  MRR=1.000
  semantic  : P@3=22.22%  P@5=20.00%  MRR=0.622
  hybrid    : P@3=37.78%  P@5=25.33%  MRR=0.856
```

---

## Methodological Honesty

We identified a limitation in our evaluation: **keyword-based relevance matching inflates P@K scores** through cross-topic false positives. For example, "再灌注" appears in both cardiology and neurology documents, causing a stroke document to be counted as "relevant" to a heart attack query.

This means:
- P@5 scores are **inflated** across all methods
- The true performance gap between BM25 and semantic search is likely **larger** than reported
- MRR is unaffected (ranks the first truly relevant result)
- Future work should use document-level relevance labels and nDCG

---

## Implications for RAG System Design

1. **For specialized technical domains**: BM25 should be the primary retriever. Semantic search may be weighted down (0.2 or lower) or disabled.
2. **Embedding model selection is critical**: A general-domain model can actively harm retrieval quality. Domain-specific embeddings are not optional — they are essential.
3. **Evaluate before hybridizing**: The default "BM25 + semantic + RRF" pipeline should be validated per-domain, not assumed optimal.

---

## Repository Structure

```
medical-rag-bench/
├── README.md           # Project overview
├── paper.md            # Full short paper (4-page academic format)
├── dataset.py          # 11 medical documents + 15 Q&A pairs
├── evaluate.py         # Reproducible evaluation pipeline
├── results.json        # Quantitative results (auto-generated)
└── PORTFOLIO.md        # Technical portfolio for graduate applications
```

## Paper

Read the full short paper: [paper.md](paper.md)

## License

Dataset: CC BY 4.0 | Code: MIT
