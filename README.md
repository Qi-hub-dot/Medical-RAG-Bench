# Medical-RAG-Bench

<p align="center">
  <strong>The Embedding Language Gap in Non-English RAG Retrieval</strong><br>
  <sub>A controlled 2×2 crossover experiment isolating language mismatch from domain specialization</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/papers-46_docs-blue" alt="Docs">
  <img src="https://img.shields.io/badge/queries-108-orange" alt="Queries">
  <img src="https://img.shields.io/badge/statistical-Wilcoxon_p<0.001-brightgreen" alt="Stats">
  <img src="https://img.shields.io/badge/embedding-BGE--small--zh-red" alt="BGE">
  <img src="https://img.shields.io/badge/hybrid-P@3=75.65%25-purple" alt="Hybrid">
  <img src="https://img.shields.io/badge/license-CC_BY_4.0-lightgrey" alt="License">
  <a href="https://doi.org/10.5281/zenodo.21509728"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21509728.svg" alt="DOI"></a>
</p>

---

## Key Finding

**The language of the embedding model — not the retrieval algorithm, not the domain — is the single largest controllable factor in non-English RAG quality.**

A 2x2 crossover experiment (Chinese vs. English text x Chinese vs. English embedding) shows that language mismatch alone causes a **13-20 percentage point penalty** in retrieval precision, controlling for domain content. Switch the embedding model to match the document language, and hybrid RAG goes from losing to BM25 to **outperforming it by 22 percentage points** (P@3=75.65%).

---

## Results

### Main Experiment (n=108 QA pairs, 46 documents)

**With English embedding (all-MiniLM-L6-v2) — the standard default:**

| Method | P@3 | P@5 | MRR |
|--------|-----:|-----:|-----:|
| BM25 | 51.6% | 38.4% | 0.978 |
| Semantic | 18.6% | 16.9% | 0.393 |
| Hybrid (RRF) | 33.0% | 25.4% | 0.671 |

**With Chinese embedding (bge-small-zh-v1.5) — language-matched:**

| Method | P@3 | P@5 | MRR |
|--------|-----:|-----:|-----:|
| BM25 | 51.6% | 38.4% | 0.978 |
| Semantic | 48.4% | 34.3% | 0.958 |
| **Hybrid (RRF)** | **75.7%** | **56.4%** | **0.977** |

- Switching embedding models: semantic P@3 jumps **+29.86pp** (p < 0.001, Wilcoxon)
- Hybrid with Chinese embedding: **22pp above BM25 alone**
- 53% of semantic queries returned zero relevant results with the English model — only 7% with Chinese

### The Crossover Experiment

A 2x2 controlled experiment with parallel Chinese-English medical documents (identical content, different language):

| Embedding \\ Text | Chinese Medical | English Medical |
|---|---|---|
| English (all-MiniLM) | 33.3% | **53.3%** |
| Chinese (BGE) | **46.7%** | 33.3% |

If the problem were domain (medical terminology), the English model would perform poorly on English medical text — but it achieves its **best** score there (53.3%). The degradation only occurs when the language changes. **Language mismatch, not domain specialization, is the root cause.**

Replicated in the legal domain with the same crossover pattern.

### Adaptive RRF Weighting

A proposed method to auto-calibrate fusion weights based on per-retriever MRR:

| Embedding Model | BM25 Weight | Semantic Weight | Best P@3 |
|-----------------|:-----------:|:---------------:|:--------:|
| English (weak) | 0.72 | 0.28 | 55.2% |
| Chinese (strong) | 0.52 | 0.48 | **75.9%** |

Adaptive weighting correctly downweights weak embeddings but doesn't beat equal-weight RRF when both retrievers are strong. Its primary value is as a **safeguard against catastrophic embedding failure**.

---

## Dataset

- **46 documents** (41 medical + 5 legal), 30+ medical specialties
- **108 annotated QA pairs** with document-level relevance labels
- 11 documents manually curated from clinical guidelines; 30 LLM-generated with manual review
- Statistical power: Wilcoxon signed-rank test, p < 0.001
- License: CC BY 4.0

---

## Quick Start

```bash
cd medical-rag-bench
python evaluate.py
```

Single-command reproduction of all experiments. Requires the RAG-Studio backend:
```bash
git clone https://github.com/Qi-hub-dot/RAG-Studio.git
```

---

## Repository Structure

```
medical-rag-bench/
├── README.md              # Project overview & quick start
├── paper.md               # Full paper (anonymous, submission-ready)
├── paper.tex              # LaTeX version (ACL template applied)
├── references.bib         # BibTeX references
├── dataset.py             # 46 documents + 108 QA pairs
├── evaluate.py            # Reproducible evaluation pipeline
├── plot_crossover.py      # 2x2 crossover heatmap generator
├── results.json           # Quantitative results
├── PORTFOLIO.md           # Technical portfolio for grad applications
└── acl-template/          # Official ACL style files
```

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21509728.svg)](https://doi.org/10.5281/zenodo.21509728)

## Paper

Full paper with methods, 2x2 crossover analysis, and adaptive RRF: [paper.md](paper.md) | [paper.tex](paper.tex)

## License

Dataset: CC BY 4.0 | Code: MIT
