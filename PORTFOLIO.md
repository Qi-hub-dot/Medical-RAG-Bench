# Engineering Portfolio — Medical-RAG-Bench

> **First Systematic RAG Retrieval Benchmark for Chinese Medical Literature**
> BM25 vs. Semantic Search · 11 Documents · 15 Q&A Pairs · Reproducible Evaluation

---

## About This Portfolio

This document presents Medical-RAG-Bench as evidence of **research methodology, domain-aware system evaluation, and critical thinking** in AI system design. Prepared in support of applications to taught postgraduate programs in Artificial Intelligence and Computer Science.

---

## 1. Project Summary

Medical-RAG-Bench is a systematic evaluation of retrieval methods for Chinese medical literature in RAG systems. It challenges the prevailing assumption that hybrid retrieval (BM25 + semantic search) is universally optimal, demonstrating that **BM25 alone significantly outperforms semantic search** in specialized technical domains with standardized terminology.

The project encompasses:
- A curated dataset of 11 Chinese medical literature excerpts across 10 specialties
- 15 annotated question-answer pairs with relevance judgments
- A reproducible evaluation pipeline comparing BM25, semantic (FAISS), and hybrid (RRF) retrieval
- A short academic paper documenting findings, analysis, and limitations

---

## 2. My Role & Contributions

| Responsibility | My Work |
|---------------|---------|
| **Research Design** | Formulated the core research question: "Does hybrid retrieval universally outperform BM25 in specialized technical domains?" Designed the experimental protocol with 3 retrieval conditions and 3 metrics. |
| **Dataset Curation** | Selected 11 representative medical literature excerpts spanning 10 clinical specialties. Authored 15 domain-appropriate Q&A pairs with keyword relevance annotations. |
| **Evaluation Engineering** | Built a reproducible evaluation pipeline (dataset.py + evaluate.py) that imports documents, builds BM25/FAISS indices, runs retrieval experiments, and generates structured results. |
| **Critical Analysis** | Identified a methodological limitation (keyword-based matching causing cross-topic false positives) and quantified its impact on reported metrics. Proposed improvements (document-level labels, nDCG). |
| **Academic Writing** | Authored paper.md — a 4-page short paper following academic conventions with abstract, introduction, methods, results, analysis, limitations, and references. |

---

## 3. The Core Finding

| Method | P@3 | P@5 | MRR |
|--------|-----:|-----:|-----:|
| **BM25** | **42.2%** | **26.7%** | **1.000** |
| Semantic | 22.2% | 20.0% | 0.622 |
| Hybrid (RRF) | 37.8% | 25.3% | 0.856 |

BM25 outperforms semantic search by 20 percentage points in P@3 and achieves perfect MRR — the first retrieved result is always the correct document.

---

## 4. Root Cause Analysis

Rather than simply reporting results, I investigated **why** BM25 dominates in this domain:

### 4.1 Low Semantic Variance
Medical terminology has near-zero paraphrasing in Chinese clinical literature. "心肌梗死" is never colloquially expressed as "心脏病发作" — the technical register is strictly maintained. General-domain embeddings trained on web text gain no advantage from paraphrase handling when the domain has no paraphrases.

### 4.2 Embedding Model Domain Gap
all-MiniLM-L6-v2 (the standard lightweight embedding model used in most RAG tutorials) was trained on English web data. Its Chinese tokenization is suboptimal, and its semantic space lacks medical domain concepts. The performance gap would likely narrow — or reverse — with a domain-specific embedding model.

### 4.3 RRF Dilution
Reciprocal Rank Fusion averages rank positions across retrievers. When one retriever (BM25) finds the correct document at rank 1-2 while the other (semantic) places it at rank 3-5, RRF produces a worse combined rank than BM25 alone. Fusion amplifies quality only when both sources are individually competent — a condition violated in this domain.

---

## 5. Methodological Honesty

A distinguishing feature of this project is its **self-critical analysis** of evaluation methodology:

**Problem identified**: Keyword-based relevance matching produces cross-topic false positives. For "急性心肌梗死的首选再灌注策略", the ischemic stroke document also mentions "再灌注" (in the context of thrombolysis, not PCI). It is counted as a P@5 "hit" despite being topically incorrect.

**Impact on results**:
- P@5 scores are inflated across all methods
- The true performance gap between BM25 and semantic search is likely **larger** than reported
- MRR is structurally unaffected (measures rank of the first truly relevant result)
- Future work should use document-level relevance labels and report nDCG

This self-critique demonstrates the ability to **evaluate one's own methodology**, a skill essential for graduate-level research and professional engineering.

---

## 6. Engineering Competencies

| Competency | Evidence |
|------------|----------|
| **Experimental Design** | 3-condition comparison (BM25 / Semantic / Hybrid), 3-metric evaluation (P@3, P@5, MRR), chunk-size ablation (256/512/1024) |
| **Domain Analysis** | Identified structural properties of medical Chinese that explain BM25's advantage: low semantic variance, standardized terminology, domain-specific vocabulary |
| **Reproducible Research** | Single-command evaluation (`python evaluate.py`); auto-generated results.json; all data in version-controlled dataset.py |
| **Critical Thinking** | Self-identified keyword-matching limitation; quantified impact on metrics; proposed methodological improvements |
| **Academic Communication** | Wrote paper.md following academic paper conventions (Abstract, Methods, Results, Discussion, References) |
| **Full-Stack Integration** | Evaluation pipeline reuses RAG-Studio's BM25/FAISS/RRF implementation — demonstrating code reuse across projects |

---

## 7. Relevance to AI/CS Graduate Study

This project demonstrates several competencies directly relevant to MSc-level coursework and projects:

- **Evaluating AI systems**: Not just building models, but rigorously measuring their performance under controlled conditions
- **Domain adaptation awareness**: Understanding that general solutions (hybrid RAG) may fail in specialized contexts (medical Chinese)
- **Research methodology**: Hypothesis formulation → experimental design → data collection → analysis → limitation acknowledgment
- **Technical writing**: Producing a structured academic paper with proper citations and self-critical analysis

---

## 8. Repository

- **GitHub**: [github.com/Qi-hub-dot/Medical-RAG-Bench](https://github.com/Qi-hub-dot/Medical-RAG-Bench)
- **Paper**: [paper.md](paper.md)
- **Dataset**: [dataset.py](dataset.py) (CC BY 4.0)
- **Results**: [results.json](results.json)

---

*Prepared for taught postgraduate applications in Artificial Intelligence and Computer Science, 2026.*
