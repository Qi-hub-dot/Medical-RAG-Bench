# Engineering Portfolio — Medical-RAG-Bench

> **The Embedding Language Gap in Non-English RAG: A Controlled Crossover Study**
> 46 Documents · 108 QA Pairs · 2x2 Crossover · Wilcoxon p<0.001 · Cross-Domain Replication

---

## About This Portfolio

This document presents Medical-RAG-Bench as evidence of **systematic experimental design, root cause analysis, and scientific communication** in AI system evaluation. Prepared for taught postgraduate applications in Artificial Intelligence and Computer Science.

---

## 1. Project Summary

Medical-RAG-Bench is a systematic investigation into why semantic search fails on non-English technical text in RAG systems. The project evolved through six experimental iterations, each driven by the findings of the previous:

1. **Initial finding**: BM25 significantly outperforms semantic search on Chinese medical literature (P@3 gap: 31.75pp)
2. **Hypothesis formation**: Is the gap caused by domain (medical terminology) or language (English-trained embeddings on Chinese text)?
3. **Controlled experiment**: 2x2 crossover (language x embedding) definitively isolates language mismatch as the root cause
4. **Solution validation**: Switching to a Chinese embedding model (BGE) closes the gap and makes hybrid retrieval the clear winner
5. **Cross-domain replication**: Legal literature confirms the language-mismatch hypothesis
6. **Engineering contribution**: Adaptive RRF weighting as a safeguard against embedding failure

---

## 2. My Role & Contributions

| Responsibility | My Work |
|---------------|---------|
| **Research Design** | Formulated the research question, designed the 2x2 crossover protocol, chose metrics (P@K, MRR, Wilcoxon), designed the adaptive RRF method |
| **Dataset Engineering** | Curated 46 documents across 30+ medical specialties; authored 108 QA pairs; designed the parallel Chinese-English crossover corpus; conducted a 20% spot-check for data quality |
| **Experimental Execution** | Built a reproducible evaluation pipeline; ran 6 iterations of experiments with progressive refinement; integrated BGE embedding model; implemented adaptive RRF weighting |
| **Root Cause Analysis** | Ruled out domain specialization via crossover experiment; identified language mismatch as the dominant factor; quantified the penalty (13-20pp) |
| **Cross-Domain Validation** | Replicated the crossover experiment in the legal domain to test generalizability |
| **Academic Writing** | Authored paper.md following academic conventions with abstract, methods, results, analysis, limitations, and references |

---

## 3. The Scientific Narrative

### Phase 1: Observation

BM25 (keyword search) dramatically outperforms semantic search on Chinese medical text — a 31.75pp gap in P@3. This contradicts the standard RAG design assumption that hybrid retrieval is universally optimal.

### Phase 2: Hypothesis

Two competing explanations:
- **Domain hypothesis**: Medical terminology is too specialized for general-domain embeddings
- **Language hypothesis**: English-trained embeddings cannot represent Chinese text meaningfully

### Phase 3: Controlled Experiment (2x2 Crossover)

Constructed parallel Chinese-English versions of the same medical documents (identical content, translated). Evaluated semantic retrieval across all four cells of the 2x2 matrix.

**Result**: Language mismatch alone accounts for a 13-20pp penalty. The English embedding model performs best on English medical text (53.3%) — if domain were the issue, it would fail there too. The problem is definitively language, not domain.

### Phase 4: Solution

Switched from all-MiniLM-L6-v2 (English) to bge-small-zh-v1.5 (Chinese). Semantic P@3 jumps from 18.55% to 48.41% (+29.86pp). Hybrid retrieval with the Chinese embedding achieves P@3=75.65% — now outperforming BM25 alone by 22pp.

### Phase 5: Generalizability

Replicated the 2x2 crossover in the legal domain. Same pattern: language-matched embeddings outperform language-mismatched ones. The effect size varies by domain (20pp in medicine, 8pp in law) but the direction is consistent.

### Phase 6: Engineering Safeguard

Proposed adaptive RRF weighting: auto-calibrate fusion weights based on per-retriever MRR on a small calibration set. Correctly downweights weak embeddings (0.28 weight for English model on Chinese text). For well-matched embeddings, equal-weight RRF is already near-optimal.

---

## 4. Key Results

### Main Experiment (n=108)

| Method | English Embedding | Chinese Embedding (BGE) |
|--------|:---:|:---:|
| BM25 P@3 | 51.6% | 51.6% |
| Semantic P@3 | 18.6% | 48.4% (+29.9pp) |
| **Hybrid P@3** | 33.0% | **75.7%** (+42.7pp) |

### 2x2 Crossover

| | Chinese Text | English Text |
|---|---|---|
| English Embedding | 33.3% | 53.3% |
| Chinese Embedding | 46.7% | 33.3% |

**Language mismatch penalty: 13-20pp.** Domain is not the cause.

---

## 5. Engineering Competencies

| Competency | Evidence |
|------------|----------|
| **Experimental Design** | 2x2 crossover controlling for language vs. domain; multi-metric evaluation (P@K, MRR, Wilcoxon) |
| **Root Cause Analysis** | Systematically ruled out competing hypotheses; isolated language mismatch as the dominant factor |
| **Statistical Rigor** | Wilcoxon signed-rank test (p<0.001) on n=108; proper effect size quantification |
| **Reproducible Research** | Single-command evaluation; version-controlled dataset; auto-generated results.json |
| **Cross-Domain Thinking** | Replicated findings in legal domain; identified generalizable principle (language > domain) |
| **Methodological Honesty** | Self-identified keyword-matching limitation; quantified impact; proposed document-level annotation |
| **Scientific Communication** | 8-section academic paper with proper structure, citations, and self-critical analysis |
| **Iterative Refinement** | 6 experimental iterations driven by findings, not predetermined plan |

---

## 6. Relevance to AI/CS Graduate Study

This project demonstrates:

- **Systems thinking**: Understanding that RAG performance depends on the entire pipeline, not individual components in isolation
- **Experimental methodology**: Designing controlled experiments to isolate causal factors — a core skill in AI research and engineering
- **Critical evaluation**: Not accepting default assumptions ("hybrid is always better"), but testing them in specific contexts
- **Communication**: Producing a structured academic paper that clearly presents findings, methods, and limitations
- **Practical impact**: The findings have direct implications for anyone building non-English RAG systems

---

## 7. Repository

- **GitHub**: [github.com/Qi-hub-dot/Medical-RAG-Bench](https://github.com/Qi-hub-dot/Medical-RAG-Bench)
- **Paper**: [paper.md](paper.md)
- **Dataset**: [dataset.py](dataset.py) (CC BY 4.0, 46 docs, 108 QA pairs)

---

*Prepared for taught postgraduate applications in Artificial Intelligence and Computer Science, 2026.*
