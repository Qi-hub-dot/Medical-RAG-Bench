# Medical-RAG-Bench

**First systematic evaluation of BM25 vs. semantic retrieval on Chinese medical literature.**

## Key Finding

BM25 alone (P@3=42.2%, MRR=1.000) significantly outperforms semantic search (P@3=22.2%, MRR=0.622) in Chinese medical RAG. **General-domain embedding models fail on technical medical Chinese.**

## Dataset

- 11 Chinese medical literature excerpts (cardiology, neurology, oncology, etc.)
- 15 annotated Q&A pairs with relevance judgments
- Source: clinical guidelines and disease summaries

## Quick Start

```bash
cd medical-rag-bench
python evaluate.py
```

## Results

| Method | P@3 | P@5 | MRR |
|--------|------|------|------|
| BM25 | 42.22% | 26.67% | 1.000 |
| Semantic | 22.22% | 20.00% | 0.622 |
| Hybrid | 37.78% | 25.33% | 0.856 |

## Paper

See [paper.md](paper.md) for the full short paper.
