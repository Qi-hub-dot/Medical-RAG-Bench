"""Medical-RAG-Bench: 中文医学文献 RAG 评测"""
import os, sys, json, time, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 添加 RAG-Studio 后端路径
sys.path.insert(0, r"C:\Users\pc\Desktop\researchmate\backend")

from dataset import CORPUS, QA_PAIRS
from database import SessionLocal, init_db, Base, engine
from models import Document, Chunk, gen_uuid
from splitters import split_text
from retriever import search
from retriever.bm25_retriever import bm25_retriever
from retriever.embedding_retriever import embedding_retriever

init_db()
db = SessionLocal()

# ── Step 1: 导入语料 ──
print("=" * 60)
print("Medical-RAG-Bench: 中文医学文献 RAG 评测")
print("=" * 60)

print(f"\n[1/4] 导入 {len(CORPUS)} 篇医学文献...")
for doc_data in CORPUS:
    existing = db.query(Document).filter(Document.id == doc_data["id"]).first()
    if existing:
        continue  # 跳过已有

    doc = Document(
        id=doc_data["id"],
        filename=doc_data["title"] + ".txt",
        title=doc_data["title"],
        content=doc_data["content"],
        file_type="txt",
        word_count=len(doc_data["content"]),
        paragraph_count=doc_data["content"].count("\n\n") + 1,
    )
    db.add(doc)

    # 切分
    chunks = split_text(doc_data["content"], strategy="recursive", chunk_size=512, chunk_overlap=64)
    for i, c in enumerate(chunks):
        db.add(Chunk(
            id=gen_uuid(), doc_id=doc.id, chunk_index=i,
            content=c["content"], token_count=c.get("token_count", 0),
            char_start=c.get("char_start", 0), char_end=c.get("char_end", 0),
        ))

db.commit()
print(f"  [OK] {len(CORPUS)} 篇文献已导入并切分")

# ── Step 2: 构建索引 ──
print(f"\n[2/4] 构建检索索引...")
all_chunks = db.query(Chunk, Document.title).join(Document, Chunk.doc_id == Document.id).all()
chunks_data = [{"chunk_id": c.id, "doc_id": c.doc_id, "doc_title": t, "content": c.content} for c, t in all_chunks]
bm25_retriever.index(chunks_data)
try:
    embedding_retriever.index_chunks(chunks_data)
except:
    print("  [WARN] FAISS 索引构建跳过（模型未就绪）")

# ── Step 3: 检索实验 ──
print(f"\n[3/4] 运行检索实验...")
modes = ["bm25", "semantic", "hybrid"]
results = {m: {"p@3": [], "p@5": [], "mrr": []} for m in modes}

for item in QA_PAIRS:
    for mode in modes:
        retrieved = search(query=item["q"], mode=mode, top_k=5)
        
        # Precision
        hits_3 = sum(1 for r in retrieved[:3] if any(kw.lower() in r["content"].lower() for kw in item["keywords"]))
        hits_5 = sum(1 for r in retrieved[:5] if any(kw.lower() in r["content"].lower() for kw in item["keywords"]))
        results[mode]["p@3"].append(hits_3 / 3)
        results[mode]["p@5"].append(hits_5 / 5)
        
        # MRR（第一个相关结果排名的倒数）
        mrr = 0
        for rank, r in enumerate(retrieved, 1):
            if any(kw.lower() in r["content"].lower() for kw in item["keywords"]):
                mrr = 1.0 / rank
                break
        results[mode]["mrr"].append(mrr)

# ── Step 4: 消融实验（chunk_size） ──
print(f"\n[4/4] 消融实验：chunk_size 对检索精度的影响...")
chunk_sizes = [256, 512, 1024]
ablation = {}
for cs in chunk_sizes:
    p5s = []
    for item in QA_PAIRS[:5]:  # 前5个问题
        doc_id = item["relevant_doc_ids"][0]
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            continue
        chunks = split_text(doc.content, strategy="recursive", chunk_size=cs, chunk_overlap=int(cs * 0.15))
        test_data = [{"chunk_id": str(i), "content": c["content"]} for i, c in enumerate(chunks)]
        # 简易 BM25
        from rank_bm25 import BM25Okapi
        import jieba
        corpus_tokens = [list(jieba.cut(c["content"])) for c in test_data]
        bm = BM25Okapi(corpus_tokens)
        scores = bm.get_scores(list(jieba.cut(item["q"])))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        hits = sum(1 for i, s in ranked if any(kw in test_data[i]["content"] for kw in item["keywords"]))
        p5s.append(hits / 5)
    ablation[cs] = sum(p5s) / len(p5s) if p5s else 0
    print(f"  chunk_size={cs:>4}: Avg P@5 = {ablation[cs]:.2%}")

# ── 生成报告 ──
print(f"\n{'='*60}")
print("实验结果汇总")
print(f"{'='*60}")

for mode in modes:
    p3 = sum(results[mode]["p@3"]) / len(QA_PAIRS)
    p5 = sum(results[mode]["p@5"]) / len(QA_PAIRS)
    mrr = sum(results[mode]["mrr"]) / len(QA_PAIRS)
    print(f"  {mode:<10}: P@3={p3:.2%}  P@5={p5:.2%}  MRR={mrr:.3f}")

print(f"\n消融实验：")
for cs in chunk_sizes:
    print(f"  chunk_size={cs}: P@5={ablation[cs]:.2%}")

# 保存结果
report = {
    "corpus_size": len(CORPUS),
    "qa_pairs": len(QA_PAIRS),
    "retrieval_results": {m: {"P@3": sum(results[m]["p@3"])/len(QA_PAIRS), "P@5": sum(results[m]["p@5"])/len(QA_PAIRS), "MRR": sum(results[m]["mrr"])/len(QA_PAIRS)} for m in modes},
    "ablation": ablation,
}

report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[OK] 结果已保存: {report_path}")
db.close()
