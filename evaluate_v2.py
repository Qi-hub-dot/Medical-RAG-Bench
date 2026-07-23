"""Medical-RAG-Bench 合并数据集 + 增强实验"""
import os, sys, json, time, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, r"C:\Users\pc\Desktop\researchmate\backend")

from dataset import CORPUS as C1, QA_PAIRS as Q1
from corpus_expanded import CORPUS_NEW, QA_PAIRS_NEW

# 合并
CORPUS = C1 + CORPUS_NEW
QA_PAIRS = Q1 + QA_PAIRS_NEW

print(f"合并数据集: {len(CORPUS)} 篇文献, {len(QA_PAIRS)} 个问答对")

# 导入数据库
from database import SessionLocal, init_db
from models import Document, Chunk, gen_uuid
from splitters import split_text
init_db()
db = SessionLocal()

# 清理旧数据
db.query(Chunk).delete()
db.query(Document).delete()
db.commit()

# 导入所有文献
print(f"\n[1/5] 导入 {len(CORPUS)} 篇文献...")
for doc_data in CORPUS:
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
    chunks = split_text(doc_data["content"], strategy="recursive", chunk_size=512, chunk_overlap=64)
    for i, c in enumerate(chunks):
        db.add(Chunk(id=gen_uuid(), doc_id=doc.id, chunk_index=i, content=c["content"], token_count=c.get("token_count",0), char_start=c.get("char_start",0), char_end=c.get("char_end",0)))
db.commit()

# 构建索引
print(f"\n[2/5] 构建索引...")
from retriever.bm25_retriever import bm25_retriever
from retriever.embedding_retriever import embedding_retriever
all_chunks = db.query(Chunk, Document.title).join(Document, Chunk.doc_id == Document.id).all()
chunks_data = [{"chunk_id":c.id,"doc_id":c.doc_id,"doc_title":t,"content":c.content} for c,t in all_chunks]
bm25_retriever.index(chunks_data)
try:
    embedding_retriever.index_chunks(chunks_data)
except:
    print("  [WARN] FAISS 跳过")

# 检索实验
print(f"\n[3/5] 检索实验...")
from retriever import search
modes = ["bm25", "semantic", "hybrid"]
results = {m: {"p@3":[],"p@5":[],"mrr":[]} for m in modes}

for item in QA_PAIRS:
    for mode in modes:
        retrieved = search(query=item["q"], mode=mode, top_k=5)
        hits_3 = sum(1 for r in retrieved[:3] if any(kw.lower() in r["content"].lower() for kw in item["keywords"]))
        hits_5 = sum(1 for r in retrieved[:5] if any(kw.lower() in r["content"].lower() for kw in item["keywords"]))
        results[mode]["p@3"].append(hits_3/3)
        results[mode]["p@5"].append(hits_5/5)
        mrr = 0
        for rank, r in enumerate(retrieved, 1):
            if any(kw.lower() in r["content"].lower() for kw in item["keywords"]):
                mrr = 1.0/rank; break
        results[mode]["mrr"].append(mrr)

# 统计检验
print(f"\n[4/5] Wilcoxon 符号秩检验...")
from scipy.stats import wilcoxon
for metric in ["p@3","p@5","mrr"]:
    for m2 in ["semantic","hybrid"]:
        try:
            stat, p = wilcoxon(results["bm25"][metric], results[m2][metric])
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            print(f"  BM25 vs {m2:<10} {metric}: p={p:.4f} {sig}")
        except:
            print(f"  BM25 vs {m2:<10} {metric}: (tie, cannot compute)")

# 输出
print(f"\n[5/5] 结果汇总")
print(f"{'='*60}")
for mode in modes:
    p3 = sum(results[mode]["p@3"])/len(QA_PAIRS)
    p5 = sum(results[mode]["p@5"])/len(QA_PAIRS)
    mrr = sum(results[mode]["mrr"])/len(QA_PAIRS)
    print(f"  {mode:<10}: P@3={p3:.2%}  P@5={p5:.2%}  MRR={mrr:.4f}")

# 保存
report = {
    "corpus_size": len(CORPUS),
    "qa_pairs": len(QA_PAIRS),
    "results": {m: {"P@3": sum(results[m]["p@3"])/len(QA_PAIRS), "P@5": sum(results[m]["p@5"])/len(QA_PAIRS), "MRR": sum(results[m]["mrr"])/len(QA_PAIRS)} for m in modes},
}
with open("results_v2.json","w",encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[OK] results_v2.json")
db.close()
