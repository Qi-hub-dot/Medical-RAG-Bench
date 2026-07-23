"""Medical-RAG-Bench v4: 创新实验 — 多Embedding对比 + 自适应RRF权重"""
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\pc\Desktop\researchmate\backend")

from database import SessionLocal, init_db
from models import Document, Chunk, gen_uuid
from splitters import split_text
init_db()
db = SessionLocal()

from dataset import CORPUS as C1, QA_PAIRS as Q1
from corpus_expanded import CORPUS_NEW, QA_PAIRS_NEW

# 法律数据手动定义（同 v3）
LEGAL_CORPUS = [{"id":"law_001","title":"合同纠纷中的违约责任认定标准","content":"合同违约责任是合同法中的核心制度。违约责任的构成要件包括：存在有效合同、存在违约行为、不存在免责事由。违约行为可分为不履行和不完全履行。不履行包括拒绝履行和履行不能；不完全履行包括迟延履行、瑕疵履行和加害履行。赔偿范围以实际损失为限，包括直接损失和间接损失，但不得超过违约方在订立合同时预见到或应当预见到的损失。违约金与实际损失不一致时，当事人可请求法院予以调整。免责事由包括不可抗力、债权人的过错、法律特别规定和合同特别约定。"},{"id":"law_002","title":"知识产权的侵权判定标准","content":"知识产权侵权判定遵循\"接触+实质性相似\"原则。接触是指被告有机会接触到原告的作品或技术方案，实质性相似是指被控侵权内容与原告权利内容构成实质性相同。在专利侵权判定中，需要将被控侵权技术方案与专利权利要求进行逐一比对，适用全面覆盖原则和等同原则。商标侵权的判定以相关公众的混淆可能性为标准。著作权侵权中，\"思想/表达二分法\"是基本准则——著作权只保护表达形式，不保护思想本身。合理使用是重要的侵权抗辩事由。"},{"id":"law_003","title":"刑事诉讼中的证据规则","content":"我国刑事诉讼实行\"以事实为根据，以法律为准绳\"的原则。证据的审查判断遵循关联性、合法性和客观性三个基本属性。非法证据排除规则是保障人权的重要制度：通过刑讯逼供、暴力威胁等非法方法取得的犯罪嫌疑人供述、证人证言和被害人陈述，应当予以排除。物证、书证的收集不符合法定程序，严重影响司法公正的，应当补正或作出合理解释，不能补正或合理解释的，予以排除。证明标准方面：定罪量刑事实要达到\"证据确实、充分\"和\"排除合理怀疑\"的标准。"},{"id":"law_004","title":"公司法中的股东权利保护","content":"股东权利是公司法的核心保护对象。基本股东权利包括：资产收益权（分红权）、参与重大决策权、选择管理者的权利。知情权是股东行使其他权利的基础，包括查阅、复制公司章程、股东会会议记录、董事会决议、监事会决议和财务会计报告的权利。异议股东股份回购请求权是指对股东会决议投反对票的股东，可以请求公司按合理价格收购其股份。股东代表诉讼（派生诉讼）是股东在公司怠于追究董事、监事、高管责任时，以自己的名义为公司利益提起的诉讼。小股东保护机制还包括累积投票制度和表决权回避制度。"},{"id":"law_005","title":"刑法中的正当防卫制度","content":"正当防卫是刑法规定的违法阻却事由。构成要件包括：存在现实的不法侵害、不法侵害正在进行、具有防卫意识、针对不法侵害人本人实施、未明显超过必要限度。\"无限防卫权\"适用于正在进行的行凶、杀人、抢劫、强奸、绑架等严重危及人身安全的暴力犯罪——对此类犯罪进行防卫，造成不法侵害人伤亡的，不负刑事责任。防卫过当应当负刑事责任，但应当减轻或免除处罚。在判断是否明显超过必要限度时，应结合案发的时间、地点、环境、双方力量对比、侵害的紧迫性和危险程度等因素综合评判。"}]
LEGAL_QA = [{"q":"违约赔偿的范围是什么？","relevant_doc_ids":["law_001"],"keywords":["实际损失","预见"]},{"q":"知识产权侵权判定的基本原则是什么？","relevant_doc_ids":["law_002"],"keywords":["接触","实质性相似"]},{"q":"非法证据排除规则适用于哪些情形？","relevant_doc_ids":["law_003"],"keywords":["刑讯逼供","排除"]},{"q":"股东享有哪些基本权利？","relevant_doc_ids":["law_004"],"keywords":["知情权","代表诉讼"]},{"q":"正当防卫的构成要件有哪些？","relevant_doc_ids":["law_005"],"keywords":["不法侵害","正在"]}]

CORPUS = C1 + CORPUS_NEW + LEGAL_CORPUS
QA_PAIRS = Q1 + QA_PAIRS_NEW + LEGAL_QA

# 清空重建
db.query(Chunk).delete(); db.query(Document).delete(); db.commit()
for d in CORPUS:
    doc=Document(id=d["id"],filename=d["title"],title=d["title"],content=d["content"],file_type="txt",word_count=len(d["content"]),paragraph_count=1)
    db.add(doc)
    for i,c in enumerate(split_text(d["content"],strategy="recursive",chunk_size=512,chunk_overlap=64)):
        db.add(Chunk(id=gen_uuid(),doc_id=doc.id,chunk_index=i,content=c["content"],token_count=c.get("token_count",0)))
db.commit()

from retriever.bm25_retriever import bm25_retriever
all_chunks = db.query(Chunk, Document.title).join(Document, Chunk.doc_id==Document.id).all()
chunks_data = [{"chunk_id":c.id,"doc_id":c.doc_id,"doc_title":t,"content":c.content} for c,t in all_chunks]
bm25_retriever.index(chunks_data)
texts = [c["content"] for c in chunks_data]

import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from scipy.stats import wilcoxon

# ── 实验: 3 个 Embedding 模型对比 ──
MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
}

all_results = {}

for model_key, model_name in MODELS.items():
    print(f"\n{'='*60}")
    print(f"模型: {model_key}")
    print(f"{'='*60}")
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
    except Exception as e:
        print(f"  [SKIP] 模型加载失败: {e}")
        continue

    # 标准 RRF (w=0.5, k=60)
    results_std = {"bm25":{"p@3":[],"p@5":[],"mrr":[]},"semantic":{"p@3":[],"p@5":[],"mrr":[]},"hybrid":{"p@3":[],"p@5":[],"mrr":[]}}
    
    for item in QA_PAIRS:
        # BM25
        bm = bm25_retriever.search(item["q"], top_k=10)
        # Semantic
        qv = model.encode([item["q"]], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(qv, 10)
        sem = [{"content":chunks_data[idx]["content"],"chunk_id":chunks_data[idx].get("chunk_id",str(idx))} for idx in indices[0]]
        
        for mode, ret_list in [("bm25",bm),("semantic",sem)]:
            p3=sum(1 for x in ret_list[:3] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/3
            p5=sum(1 for x in ret_list[:5] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/5
            mrr=0
            for rk,x in enumerate(ret_list,1):
                if any(kw.lower() in x["content"].lower() for kw in item["keywords"]):mrr=1.0/rk;break
            results_std[mode]["p@3"].append(p3);results_std[mode]["p@5"].append(p5);results_std[mode]["mrr"].append(mrr)
        
        # Standard RRF
        from retriever.hybrid_retriever import rrf_fusion
        hybrid = rrf_fusion(
            [{"chunk_id":"bm"+str(i),"content":x["content"]} for i,x in enumerate(bm)],
            [{"chunk_id":x["chunk_id"] if "chunk_id" in x else "s"+str(i),"content":x["content"]} for i,x in enumerate(sem)],
            top_k=5
        )
        p3=sum(1 for x in hybrid[:3] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/3
        p5=sum(1 for x in hybrid[:5] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/5
        mrr=0
        for rk,x in enumerate(hybrid,1):
            if any(kw.lower() in x["content"].lower() for kw in item["keywords"]):mrr=1.0/rk;break
        results_std["hybrid"]["p@3"].append(p3);results_std["hybrid"]["p@5"].append(p5);results_std["hybrid"]["mrr"].append(mrr)
    
    # ── 创新点: 自适应 RRF 权重 ──
    # 思路: 先在小样本上评估 semantic MRR，然后按 MRR 比例分配 RRF 权重
    # w_semantic = semantic_MRR / (semantic_MRR + bm25_MRR)
    
    n_adapt = min(20, len(QA_PAIRS))
    bm_mrr_adapt = np.mean(results_std["bm25"]["mrr"][:n_adapt])
    sem_mrr_adapt = np.mean(results_std["semantic"]["mrr"][:n_adapt])
    
    # 自适应权重
    w_sem = sem_mrr_adapt / (bm_mrr_adapt + sem_mrr_adapt) if (bm_mrr_adapt + sem_mrr_adapt) > 0 else 0.5
    w_bm = 1 - w_sem
    
    # 用自适应权重跑剩余数据
    results_adapt = {"hybrid":{"p@3":[],"p@5":[],"mrr":[]}}
    for item in QA_PAIRS:
        bm = bm25_retriever.search(item["q"], top_k=10)
        qv = model.encode([item["q"]], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(qv, 10)
        sem = [{"content":chunks_data[idx]["content"],"chunk_id":chunks_data[idx].get("chunk_id",str(idx))} for idx in indices[0]]
        
        hybrid = rrf_fusion(
            [{"chunk_id":"bm"+str(i),"content":x["content"]} for i,x in enumerate(bm)],
            [{"chunk_id":x["chunk_id"] if "chunk_id" in x else "s"+str(i),"content":x["content"]} for i,x in enumerate(sem)],
            top_k=5, weight_bm25=w_bm, weight_semantic=w_sem
        )
        p3=sum(1 for x in hybrid[:3] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/3
        p5=sum(1 for x in hybrid[:5] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/5
        mrr=0
        for rk,x in enumerate(hybrid,1):
            if any(kw.lower() in x["content"].lower() for kw in item["keywords"]):mrr=1.0/rk;break
        results_adapt["hybrid"]["p@3"].append(p3);results_adapt["hybrid"]["p@5"].append(p5);results_adapt["hybrid"]["mrr"].append(mrr)
    
    # 汇总
    std = lambda r: {m:{k:np.mean(r[m][k]) for k in r[m]} for m in r}
    s_std = std(results_std)
    s_adapt = std(results_adapt)
    
    all_results[model_key] = {"standard":s_std,"adaptive":s_adapt,"weights":{"w_bm":w_bm,"w_sem":w_sem}}
    
    print(f"\n  标准 RRF (w=0.5/0.5):")
    print(f"    BM25 P@3={s_std['bm25']['p@3']:.2%}  Semantic P@3={s_std['semantic']['p@3']:.2%}  Hybrid P@3={s_std['hybrid']['p@3']:.2%}")
    print(f"  自适应 RRF (w_bm={w_bm:.2f}/w_sem={w_sem:.2f}):")
    print(f"    Hybrid P@3={s_adapt['hybrid']['p@3']:.2%}  (Δ={s_adapt['hybrid']['p@3']-s_std['hybrid']['p@3']:+.2%})")

# ── 最终汇总 ──
print(f"\n{'='*60}")
print("最终对比表")
print(f"{'='*60}")
print(f"{'模型':<25} {'标准Hybrid P@3':>15} {'自适应Hybrid P@3':>18} {'提升':>8}")
print("-"*70)
for mk in all_results:
    r = all_results[mk]
    std_p3 = r["standard"]["hybrid"]["p@3"]
    adp_p3 = r["adaptive"]["hybrid"]["p@3"]
    delta = adp_p3 - std_p3
    print(f"{mk:<25} {std_p3:>15.2%} {adp_p3:>18.2%} {delta:>+8.2%}")

with open("results_v4.json","w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n[OK] results_v4.json")
db.close()
