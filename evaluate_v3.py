"""Medical-RAG-Bench v3: 中文 Embedding 对比 + Failure 分析 + 跨领域验证"""
import os, sys, json, time, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, r"C:\Users\pc\Desktop\researchmate\backend")

from database import SessionLocal, init_db
from models import Document, Chunk, gen_uuid
from splitters import split_text
init_db()
db = SessionLocal()

# ── 1. 清理并导入全部数据 ──
from dataset import CORPUS as C1, QA_PAIRS as Q1
from corpus_expanded import CORPUS_NEW, QA_PAIRS_NEW

# 法律领域扩展数据
LEGAL_CORPUS = [
    {"id":"law_001","title":"合同纠纷中的违约责任认定标准","content":"合同违约责任是合同法中的核心制度。违约责任的构成要件包括：存在有效合同、存在违约行为、不存在免责事由。违约行为可分为不履行和不完全履行。不履行包括拒绝履行和履行不能；不完全履行包括迟延履行、瑕疵履行和加害履行。赔偿范围以实际损失为限，包括直接损失和间接损失，但不得超过违约方在订立合同时预见到或应当预见到的损失。违约金与实际损失不一致时，当事人可请求法院予以调整。免责事由包括不可抗力、债权人的过错、法律特别规定和合同特别约定。"},
    {"id":"law_002","title":"知识产权的侵权判定标准","content":"知识产权侵权判定遵循\"接触+实质性相似\"原则。接触是指被告有机会接触到原告的作品或技术方案，实质性相似是指被控侵权内容与原告权利内容构成实质性相同。在专利侵权判定中，需要将被控侵权技术方案与专利权利要求进行逐一比对，适用全面覆盖原则和等同原则。商标侵权的判定以相关公众的混淆可能性为标准。著作权侵权中，\"思想/表达二分法\"是基本准则——著作权只保护表达形式，不保护思想本身。合理使用是重要的侵权抗辩事由，需要考虑使用目的、被使用作品的性质、使用数量和质量、对潜在市场的影响四个因素。"},
    {"id":"law_003","title":"刑事诉讼中的证据规则","content":"我国刑事诉讼实行\"以事实为根据，以法律为准绳\"的原则。证据的审查判断遵循关联性、合法性和客观性三个基本属性。非法证据排除规则是保障人权的重要制度：通过刑讯逼供、暴力威胁等非法方法取得的犯罪嫌疑人供述、证人证言和被害人陈述，应当予以排除。物证、书证的收集不符合法定程序，严重影响司法公正的，应当补正或作出合理解释，不能补正或合理解释的，予以排除。证明标准方面：定罪量刑事实要达到\"证据确实、充分\"和\"排除合理怀疑\"的标准。"},
    {"id":"law_004","title":"公司法中的股东权利保护","content":"股东权利是公司法的核心保护对象。基本股东权利包括：资产收益权（分红权）、参与重大决策权、选择管理者的权利。知情权是股东行使其他权利的基础，包括查阅、复制公司章程、股东会会议记录、董事会决议、监事会决议和财务会计报告的权利。异议股东股份回购请求权是指对股东会决议投反对票的股东，可以请求公司按合理价格收购其股份。股东代表诉讼（派生诉讼）是股东在公司怠于追究董事、监事、高管责任时，以自己的名义为公司利益提起的诉讼。小股东保护机制还包括累积投票制度和表决权回避制度。"},
    {"id":"law_005","title":"行政处罚的合法性审查标准","content":"行政处罚的合法性审查包括主体合法性、程序合法性和内容合法性三个维度。主体合法性要求处罚机关具有法定处罚权限，不得超越管辖范围和处罚种类。程序合法性要求处罚决定作出前必须告知当事人拟处罚的事实、理由和依据，并告知其享有陈述权、申辩权和听证权。对于责令停产停业、吊销许可证和较大数额罚款等严重处罚，当事人要求听证的，行政机关必须组织听证。内容合法性要求处罚决定认定的事实清楚、证据确凿、适用法律正确。\"一事不再罚\"原则是指对当事人的同一个违法行为，不得给予两次以上罚款的行政处罚。"},
    {"id":"law_006","title":"劳动法中的劳动合同解除制度","content":"劳动合同解除分为协商解除、劳动者单方解除和用人单位单方解除三种类型。劳动者提前30日书面通知用人单位，可以解除劳动合同；在试用期内提前3日通知即可。用人单位单方解除受到严格限制，主要包括：过失性解除（劳动者严重违纪、失职、被追究刑事责任等）、非过失性解除（医疗期满不能从事原工作、不胜任工作经培训或调岗仍不胜任、客观情况重大变化致合同无法履行）和经济性裁员。非过失性解除和经济性裁员需支付经济补偿金，标准为每满一年支付一个月工资。违法解除劳动合同的，需按经济补偿标准的二倍支付赔偿金。"},
    {"id":"law_007","title":"民法典中的侵权责任归责原则","content":"《民法典》侵权责任编确立了过错责任、无过错责任和公平责任三大归责原则。过错责任是基本归责原则，以行为人存在过错为前提。过错推定是过错责任的特殊形式——在法定情形下推定有过错，由行为人举证证明自己无过错（如建筑物倒塌致损、医疗损害的部分情形）。无过错责任适用于高度危险作业、环境污染、产品责任、动物致害等领域，不以过错为要件。公平责任是补充性原则，适用于双方均无过错但受害人确有重大损失的情形，由双方分担损失。\"自甘风险\"是新增的抗辩事由——自愿参加具有一定风险的文体活动，因其他参加者的行为受到损害的，不得请求其他参加者承担责任。"},
    {"id":"law_008","title":"电子商务中的消费者权益保护","content":"《电子商务法》对网络交易中的消费者权益作出特别规定。电子商务经营者应当全面、真实、准确、及时地披露商品或服务信息，保障消费者的知情权和选择权。不得以虚构交易、编造用户评价等方式进行虚假或引人误解的商业宣传。消费者享有七日无理由退货的权利（特殊商品除外）。电子商务平台经营者知道或应当知道平台内经营者的商品或服务不符合保障人身、财产安全要求，未采取必要措施的，依法与该平台内经营者承担连带责任。\"大数据杀熟\"被明确禁止——电子商务经营者根据消费者兴趣爱好、消费习惯等特征提供商品或服务的搜索结果的，应当同时提供不针对个人特征的选项。"},
    {"id":"law_009","title":"行政诉讼的受案范围与起诉条件","content":"行政诉讼的受案范围包括：对行政处罚不服、对行政强制措施不服、对行政许可决定不服、对行政确权决定不服、对行政征收征用决定不服、对行政不作为不服、认为行政机关侵犯经营自主权、认为行政机关滥用行政权力排除或限制竞争、认为行政机关违法要求履行义务、认为行政机关未依法支付抚恤金或最低生活保障待遇。不属于受案范围的有：国防外交等国家行为、抽象行政行为、内部行政行为和终局裁决行为。起诉期限为知道或应当知道行政行为之日起六个月内，最长不超过五年（涉及不动产的为二十年）。起诉条件包括：原告适格、有明确的被告、有具体的诉讼请求和事实依据、属于受案范围和受诉法院管辖。"},
    {"id":"law_010","title":"刑法中的正当防卫制度","content":"正当防卫是刑法规定的违法阻却事由。构成要件包括：存在现实的不法侵害、不法侵害正在进行、具有防卫意识、针对不法侵害人本人实施、未明显超过必要限度。\"无限防卫权\"适用于正在进行的行凶、杀人、抢劫、强奸、绑架等严重危及人身安全的暴力犯罪——对此类犯罪进行防卫，造成不法侵害人伤亡的，不负刑事责任。防卫过当应当负刑事责任，但应当减轻或免除处罚。在判断是否明显超过必要限度时，应结合案发的时间、地点、环境、双方力量对比、侵害的紧迫性和危险程度等因素综合评判。对于\"挑衅防卫\"和\"假想防卫\"，不能成立正当防卫。"},
]

LEGAL_QA = [
    {"q":"违约赔偿的范围是什么？","relevant_doc_ids":["law_001"],"keywords":["实际损失","直接损失","间接损失","预见"]},
    {"q":"知识产权侵权判定的基本原则是什么？","relevant_doc_ids":["law_002"],"keywords":["接触","实质性相似","全面覆盖"]},
    {"q":"非法证据排除规则适用于哪些情形？","relevant_doc_ids":["law_003"],"keywords":["刑讯逼供","非法方法","排除"]},
    {"q":"股东享有哪些基本权利？","relevant_doc_ids":["law_004"],"keywords":["资产收益","知情权","代表诉讼"]},
    {"q":"行政处罚的合法性审查包括哪些维度？","relevant_doc_ids":["law_005"],"keywords":["主体合法","程序合法","内容合法"]},
    {"q":"用人单位违法解除劳动合同的赔偿标准是什么？","relevant_doc_ids":["law_006"],"keywords":["经济补偿","二倍","赔偿金"]},
    {"q":"民法典规定了哪几种侵权责任归责原则？","relevant_doc_ids":["law_007"],"keywords":["过错责任","无过错责任","公平责任"]},
    {"q":"电子商务消费者的七日无理由退货权有哪些例外？","relevant_doc_ids":["law_008"],"keywords":["无理由退货","消费者","知情权"]},
    {"q":"行政诉讼的起诉期限是多长？","relevant_doc_ids":["law_009"],"keywords":["六个月","五年","二十年"]},
    {"q":"正当防卫的构成要件有哪些？","relevant_doc_ids":["law_010"],"keywords":["不法侵害","正在进行","必要限度"]},
]

CORPUS = C1 + CORPUS_NEW + LEGAL_CORPUS
QA_PAIRS = Q1 + QA_PAIRS_NEW + LEGAL_QA

print(f"v3 数据集: {len(CORPUS)} 篇文献 (医学{len(C1)+len(CORPUS_NEW)} + 法律{len(LEGAL_CORPUS)}), {len(QA_PAIRS)} 个问答对")

# 清空重建
db.query(Chunk).delete()
db.query(Document).delete()
db.commit()
for doc_data in CORPUS:
    doc = Document(id=doc_data["id"], filename=doc_data["title"]+".txt", title=doc_data["title"], content=doc_data["content"], file_type="txt", word_count=len(doc_data["content"]), paragraph_count=doc_data["content"].count("\n\n")+1)
    db.add(doc)
    chunks = split_text(doc_data["content"], strategy="recursive", chunk_size=512, chunk_overlap=64)
    for i,c in enumerate(chunks):
        db.add(Chunk(id=gen_uuid(), doc_id=doc.id, chunk_index=i, content=c["content"], token_count=c.get("token_count",0), char_start=c.get("char_start",0), char_end=c.get("char_end",0)))
db.commit()

# ── 构建索引 ──
from retriever.bm25_retriever import bm25_retriever
from retriever.embedding_retriever import embedding_retriever
all_chunks = db.query(Chunk, Document.title).join(Document, Chunk.doc_id==Document.id).all()
chunks_data = [{"chunk_id":c.id,"doc_id":c.doc_id,"doc_title":t,"content":c.content} for c,t in all_chunks]
bm25_retriever.index(chunks_data)
try:
    embedding_retriever.index_chunks(chunks_data)
except:
    pass

# ── 实验 1: 通用 Embedding (all-MiniLM-L6-v2) ──
from retriever import search
modes = ["bm25","semantic","hybrid"]
results_general = {m:{"p@3":[],"p@5":[],"mrr":[]} for m in modes}
for item in QA_PAIRS:
    for mode in modes:
        r = search(query=item["q"], mode=mode, top_k=5)
        p3 = sum(1 for x in r[:3] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/3
        p5 = sum(1 for x in r[:5] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/5
        mrr=0
        for rank,x in enumerate(r,1):
            if any(kw.lower() in x["content"].lower() for kw in item["keywords"]):
                mrr=1.0/rank;break
        results_general[mode]["p@3"].append(p3)
        results_general[mode]["p@5"].append(p5)
        results_general[mode]["mrr"].append(mrr)

# ── 实验 2: 中文 Embedding (bge-small-zh-v1.5) ──
print("\n加载中文 Embedding 模型 bge-small-zh-v1.5 ...")
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

bge_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
texts = [c["content"] for c in chunks_data]
bge_embeddings = bge_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
bge_index = faiss.IndexFlatIP(bge_embeddings.shape[1])
bge_index.add(bge_embeddings.astype(np.float32))

results_bge = {m:{"p@3":[],"p@5":[],"mrr":[]} for m in modes}
for item in QA_PAIRS:
    qv = bge_model.encode([item["q"]], normalize_embeddings=True).astype(np.float32)
    
    for mode in modes:
        if mode == "bm25":
            bm_results = bm25_retriever.search(item["q"], top_k=5)
            r = [{"content":x["content"]} for x in bm_results]
        elif mode == "semantic":
            scores, indices = bge_index.search(qv, 5)
            r = [{"content":chunks_data[idx]["content"]} for idx in indices[0] if idx<len(chunks_data)]
        else:  # hybrid
            bm_results = bm25_retriever.search(item["q"], top_k=10)
            scores, indices = bge_index.search(qv, 10)
            sem_results = [{"content":chunks_data[idx]["content"], "chunk_id":chunks_data[idx].get("chunk_id",str(idx))} for idx in indices[0] if idx<len(chunks_data)]
            # Simple RRF
            from retriever.hybrid_retriever import rrf_fusion
            r = rrf_fusion([{"chunk_id":"bm"+str(i),"content":x["content"]} for i,x in enumerate(bm_results)], sem_results, top_k=5)
        
        p3 = sum(1 for x in r[:3] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/3
        p5 = sum(1 for x in r[:5] if any(kw.lower() in x["content"].lower() for kw in item["keywords"]))/5
        mrr=0
        for rank,x in enumerate(r,1):
            if any(kw.lower() in x["content"].lower() for kw in item["keywords"]):
                mrr=1.0/rank;break
        results_bge[mode]["p@3"].append(p3)
        results_bge[mode]["p@5"].append(p5)
        results_bge[mode]["mrr"].append(mrr)

# ── 实验 3: Failure Case 分析 ──
print("\n失败案例分析...")
failures = {"bm25":0,"semantic":0,"hybrid":0,"categories":{"synonym":0,"acronym":0,"cross_domain":0}}
for item in QA_PAIRS:
    r = search(query=item["q"], mode="semantic", top_k=5)
    if not any(any(kw.lower() in x["content"].lower() for kw in item["keywords"]) for x in r[:3]):
        failures["semantic"] += 1
        # 分类
        q = item["q"]
        kws = item["keywords"]
        if any(kw in q for kw in ["替代","另一","称作"]): failures["categories"]["synonym"]+=1
        elif any(len(kw)<=4 and kw.isascii() for kw in kws): failures["categories"]["acronym"]+=1
        else: failures["categories"]["cross_domain"]+=1

# ── 统计检验 ──
from scipy.stats import wilcoxon

print("\n"+"="*60)
print("实验 1: 通用 Embedding (all-MiniLM-L6-v2)")
print("="*60)
for m in modes:
    p3=sum(results_general[m]["p@3"])/len(QA_PAIRS)
    p5=sum(results_general[m]["p@5"])/len(QA_PAIRS)
    mr=sum(results_general[m]["mrr"])/len(QA_PAIRS)
    print(f"  {m:<10}: P@3={p3:.2%}  P@5={p5:.2%}  MRR={mr:.4f}")

print("\n"+"="*60)
print("实验 2: 中文 Embedding (bge-small-zh-v1.5)")
print("="*60)
for m in modes:
    p3=sum(results_bge[m]["p@3"])/len(QA_PAIRS)
    p5=sum(results_bge[m]["p@5"])/len(QA_PAIRS)
    mr=sum(results_bge[m]["mrr"])/len(QA_PAIRS)
    print(f"  {m:<10}: P@3={p3:.2%}  P@5={p5:.2%}  MRR={mr:.4f}")

print("\n"+"="*60)
print("实验 3: Embedding 模型对比 (Semantic P@3)")
print("="*60)
gen_p3 = sum(results_general["semantic"]["p@3"])/len(QA_PAIRS)
bge_p3 = sum(results_bge["semantic"]["p@3"])/len(QA_PAIRS)
print(f"  all-MiniLM-L6-v2 (通用):  {gen_p3:.2%}")
print(f"  bge-small-zh-v1.5 (中文): {bge_p3:.2%}")
try:
    s,p = wilcoxon(results_general["semantic"]["p@3"], results_bge["semantic"]["p@3"])
    print(f"  Wilcoxon: p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")
except: pass

# ── 保存 ──
report = {
    "v3_stats": {"corpus":len(CORPUS),"qa":len(QA_PAIRS)},
    "general_embedding": {m:{"P@3":sum(results_general[m]["p@3"])/len(QA_PAIRS),"P@5":sum(results_general[m]["p@5"])/len(QA_PAIRS),"MRR":sum(results_general[m]["mrr"])/len(QA_PAIRS)} for m in modes},
    "chinese_embedding": {m:{"P@3":sum(results_bge[m]["p@3"])/len(QA_PAIRS),"P@5":sum(results_bge[m]["p@5"])/len(QA_PAIRS),"MRR":sum(results_bge[m]["mrr"])/len(QA_PAIRS)} for m in modes},
    "failures": failures,
}
with open("results_v3.json","w",encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[OK] results_v3.json")
print(f"Failure: 语义检索 P@3 完全失败 {failures['semantic']}/{len(QA_PAIRS)} 个查询")
db.close()
