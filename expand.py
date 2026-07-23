"""数据集扩充脚本：用 LLM 生成额外医学文献 + 问答对"""
import json, os, time, sys, re

sys.path.insert(0, r"C:\Users\pc\Desktop\researchmate\backend")
from llm.adapter import LLMAdapter, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 待覆盖的科室（现有 11 篇之外的）
TOPICS = {
    "gastro_001": "幽门螺杆菌感染的诊断与根除治疗",
    "gastro_002": "炎症性肠病的生物制剂治疗进展",
    "pulmo_001": "慢性阻塞性肺疾病的全程管理",
    "pulmo_002": "支气管哮喘的阶梯治疗方案",
    "hema_001": "急性髓系白血病的靶向治疗",
    "hema_002": "免疫性血小板减少症的诊疗规范",
    "rheum_001": "类风湿关节炎的达标治疗策略",
    "rheum_002": "系统性红斑狼疮的免疫抑制治疗",
    "derm_001": "银屑病的生物制剂治疗",
    "derm_002": "特应性皮炎的阶梯治疗",
    "ortho_001": "骨质疏松症的诊断与药物治疗",
    "ortho_002": "腰椎间盘突出症的保守与手术治疗",
    "obst_001": "妊娠期糖尿病的筛查与管理",
    "obst_002": "子痫前期的预测与预防策略",
    "emerge_001": "脓毒症的早期识别与集束化治疗",
    "emerge_002": "急性胰腺炎的严重度评估与治疗",
    "anes_001": "围术期疼痛管理的多模式镇痛策略",
    "radio_001": "肺结节的影像学评估与管理策略",
    "patho_001": "肿瘤分子病理检测的临床意义",
    "geria_001": "老年综合评估在老年共病管理中的应用",
    "urol_001": "良性前列腺增生的药物治疗与手术指征",
    "urol_002": "泌尿系结石的代谢评估与预防",
    "ent_001": "慢性鼻窦炎的药物与手术治疗",
    "ent_002": "突发性耳聋的诊疗策略",
    "rehab_001": "脑卒中后康复治疗的时机与方法",
    "nutri_001": "肠内营养在危重症患者中的应用",
    "pharm_001": "治疗药物监测在抗感染治疗中的应用",
    "micro_001": "碳青霉烯耐药革兰阴性菌的诊治策略",
    "img_001": "冠状动脉CTA在冠心病诊断中的应用",
    "genet_001": "遗传性肿瘤综合征的基因检测与咨询",
}

PROMPT_DOC = """你是一位临床医学专家。请为以下医学主题撰写一段 200-400 字的专业摘要（类似临床指南格式）。

主题：{topic}

要求：
1. 使用标准医学术语（中文）
2. 包含：病因/定义、诊断标准/方法、治疗方案/药物、关键数据（如有）
3. 格式为连贯段落，不要用列表
4. 内容准确专业"""

PROMPT_QA = """基于以下医学文献摘要，生成 3 个医学问答对。

文献：
{content}

要求：
1. 问题为医生或患者可能提出的真实问题
2. 答案来自文献内容，准确简洁
3. 每个问答对配 3-5 个关键词（用于检索评估）

输出严格JSON格式（不要输出其他内容）：
[{{"q": "问题1", "a": "答案1", "keywords": ["词1","词2","词3"]}}, ...]"""

llm = LLMAdapter(provider="openai", model="deepseek-chat", base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)

import asyncio

async def generate_all():
    new_corpus = []
    new_qa = []
    i = 0

    for doc_id, topic in list(TOPICS.items())[:30]:  # 先生成 30 篇
        i += 1
        print(f"[{i}/30] 生成: {topic}")
        
        # 生成文献
        prompt = PROMPT_DOC.format(topic=topic)
        content = await llm.generate(prompt, temperature=0.4, max_tokens=1024)
        content = content.strip()
        
        new_corpus.append({
            "id": doc_id,
            "title": topic,
            "content": content,
        })
        
        # 生成 QA
        time.sleep(1)  # rate limit
        qa_prompt = PROMPT_QA.format(content=content[:1500])
        qa_raw = await llm.generate(qa_prompt, temperature=0.4, max_tokens=1024)
        
        try:
            text = qa_raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            qas = json.loads(text)
            for qa in qas:
                new_qa.append({
                    "q": qa["q"],
                    "relevant_doc_ids": [doc_id],
                    "keywords": qa.get("keywords", []),
                })
        except Exception as e:
            print(f"  [WARN] QA parse failed: {e}")
        
        time.sleep(1)
    
    # 保存
    with open(os.path.join(OUT_DIR, "corpus_expanded.py"), "w", encoding="utf-8") as f:
        f.write("# Auto-generated medical corpus extension\n")
        f.write(f"# {len(new_corpus)} documents\n\n")
        f.write("CORPUS_NEW = ")
        json.dump(new_corpus, f, ensure_ascii=False, indent=2)
        f.write("\n\n")
        f.write("QA_PAIRS_NEW = ")
        json.dump(new_qa, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 生成 {len(new_corpus)} 篇文献, {len(new_qa)} 个问答对")
    print(f"保存在: {os.path.join(OUT_DIR, 'corpus_expanded.py')}")

asyncio.run(generate_all())
