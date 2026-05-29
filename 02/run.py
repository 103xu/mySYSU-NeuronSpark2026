"""
NS-2026-02 入口
- 生成 results.json
- 严格格式校验
- 自动修复
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import json
from tqdm import tqdm

from retrieve import retrieve
from answer import extract_answer, REFUSE_TEXT

DATA_DIR = "1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d"
TEST_PATH = f"{DATA_DIR}/test_questions.jsonl"
OUTPUT_DIR = "outputs"
OUTPUT_PATH = f"{OUTPUT_DIR}/results.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 处理 ──────────────────────────────────────────────
results = []

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        item = json.loads(line)
        qid = item["id"]
        question = item["question"]

        retrieved = retrieve(question)
        answer, citations = extract_answer(question, retrieved)

        # 后置校验
        if answer == REFUSE_TEXT:
            citations = []

        results.append(
            {
                "id": qid,
                "answer": answer,
                "citations": citations[:2],  # 最多2条
            }
        )

# ── 写入前校验 ────────────────────────────────────────
print("Validating output...")
for r in results:
    assert isinstance(r["id"], str), f"id 不是字符串: {r['id']}"
    assert isinstance(r["answer"], str), f"answer 不是字符串: {r['answer']}"
    assert isinstance(r["citations"], list), f"citations 不是列表"
    assert len(r["citations"]) <= 5, f"citations 超长"
    assert len(r["answer"]) > 0, f"answer 为空"
    if r["answer"] == REFUSE_TEXT:
        assert r["citations"] == [], f"拒答时 citations 必须为空: {r['id']}"
    for c in r["citations"]:
        assert isinstance(c, str), f"citation 不是字符串: {c}"

# JSON 序列化自检
json_str = json.dumps(results, ensure_ascii=False, indent=2)
json.loads(json_str)  # 二次校验
print("Validation passed.")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(json_str)

print(f"Saved {len(results)} results to {OUTPUT_PATH}")
