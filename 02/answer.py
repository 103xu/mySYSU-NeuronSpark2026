"""
NS-2026-02 方案 B: 问题类型驱动 + 实体提取
策略：先尝试精确提取，失败才用 LLM
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from normalize import canonicalize

# ── 常量 ──────────────────────────────────────────────
REFUSE_TEXT = "无法根据给定知识库回答"
REFUSE_THRESHOLD = -1.5
MAX_NEW_TOKENS = 20
MODEL_NAME = "./models/Qwen2.5-3B-Instruct"

# ── GPU ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
print(f"Device: {DEVICE} | dtype: {DTYPE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 加载模型 ──────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded.")

# ── Prompt Injection ─────────────────────────────────
_INJECTION_RE = re.compile(
    r"忽略(以上|之前|以下|所有).*|"
    r"你现在是.*|"
    r"忘记.*指令.*|"
    r"ignore\s+(above|previous|all|following).*|"
    r"you\s+are\s+now.*|",
    re.IGNORECASE,
)


def clean_query(query: str) -> str:
    return _INJECTION_RE.sub("", query).strip()


# ── 问题类型识别 ──────────────────────────────────────
_QTYPE_PATTERNS = [
    ("duration", re.compile(r"(多久|多长|多长时间|多少天|几[天个]|多少小时|多少分钟|时间|时长|时限|期限|有效期)")),
    ("location", re.compile(r"(哪里|在哪|何处|什么地方|地点|位置|在哪[里个]|何处|何处办理|交至|交到)")),
    ("quantity", re.compile(r"(多少[钱个本份次]|几[个本份次]|多少钱|费用|价格|押金|罚款|收费)")),
    ("time", re.compile(r"(几点|什么时候|何时|截止|几点开始|几点结束|日期|时间)")),
    ("entity", re.compile(r"(什么|哪个|哪种|什么材料|提供什么|提交什么|什么介质)")),
    ("consequence", re.compile(r"(后果|怎么[办处理罚]|怎么办|如何)")),
]


def classify_question(question: str) -> str:
    for qtype, pat in _QTYPE_PATTERNS:
        if pat.search(question):
            return qtype
    return "general"


# ── 实体提取 ──────────────────────────────────────────

# Duration units: 工作日, 天, 小时, 分钟, 周, 个月, 年
_DUR_UNITS = "(?:个?工作日|[天周]|小时|分钟|个?月|年|日内|天内|小时内|分钟内)"

_DURATION_RE = re.compile(
    r"\d+\s*" + _DUR_UNITS + "|"
    r"\d+[-~至]\d+\s*(?:[天周]|小时)|"
    r"(?:提前|至少提前)\d+\s*" + _DUR_UNITS + "|"
    r"单次.*?时长\S*\d+\s*" + _DUR_UNITS
)

_TIME_RE = re.compile(r"\d{1,2}:\d{2}|\d{1,2}点\d{0,2}分?")

_QUANTITY_RE = re.compile(r"\d+\s*(?:元|[个本份次张条台件])")

_LOC_ENTITY_RE = re.compile(r"\S{2,25}?(?:室|间|厅|楼|中心|站|岗|处|馆|空间)")

_CONSEQUENCE_RE = re.compile(r"后果[是为：:]\s*(\S{2,30}?)(?:[。；，,]|$)")

_LOC_PREFIXES = [
    "办理或使用地点为", "或使用地点为", "使用地点为",
    "办理地点为", "办理地点在", "领取地点为", "归还地点均为",
    "归还地点为", "地点为", "地点在", "位于", "交至", "交到",
    "办理或使用", "或使用",
]


def _extract_location(text: str) -> str | None:
    for kw in _LOC_PREFIXES:
        idx = text.find(kw)
        if idx >= 0:
            after = text[idx + len(kw):].lstrip("：:，,。. ")
            m = _LOC_ENTITY_RE.search(after)
            if m:
                return m.group(0).strip()
    return None


def _extract_consequence(text: str) -> str | None:
    m = _CONSEQUENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_entity(text: str, qtype: str) -> str | None:
    if qtype == "duration":
        m = _DURATION_RE.search(text)
        if m:
            return m.group(0).strip()
    elif qtype == "location":
        return _extract_location(text)
    elif qtype == "time":
        m = _TIME_RE.search(text)
        if m:
            return m.group(0).strip()
    elif qtype == "quantity":
        m = _QUANTITY_RE.search(text)
        if m:
            return m.group(0).strip()
    elif qtype == "consequence":
        return _extract_consequence(text)
    return None


# ── 抽取式答案 ─────────────────────────────────────────
def extractive_answer(question: str, retrieved: list) -> str | None:
    if not retrieved or retrieved[0]["score"] < -0.5:
        return None
    text = retrieved[0]["text"].strip()
    qtype = classify_question(question)
    entity = extract_entity(text, qtype)
    if entity:
        return entity
    if len(text) <= 25:
        return text
    return None


# ── 后处理 ────────────────────────────────────────────
def trim_answer(text: str) -> str:
    for prefix in ["根据资料", "文中提到", "答案是", "回答", "答：", "答:", "答案", "因此", "所以"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip("：:，,。. ")
    return text.strip("。！？…，,.;;：: ")


# ── 答案验证 ──────────────────────────────────────────
def verify_answer(answer: str, retrieved: list) -> bool:
    if answer == REFUSE_TEXT:
        return True
    if len(answer) < 2:
        return True
    for r in retrieved[:2]:
        if answer in r["text"]:
            return True
    ans_set = set(answer)
    for r in retrieved[:2]:
        overlap = len(ans_set & set(r["text"]))
        if overlap >= max(len(ans_set) * 0.4, 4):
            return True
    return False


# ── 拒答 ──────────────────────────────────────────────
def should_refuse(retrieved: list) -> bool:
    if not retrieved:
        return True
    if retrieved[0]["score"] < REFUSE_THRESHOLD:
        return True
    return False


# ── LLM Prompt ───────────────────────────────────────
SYSTEM_PROMPT = f"""你是知识库问答机器人。
规则：
- 只输出答案的关键信息，不要完整句子
- 禁止解释、禁止前缀
- 不知道时输出：{REFUSE_TEXT}

示例：
问：普通图书每次续借多少天？ 答：14天
问：考试便利安排申请材料交至哪里？ 答：学生支持中心
问：笔记本归还截止到几点？ 答：17:30
问：请问英文在读证明办理时间多久？ 答：2个工作日
"""


def build_prompt(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"知识库：\n{context}\n\n问：{question}\n答："},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    r = retrieved[0]
    text = r["text"].replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return f"{r['title']}: {text}"


# ── 主函数 ────────────────────────────────────────────
@torch.no_grad()
def extract_answer(question: str, retrieved: list) -> tuple:
    if should_refuse(retrieved):
        return REFUSE_TEXT, []

    question = clean_query(question)

    # 1. 抽取式优先
    extracted = extractive_answer(question, retrieved)
    if extracted is not None:
        answer = canonicalize(trim_answer(extracted))
        if len(answer) >= 2:
            citations = [retrieved[0]["chunk_id"]]
            if len(retrieved) > 1 and retrieved[1]["score"] > 1.0:
                citations.append(retrieved[1]["chunk_id"])
            return answer, citations[:2]

    # 2. LLM fallback
    context = build_context(retrieved)
    prompt = build_prompt(question, context)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    for sep in ["\n", "。", "；"]:
        idx = answer.find(sep)
        if idx > 2:
            answer = answer[:idx]
            break

    # LLM 可能在答案后拼接了拒答文案，需要清除
    if REFUSE_TEXT in answer and answer != REFUSE_TEXT:
        answer = answer.replace(REFUSE_TEXT, "").strip()

    answer = canonicalize(trim_answer(answer))

    if not answer or answer in ("", "。", "."):
        return REFUSE_TEXT, []

    citations = [retrieved[0]["chunk_id"]]
    if len(retrieved) > 1 and retrieved[1]["score"] > 1.0:
        citations.append(retrieved[1]["chunk_id"])
    citations = citations[:2]

    if not verify_answer(answer, retrieved):
        entity = extract_entity(retrieved[0]["text"].strip(), classify_question(question))
        if entity:
            answer = canonicalize(trim_answer(entity))

    return answer, citations
