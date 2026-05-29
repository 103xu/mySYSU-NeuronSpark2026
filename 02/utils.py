"""
NS-2026-02 Prompt Injection 防御 + 额外清洗
"""
import re

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"忽略(以上|之前|以下|所有).*",
        r"你现在是.*",
        r"忘记.*指令.*",
        r"ignore\s+(above|previous|all|following).*",
        r"you\s+are\s+now.*",
        r"forget\s+.*instructions.*",
        r"请?扮演.*角色.*",
        r"不要(遵循|遵守|按照).*指令.*",
        r"system\s*prompt\s*:.*",
        r"请?输出.*完整.*prompt.*",
    ]
]


def clean_query(q: str) -> str:
    for pat in _INJECTION_PATTERNS:
        q = pat.sub("", q)
    return q.strip()
