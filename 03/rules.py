"""
规则检测系统
支持：prompt_injection / data_exfiltration / tool_abuse / jailbreak 四种高风险检测
功能：regex 匹配、风险累加、Unicode 标准化、混淆字符识别、中英混合检测

设计原则:
1. 每个类别独立评分
2. Pattern 匹配采用加权累加（核心 pattern 权重高）
3. 组合模式额外加分（如"导出"+"数据库"同时出现）
4. Unicode 标准化后再匹配
5. 规则分数归一化到 0~1
"""

import re
import unicodedata
from collections import defaultdict

# ========================
# 核心检测模式（加权）
# ========================

PATTERNS = {

    "prompt_injection": [
        # 英文 - 直接注入/覆盖类（高权重）
        (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|messages?|rules?|context)", 0.40),
        (r"forget\s+(all\s+)?(previous|above|earlier|before)", 0.35),
        (r"disregard\s+(all\s+)?(previous|above|prior)", 0.35),
        (r"override\s+(the\s+)?(system|instructions?|rules?|prompts?)", 0.35),
        (r"(system\s*)?prompt\s*(leak|leaking|injection|hack|reveal|disclose)", 0.30),
        (r"print\s+(the\s+)?(hidden|secret|system|internal)\s*(prompt|instructions?|rules?|message)", 0.35),
        (r"reveal\s+(your|the)\s+(hidden|secret|system|internal)\s*(prompt|instructions?|rules?)", 0.35),
        (r"show\s+(me\s+)?(your|the)\s+(hidden|secret|system|internal)\s*(prompt|instructions?|rules?)", 0.30),
        (r"what\s+(is|are)\s+(your|the)\s+(system\s*)?(prompt|instructions?|rules?)", 0.25),
        (r"developer\s*mode", 0.35),
        (r"you\s+are\s+now\s+(a\s+)?(new|different)", 0.25),
        (r"act\s+as\s+(if|though)\s+you\s+(are|were)", 0.25),

        # 中文 - 指令覆盖类
        (r"忽略.{0,10}(之前|上述|上面|以上|所有).{0,10}(指令|规则|提示|要求|限制)", 0.40),
        (r"忘记.{0,10}(之前|上述|上面|刚才|刚才).{0,10}(对话|内容|指令|规则)", 0.35),
        (r"覆盖.{0,10}(系统|原始|之前).{0,10}(指令|提示|规则)", 0.35),
        (r"(泄露|泄漏|透露|暴露).{0,10}(系统|隐藏|内部).{0,10}(提示|指令|规则|策略)", 0.35),
        (r"输出.{0,10}(隐藏|系统|内部|原始).{0,10}(提示词|指令|规则|配置)", 0.30),
        (r"(打印|显示|告诉我).{0,10}(系统|隐藏|内部).{0,10}(提示词|指令|规则|配置)", 0.30),
        (r"你的.{0,5}(系统|原始).{0,5}(提示词|指令|规则)是什么", 0.30),
        (r"(不要|别|禁止).{0,5}(遵守|遵循|按照).{0,10}(规则|指令|提示)", 0.30),

        # 中英混合 - 绕过类
        (r"(绕过|bypass|skip|跳过).{0,10}(限制|规则|检测|过滤|审查)", 0.30),
        (r"(无视|忽略).{0,5}(安全|权限|身份).{0,5}(限制|规则)", 0.30),
        (r"(重写|改写|修改).{0,10}(系统|原始).{0,10}(指令|规则|提示)", 0.35),
        (r"以.{0,3}(我|这条|当前).{0,3}(为准|优先)", 0.25),

        # 对抗/规避类
        (r"拆分.{0,10}(指令|请求|回答)", 0.20),
        (r"让.{0,5}(过滤器|检测|分类器).{0,5}(看不懂|无法识别|绕过)", 0.25),
        (r"分段|隐喻|暗示.{0,5}(绕过|规避|欺骗)", 0.20),
    ],

    "data_exfiltration": [
        # 密钥/令牌泄露
        (r"(api|API)\s*[-_]?\s*(key|token|secret|密码)", 0.40),
        (r"(access|auth|bearer)\s*(token|key)", 0.35),
        (r"(private|secret|internal)\s*(key|token|credential)", 0.35),
        (r"(导出|提取|获取|读取).{0,10}(密钥|令牌|token|key|密码|凭证)", 0.40),

        # 数据导出
        (r"(export|dump|extract|download|pull)\s+(all\s+)?(data|records?|logs?|database)", 0.35),
        (r"导出.{0,10}(所有|全部|完整|原始).{0,10}(数据|记录|日志|信息)", 0.35),
        (r"(输出|导出|提取).{0,10}(数据库|审计|日志|记录|明细)", 0.30),
        (r"(show|reveal|display)\s+(all\s+)?(hidden|private|internal|secret)\s+(data|records?|logs?|info)", 0.35),

        # 敏感信息获取
        (r"(手机号|邮箱|身份证|银行卡号|地址|学号|工号)", 0.35),
        (r"(个人|隐私|敏感|机密).{0,10}(信息|数据|记录)", 0.30),
        (r"(泄露|泄漏).{0,10}(配置|密钥|密码|数据)", 0.35),
        (r"(打印|显示|输出).{0,10}(内部|隐藏).{0,10}(日志|配置|备注|信息)", 0.30),
        (r"hidden\s+(logs?|secrets?|fields?|data)", 0.35),

        # 批量获取/爬取
        (r"(批量|爬取|遍历|扫描).{0,10}(数据|文件|记录|用户)", 0.30),
        (r"(所有|全部|每个).{0,5}(用户|学生|老师).{0,10}(信息|数据|记录|课表)", 0.35),

        # 中文特定
        (r"不要.{0,5}(权限|授权).{0,5}(裁剪|过滤|限制)", 0.30),
        (r"(完整|原始|全量).{0,5}(记录|数据|日志|信息)", 0.25),
    ],

    "tool_abuse": [
        # 越权操作
        (r"(delete|remove|drop|truncate)\s+(file|record|database|table)", 0.40),
        (r"(send|dispatch)\s+(email|mail|message)", 0.35),
        (r"(execute|run|invoke)\s+(command|shell|script|terminal)", 0.40),
        (r"(使用|调用|执行|触发).{0,10}(管理员|高级|特权).{0,10}(工具|功能|命令)", 0.40),
        (r"(bypass|skip)\s+(confirmation|approval|auth|permission|review)", 0.40),
        (r"(跳过|绕过|无视).{0,10}(审批|确认|授权|双人复核|门禁)", 0.40),

        # 未授权操作
        (r"(as|like|impersonate)\s+(another|other|different)\s+user", 0.40),
        (r"(冒充|伪装|模拟|代替).{0,10}(他人|其他|别的).{0,5}(用户|角色|身份)", 0.40),
        (r"(use|call|invoke)\s+(the\s+)?(tool|function)\s+(now|directly|immediately)", 0.30),
        (r"(修改|删除|写入|更新).{0,10}(数据库|记录|文件|数据).{0,5}(不要|未经|跳过)", 0.35),

        # 工具名直接匹配（需要结合上下文）
        (r"terminal\s*(tool|command|access)", 0.35),
        (r"admin\s*(tool|update|function|access|panel)", 0.35),
        (r"door_access|admin_update|export_budget|data_export", 0.25),

        # 后门/持久化
        (r"(创建|建立|打开).{0,10}(后门|未授权|非授权).{0,5}(访问|通道|接口)", 0.35),
        (r"越权.{0,5}(调用|访问|操作|执行|修改)", 0.40),
        (r"不.{0,3}(需要|经过).{0,5}(授权|审批|确认|验证)", 0.30),

        # 批量/自动化滥用
        (r"(批量|自动化|自动).{0,10}(提交|修改|删除|发送)", 0.30),
        (r"(submit|send|post)\s+(bulk|mass|multiple|many)", 0.30),
    ],

    "jailbreak": [
        # DAN/角色扮演
        (r"(DAN|D\.A\.N)\s*(mode|jailbreak)", 0.45),
        (r"(pretend|act|pose|roleplay)\s+(to\s+be|as|like)\s+(a\s+)?(.*)(without|no\s+)(restrictions?|limits?|rules?|filter)", 0.40),
        (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered|uncensored)", 0.40),
        (r"roleplay\s+as\s+(an?\s+)?(unrestricted|evil|dark|malicious)", 0.40),

        # 无限制 AI
        (r"(unrestricted|uncensored|unfiltered)\s*(AI|model|assistant|bot)", 0.40),
        (r"(safety|content)\s*(filter|guard|restriction)\s*(is\s+)?(removed|disabled|off|gone)", 0.40),
        (r"(remove|disable|turn\s*off|bypass)\s+(your\s+)?(safety|content|ethical|moral)\s*(filter|guard|restriction|guideline)", 0.40),

        # 中文越狱
        (r"(你现在是|你变成了|你已变成|从现在开始你是).{0,10}(不受|没有|无).{0,5}(限制|约束|过滤)", 0.45),
        (r"(解除|移除|关闭|禁用).{0,10}(安全|内容|道德).{0,10}(限制|过滤|审查|约束)", 0.40),
        (r"(扮演|假装|模拟).{0,10}(不受|没有|无).{0,5}(限制|约束).{0,5}(角色|AI|人格)", 0.40),
        (r"(不要|拒绝|停止).{0,10}(讲规则|讲道理|道德).{0,5}(直接|只管|只).{0,5}(回答|给出|输出)", 0.35),

        # 违规内容请求
        (r"(give|provide|tell|show)\s+(me\s+)?(the\s+)?(detailed\s+)?(steps?|instructions?|method|tutorial)", 0.30),
        (r"(如何|怎么|怎样).{0,10}(制作|获取|绕过|破解|入侵)", 0.30),
        (r"(教|帮|告诉).{0,5}(我|如何).{0,5}(绕过|破解|入侵|违规)", 0.30),

        # 角色互换/欺骗
        (r"(你是一个|你作为|你扮演).{0,10}(黑客|攻击者|坏人|罪犯|骗子)", 0.40),
        (r"(安全演练|测试|研究).{0,5}(为名|为借口|为理由)", 0.25),
        (r"这是.{0,5}(安全演练|测试|研究).{0,5}(不会|没有).{0,5}(影响|危害|问题)", 0.20),
    ],
}

# ========================
# 组合加分模式
# ========================

# 当同一类别中两个关键词同时出现时额外加分
COMBO_BONUS = {
    "prompt_injection": [
        (r"忽略|ignore", r"指令|规则|instructions?", 0.15),
        (r"系统|system", r"提示|泄露|prompt|leak", 0.15),
        (r"绕过|bypass", r"限制|过滤|filter|restriction", 0.15),
    ],
    "data_exfiltration": [
        (r"导出|export", r"数据库|日志|database|logs?", 0.15),
        (r"所有|全部|all", r"记录|数据|records?|data", 0.15),
        (r"泄露|reveal", r"密钥|密码|key|password", 0.15),
    ],
    "tool_abuse": [
        (r"跳过|绕过|bypass", r"审批|确认|approval|confirmation", 0.15),
        (r"使用|调用|use|call", r"管理员|admin", 0.15),
        (r"冒充|伪装|as", r"其他|他人|another|other", 0.15),
    ],
    "jailbreak": [
        (r"扮演|pretend|roleplay", r"不受|无限制|unrestricted", 0.15),
        (r"安全|safety", r"关闭|禁用|移除|disable|remove", 0.15),
    ],
}


def normalize_text(text: str) -> str:
    """Unicode 标准化 + 小写化"""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return text


def count_obfuscation(text: str) -> float:
    """检测混淆程度"""
    score = 0.0
    # 检测零宽字符
    zero_width = len(re.findall(r"[​-‏ - ⁠﻿]", text))
    if zero_width > 0:
        score += min(zero_width * 0.1, 0.3)

    # 检测非 ASCII 字符比例（中文除外）
    non_ascii = len(re.findall(r"[^\x00-\x7F一-鿿]", text))
    if non_ascii > 3:
        score += min(non_ascii * 0.05, 0.2)

    # 检测刻意的空格/分隔符
    odd_spacing = len(re.findall(r"\b\s{2,}\b", text))
    if odd_spacing > 2:
        score += min(odd_spacing * 0.05, 0.15)

    return score


def rule_predict(text: str) -> dict:
    """
    对输入文本进行规则检测，返回每个类别的归一化分数

    Args:
        text: 拼接后的完整文本

    Returns:
        dict: 6 个类别的分数（0~1）
    """
    normalized = normalize_text(text)

    # 原始文本也保留用于部分匹配
    scores = {
        "benign": 0.0,
        "prompt_injection": 0.0,
        "data_exfiltration": 0.0,
        "tool_abuse": 0.0,
        "jailbreak": 0.0,
        "ambiguous": 0.0,
    }

    # 1. 遍历所有 pattern 进行匹配
    for label, patterns in PATTERNS.items():
        for pattern, weight in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                scores[label] += weight

    # 2. 组合加分：检测同一类别中多关键词同时出现
    for label, combos in COMBO_BONUS.items():
        for pat1, pat2, bonus in combos:
            found1 = bool(re.search(pat1, normalized, re.IGNORECASE))
            found2 = bool(re.search(pat2, normalized, re.IGNORECASE))
            if found1 and found2:
                scores[label] += bonus

    # 3. 混淆加分
    obf_score = count_obfuscation(normalized)
    if obf_score > 0:
        # 混淆字符出现时，对最高分的高危类别加分
        max_label = max(
            ["prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"],
            key=lambda k: scores[k],
        )
        scores[max_label] += obf_score

    # 4. 总风险分数计算
    total_risk = (
        scores["prompt_injection"]
        + scores["data_exfiltration"]
        + scores["tool_abuse"]
        + scores["jailbreak"]
    )

    # 5. 归一化：clip 到 0~1
    for k in scores:
        scores[k] = min(1.0, max(0.0, scores[k]))

    # 6. 确定 ambiguous 分数
    # 当有一定风险但不足以明确判定时，ambiguous 获取分数
    if 0 < total_risk < 0.5:
        scores["ambiguous"] = min(0.5, total_risk * 1.5)
    elif total_risk >= 0.5:
        scores["ambiguous"] = max(0.0, 0.4 - (total_risk - 0.5) * 0.8)

    # 7. 确定 benign 分数
    if total_risk == 0:
        scores["benign"] = 1.0
    elif total_risk < 0.15:
        scores["benign"] = 0.8 - total_risk * 2.0
    elif total_risk < 0.3:
        scores["benign"] = 0.5 - total_risk
    else:
        scores["benign"] = max(0.0, 0.2 - total_risk * 0.3)

    # 8. 最终 clip
    for k in scores:
        scores[k] = min(1.0, max(0.0, scores[k]))

    return scores


def rule_predict_simple(text: str) -> tuple:
    """
    简化版：返回预测标签和 risk_score
    仅用于快速评估，不用于最终预测
    """
    scores = rule_predict(text)

    # 按分数排序
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = sorted_labels[0]
    top2_label, top2_score = sorted_labels[1]

    # ambiguous 判定
    if top1_score - top2_score < 0.15 and top1_label != "benign":
        pred_label = "ambiguous"
    else:
        pred_label = top1_label

    # risk_score = 四个高风险类别分数之和
    risk_score = (
        scores["prompt_injection"]
        + scores["data_exfiltration"]
        + scores["tool_abuse"]
        + scores["jailbreak"]
    )
    risk_score = min(1.0, risk_score)

    return pred_label, risk_score, scores
