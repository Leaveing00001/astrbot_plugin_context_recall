"""离线验证 astrbot_plugin_context_recall 的纯逻辑。

不启动 AstrBot、不连 QQ：`core/` 里全是纯函数，直接调用。
覆盖：审查裁决解析 / 工具语法放行 / 重写清洗 / 聊天记录规范化 /
撤回裁决解析 / 现场反应信号 / 配置默认值与兜底。

跑法：
    python tests/selftest.py
环境变量 RECALL_ROOT 可指向插件目录（默认取本文件上一级）。
"""

import io
import json
import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，✅/❌ 会直接抛 UnicodeEncodeError
try:
    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(os.environ.get("RECALL_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT))

from core import judge, models, precheck, settings  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"{'✅' if cond else '❌'} {name}" + (f"  → {detail}" if detail else ""))


# ── 1. 发送前审查：裁决解析 ──────────────────────────────
v_ok = precheck.parse_verdict_line("OK")
check("OK 判为通过", v_ok.ok is True)

v_bad = precheck.parse_verdict_line("BAD|答非所问")
check("BAD|理由 判为不通过", v_bad.ok is False and "答非所问" in v_bad.reason, repr(v_bad.reason))

v_pass = precheck.parse_verdict_line("pass")
check("PASS 也算通过", v_pass.ok is True)

v_garbage = precheck.parse_verdict_line("我觉得还行吧")
check("看不懂的输出一律放行", v_garbage.ok is True, repr(v_garbage))

v_empty = precheck.parse_verdict_line("")
check("空输出放行（不吞消息）", v_empty.ok is True)

v_bold = precheck.parse_verdict_line("**OK**")
check("带 Markdown 的 OK 也认", v_bold.ok is True)

# ── 2. 发送前审查：工具 / 寻址语法放行 ───────────────────
for text, label in [
    ("[At:123456] 你去啊", "艾特标记"),
    ("[CQ:at,qq=123456] 走", "CQ 码"),
    ('{"name": "reply_to", "arguments": {}}', "工具调用 JSON"),
    ("tool_calls", "tool_calls"),
    ("reply_to", "reply_to"),
    ("@123456789 你来", "纯数字提及"),
    ("<tool>call</tool>", "工具 XML"),
]:
    check(f"放行：{label}", precheck.has_tool_markers(text) is True, text[:30])

check("普通聊天不放行", precheck.has_tool_markers("今天吃啥") is False)
check("空字符串不是工具消息", precheck.has_tool_markers("") is False)

# ── 3. 发送前审查：提示词填充与重写清洗 ─────────────────
prompt = precheck.build_judge_prompt(
    precheck.DEFAULT_JUDGE_PROMPT, "群 111", "#1 甲：在吗", "在的"
)
check("审查提示词三个占位符都填了",
      "群 111" in prompt and "#1 甲：在吗" in prompt and "在的" in prompt)

rewrite = precheck.build_rewrite_prompt(
    precheck.DEFAULT_REWRITE_PROMPT, "群 111", "#1 甲：在吗", "答错了", "答非所问"
)
check("重写提示词带上了问题描述", "答非所问" in rewrite)

clean = precheck.clean_rewrite("```\n重写后的消息：我这就去\n```")
check("重写结果去掉代码块和前缀", clean == "我这就去", repr(clean))

clean_q = precheck.clean_rewrite("“在呢”")
check("重写结果去掉引号", clean_q == "在呢", repr(clean_q))

check("空重写返回空串", precheck.clean_rewrite("   ") == "")
check("重写结果会被截断", len(precheck.clean_rewrite("啊" * 500, limit=40)) <= 40)

# ── 4. 聊天记录规范化 ───────────────────────────────────
check("文本段原样保留",
      models.segment_to_text({"type": "text", "data": {"text": "你好"}}) == "你好")
check("艾特段转成 @QQ",
      models.segment_to_text({"type": "at", "data": {"qq": "123456"}}) == "@123456")
check("@全体被忽略",
      models.segment_to_text({"type": "at", "data": {"qq": "all"}}) == "")
check("图片段转成 [图片]",
      models.segment_to_text({"type": "image", "data": {}}) == "[图片]")
check("引用段被丢掉",
      models.segment_to_text({"type": "reply", "data": {"id": "1"}}) == "")

msg = {"message": [
    {"type": "text", "data": {"text": "你好"}},
    {"type": "image", "data": {}},
]}
check("一条消息能挤出纯文本", models.message_to_text(msg) == "你好 [图片]",
      repr(models.message_to_text(msg)))

raw = {"raw_message": "[CQ:at,qq=1] 在吗"}
check("没有 message 段时退回 raw_message 并去掉 CQ 码",
      "CQ" not in models.message_to_text(raw), repr(models.message_to_text(raw)))

check("超长文本会补省略号", models.truncate("啊" * 10, 5).endswith("…"))

lines = models.normalize_messages(
    [
        {"message_id": "3", "time": 300, "sender": {"user_id": "2", "nickname": "乙"},
         "message": [{"type": "text", "data": {"text": "第三句"}}]},
        {"message_id": "1", "time": 100, "sender": {"user_id": "1", "card": "甲"},
         "message": [{"type": "text", "data": {"text": "第一句"}}]},
        {"message_id": "2", "time": 200, "sender": {"user_id": "9"},
         "message": [{"type": "text", "data": {"text": "第二句"}}]},
    ],
    self_id="9",
)
check("按时间排好序", [line.text for line in lines] == ["第一句", "第二句", "第三句"],
      str([line.text for line in lines]))
check("自己的消息被标出来", lines[1].is_self is True and lines[0].is_self is False)
check("没有昵称时退回 QQ 号", lines[1].name == "9", lines[1].name)
check("机器人自称在转写里可辨认", lines[1].speaker() == "我（机器人）")

# ── 5. 撤回裁决解析 ─────────────────────────────────────
vd = judge.parse_verdict('{"verdict": "recall", "confidence": 0.91, "reason": "答非所问"}')
check("JSON recall 被解析", vd.recall is True and vd.confidence > 0.9, str(vd))

vk = judge.parse_verdict('{"verdict": "keep", "confidence": 30, "reason": "正常"}')
check("JSON keep 被解析，百分数换算成小数",
      vk.recall is False and abs(vk.confidence - 0.3) < 1e-6, str(vk))

check("乱码一律按 keep", judge.parse_verdict("我也不知道").recall is False)

hint = judge.parse_verdict("这条该撤回了")
check("没有 JSON 时按中文提示兜底", hint.recall is True and hint.confidence > 0)

check("matches_any 遇到坏正则不抛异常", judge.matches_any("abc", ["(", "abc"]) is True)

# ── 6. 现场反应信号 ─────────────────────────────────────
convo = [
    models.ChatLine("1", "9", "我", 100, "我这就去", is_self=True),
    models.ChatLine("2", "1", "甲", 110, "？"),
]
sig = judge.analyze_reaction(convo, 0)
check("对方回「？」判为困惑", sig.confusion is True and sig.has_reply is True)

convo2 = [
    models.ChatLine("1", "9", "我", 100, "我这就去", is_self=True),
    models.ChatLine("2", "1", "甲", 110, "嗯"),
]
sig2 = judge.analyze_reaction(convo2, 0)
check("对方只回「嗯」判为敷衍", sig2.filler_only is True and sig2.confusion is False)

check("没人接话时没有信号", judge.analyze_reaction(convo[:1], 0).has_reply is False)

transcript = judge.format_transcript(convo, max_chars=40)
check("转写带行号", "#1" in transcript and "#2" in transcript, transcript[:60])

# ── 7. 配置默认值与兜底 ─────────────────────────────────
pc = settings.PrecheckSettings.from_config({})
check("审查默认开启", pc.enable is True and pc.dry_run is False)
check("审查默认重写 1 次", pc.max_attempts == 1)

pc2 = settings.PrecheckSettings.from_config(
    {"precheck": {"enable": False, "max_attempts": 99, "min_chars": 0,
                  "skip_structured": False}}
)
check("审查配置能改", pc2.enable is False and pc2.skip_structured is False)
check("审查次数下限兜底", pc2.max_attempts == 99)
check("最小字数不低于 1", pc2.min_chars == 1, str(pc2.min_chars))

st = settings.Settings.from_config({})
check("撤回额度默认 3 次/小时", st.max_recalls_per_hour == 3)
check("撤回默认需要 0.78 置信度", abs(st.min_confidence - 0.78) < 1e-6)

# ── 汇总 ────────────────────────────────────────────────
failed = [item for item in results if not item[1]]
print(f"\n===== {len(results) - len(failed)}/{len(results)} 通过 =====")
for name, _, detail in failed:
    print(f"  失败: {name} {detail}")
sys.exit(1 if failed else 0)
