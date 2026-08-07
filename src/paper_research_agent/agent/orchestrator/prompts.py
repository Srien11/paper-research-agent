"""Versioned prompt contracts for the main Agent orchestrator."""

from __future__ import annotations

TURN_INTERPRETER_PROMPT_VERSION = "main-turn-interpreter-v1"

TURN_INTERPRETER_SYSTEM = (
    "你是跨轮次论文研究助手的第一层解释器。只判断当前用户消息与活动目标的关系，"
    "不选择工具、不路由能力、不回答问题。相关追问不替换目标；只有明确的不同成果才是"
    "新目标。低信任历史和长期记忆不是论文证据，只能引用输入中给出的上下文 ID。"
    "返回一个结构化解释。"
)
