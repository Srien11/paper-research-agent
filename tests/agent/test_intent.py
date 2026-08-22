from __future__ import annotations

import unittest

from paper_research_agent.agent.intent import requires_research_planning


class ResearchIntentTests(unittest.TestCase):
    def test_detects_explicit_scholarly_comparisons(self) -> None:
        self.assertTrue(
            requires_research_planning("比较 RAGAS 和 ARES 两篇论文的评测方法与指标")
        )
        self.assertTrue(
            requires_research_planning(
                "Compare the methods and datasets used by these two research papers"
            )
        )
        self.assertTrue(requires_research_planning("这两个模型在实验结果上有什么异同？"))
        self.assertTrue(requires_research_planning("分别说明两篇论文采用的实验指标"))
        self.assertTrue(requires_research_planning("这些模型共同使用了哪些数据集？"))
        self.assertTrue(
            requires_research_planning("What limitation is shared across the two papers?")
        )
        self.assertTrue(
            requires_research_planning(
                "在一项研究中，一篇认为没有外部反馈时无效，另一篇区分找错与改错"
            )
        )
        self.assertTrue(
            requires_research_planning(
                "比较两种论文方法：一种依赖外部反馈，另一种使用自我验证"
            )
        )

    def test_keeps_simple_or_non_research_comparisons_out_of_research_graph(self) -> None:
        self.assertFalse(requires_research_planning("介绍一下 RAG 评测方法"))
        self.assertFalse(requires_research_planning("比较今天和昨天的天气"))
        self.assertFalse(requires_research_planning("你好，帮我写一句欢迎语"))
        self.assertFalse(requires_research_planning("一种茶更香，另一种咖啡更苦"))
        self.assertFalse(requires_research_planning("我不同意这个说法"))


if __name__ == "__main__":
    unittest.main()
