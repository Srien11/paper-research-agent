from __future__ import annotations

import asyncio
import unittest

from paper_research_agent.answering.streaming import (
    answer_preview,
    preview_scope,
    project_preview,
)


class PreviewProjectionTests(unittest.TestCase):
    def test_partial_json_only_projects_claim_text(self) -> None:
        self.assertEqual(
            project_preview('{"status":"answered","claims":[{"citation_ids":["E1"],"text":"正文'),
            "正文",
        )
        self.assertEqual(project_preview('{"reasoning":"private","claims":['), "")
        self.assertEqual(project_preview('{"claims":[{"text":"甲"},{"text":"乙'), "甲\n\n乙")

    def test_split_escapes_and_surrogate_pairs_are_wire_safe(self) -> None:
        prefix = '{"claims":[{"text":"hello'
        for suffix in ('\\', '\\u', '\\u4', '\\u4e', '\\u4e2'):
            self.assertEqual(project_preview(prefix + suffix), "hello")
        self.assertEqual(project_preview(prefix + '\\u4e2d'), "hello中")
        self.assertEqual(project_preview(prefix + '\\ud83d'), "hello")
        self.assertEqual(project_preview(prefix + '\\ud83d\\ude00'), "hello😀")
        project_preview(prefix + '\\ud83d"}]}').encode("utf-8")

    def test_insufficient_malformed_and_bounded_drafts(self) -> None:
        self.assertEqual(project_preview('{"status":"insufficient_evidence","claims":[{"text":"x"}]}'), "")
        self.assertEqual(project_preview('{"claims":[],"claims":[{"text":"x"}]}'), "")
        self.assertEqual(project_preview('[' * 100), "")
        self.assertEqual(project_preview('x' * 262145), "")
        self.assertEqual(project_preview('{"conclusion":"结论', synthesis=True), "结论")


class PreviewScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_and_exception_restore_scope(self) -> None:
        seen: list[tuple[str, str]] = []

        async def run(name: str) -> None:
            async def callback(text: str) -> None:
                seen.append((name, text))

            with preview_scope(callback):
                await asyncio.sleep(0)
                active = answer_preview.get()
                assert active is not None
                await active(name)
            self.assertIsNone(answer_preview.get())

        await asyncio.gather(run("one"), run("two"))
        self.assertCountEqual(seen, [("one", "one"), ("two", "two")])
        with self.assertRaises(RuntimeError), preview_scope(run):
            raise RuntimeError("cancelled")
        self.assertIsNone(answer_preview.get())
