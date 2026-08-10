from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_research_agent.web.files import AttachmentStore


async def chunks(*values: bytes):
    for value in values:
        yield value


class AttachmentStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_text_creates_a_new_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AttachmentStore(Path(directory))
            item = await store.save_generated_text(
                session_id="session-a",
                filename="edited-notes.md",
                text="新内容",
            )

            self.assertIn("新内容", store.extract("session-a", (item.attachment_id,))[0])
            content = store.read("session-a", item.attachment_id)
            self.assertEqual(content.filename, "edited-notes.md")
            self.assertEqual(content.content_type, "text/markdown")
            self.assertEqual(content.data.decode("utf-8"), "新内容")
            with self.assertRaises(FileNotFoundError):
                store.read("session-b", item.attachment_id)

    async def test_save_extract_and_delete_are_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AttachmentStore(Path(directory))
            item = await store.save(
                session_id="session-a",
                filename="notes.txt",
                content_type="text/plain",
                chunks=chunks("旧内容".encode()),
            )

            self.assertIn("旧内容", store.extract("session-a", (item.attachment_id,))[0])
            with self.assertRaises(FileNotFoundError):
                store.extract("session-b", (item.attachment_id,))
            self.assertTrue(store.delete("session-a", item.attachment_id))
            self.assertFalse(store.delete("session-a", item.attachment_id))

    async def test_rejects_traversal_unsupported_and_oversized_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AttachmentStore(Path(directory), max_file_bytes=4)
            for filename in ("../secret.txt", "script.exe"):
                with self.assertRaises(ValueError):
                    await store.save(
                        session_id="session-a",
                        filename=filename,
                        content_type="application/octet-stream",
                        chunks=chunks(b"abc"),
                    )
            with self.assertRaises(ValueError):
                await store.save(
                    session_id="session-a",
                    filename="large.txt",
                    content_type="text/plain",
                    chunks=chunks(b"12345"),
                )
