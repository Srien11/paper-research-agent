from __future__ import annotations

import hashlib
import unittest

from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.retrieval.papers import (
    HybridPaperCandidateRetriever,
    PaperCandidateQuery,
    build_paper_candidate_documents,
)


def _chunk(chunk_id: str, corpus_id: str, text: str, page: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        asset_id=f"asset-{corpus_id}",
        corpus_id=corpus_id,
        element_ids=(f"element-{chunk_id}",),
        page_start=page,
        page_end=page,
        token_start=0,
        token_end=max(1, len(text.split())),
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        config_sha256="a" * 64,
    )


def _card(
    corpus_id: str,
    *,
    title: str,
    abstract: str | None,
    evidence_chunk_ids: tuple[str, ...],
) -> PaperCard:
    return PaperCard(
        card_id=f"card-{corpus_id}",
        asset_id=f"asset-{corpus_id}",
        corpus_id=corpus_id,
        title=title,
        abstract=abstract,
        evidence_chunk_ids=evidence_chunk_ids,
        source_element_ids=(f"element-{corpus_id}",),
        config_sha256="b" * 64,
    )


class KeywordSemanticEncoder:
    def __init__(self):
        self.queries = []

    def encode_documents(self, texts):
        return [self._encode(text) for text in texts]

    def encode_query(self, query):
        self.queries.append(query)
        return self._encode(query)

    @staticmethod
    def _encode(text):
        normalized = text.casefold()
        return [
            1.0 if "semantic" in normalized else 0.1,
            1.0 if "lexical" in normalized else 0.1,
        ]


class PaperCandidateDocumentTests(unittest.TestCase):
    def test_document_preserves_identity_title_and_abstract(self) -> None:
        card = _card(
            "C001",
            title="Lexical Paper",
            abstract="A complete paper-level abstract.",
            evidence_chunk_ids=("c1",),
        )

        documents = build_paper_candidate_documents((card,), ())

        self.assertEqual(documents[0].corpus_id, "C001")
        self.assertEqual(documents[0].title, "Lexical Paper")
        self.assertEqual(documents[0].abstract, "A complete paper-level abstract.")
        self.assertFalse(documents[0].used_fallback)
        self.assertIn("Lexical Paper", documents[0].retrieval_text)
        self.assertIn("A complete paper-level abstract.", documents[0].retrieval_text)

    def test_missing_abstract_uses_only_first_bound_evidence_chunks(self) -> None:
        card = _card(
            "C001",
            title="Paper Without Abstract",
            abstract=None,
            evidence_chunk_ids=("c1", "c2", "c3"),
        )
        chunks = (
            _chunk("c1", "C001", "first introduction fragment", 1),
            _chunk("c2", "C001", "second introduction fragment", 2),
            _chunk("c3", "C001", "late body fragment", 9),
        )

        documents = build_paper_candidate_documents(
            (card,),
            chunks,
            fallback_chunk_limit=2,
        )

        document = documents[0]
        self.assertTrue(document.used_fallback)
        self.assertIsNone(document.abstract)
        self.assertIn("first introduction fragment", document.retrieval_text)
        self.assertIn("second introduction fragment", document.retrieval_text)
        self.assertNotIn("late body fragment", document.retrieval_text)

    def test_missing_abstract_rejects_cross_paper_fallback_chunk(self) -> None:
        card = _card(
            "C001",
            title="Paper Without Abstract",
            abstract=None,
            evidence_chunk_ids=("other",),
        )
        chunks = (_chunk("other", "T001", "wrong paper text", 1),)

        with self.assertRaisesRegex(ValueError, "crosses corpus boundary"):
            build_paper_candidate_documents((card,), chunks)


class HybridPaperCandidateRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_fuses_paper_level_bm25_and_vector_rankings_deterministically(self) -> None:
        cards = (
            _card(
                "C001",
                title="Lexical alpha paper",
                abstract="keyword retrieval",
                evidence_chunk_ids=("c1",),
            ),
            _card(
                "T001",
                title="Semantic beta paper",
                abstract="dense retrieval",
                evidence_chunk_ids=("c2",),
            ),
        )
        documents = build_paper_candidate_documents(cards, ())
        retriever = HybridPaperCandidateRetriever(documents, KeywordSemanticEncoder())

        hits = await retriever.search(
            PaperCandidateQuery(original_query="alpha semantic"),
            top_k=2,
        )

        self.assertEqual([hit.corpus_id for hit in hits], ["C001", "T001"])
        self.assertEqual(hits[0].ranks, {"bm25": 1, "vector": 2})
        self.assertEqual(hits[1].ranks, {"bm25": 2, "vector": 1})
        self.assertEqual(len({hit.corpus_id for hit in hits}), 2)

    async def test_english_paper_candidates_use_only_english_retrieval_view(self) -> None:
        cards = (
            _card(
                "C001",
                title="中文原问题命中 lexical",
                abstract="original route evidence",
                evidence_chunk_ids=("c1",),
            ),
            _card(
                "T001",
                title="English translated semantic",
                abstract="translated route evidence",
                evidence_chunk_ids=("c2",),
            ),
        )
        encoder = KeywordSemanticEncoder()
        retriever = HybridPaperCandidateRetriever(
            build_paper_candidate_documents(cards, ()),
            encoder,
        )
        original = "中文原问题 lexical；保留否定、范围、数字 80%"

        hits = await retriever.search(
            PaperCandidateQuery(
                original_query=original,
                english_query="English translated semantic with negation scope 80%",
            ),
            top_k=1,
        )

        self.assertEqual([hit.corpus_id for hit in hits], ["T001"])
        self.assertEqual(encoder.queries, ["English translated semantic with negation scope 80%"])
        self.assertEqual(set(hits[0].ranks), {"bm25", "vector"})


if __name__ == "__main__":
    unittest.main()
