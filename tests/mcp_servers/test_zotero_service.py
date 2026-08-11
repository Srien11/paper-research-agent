from __future__ import annotations

import json
import unittest

import httpx

from paper_research_agent.mcp_servers.zotero_service import (
    ZoteroLocalService,
    ZoteroServiceError,
)


def _item(key: str = "ABCD1234") -> dict[str, object]:
    return {
        "key": key,
        "library": {"id": 1},
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": "Agent Systems",
            "creators": [{"creatorType": "author", "firstName": "A", "lastName": "Li"}],
            "date": "2025-03-01",
            "DOI": "10.1/example",
            "url": "https://example.org/paper",
            "tags": [{"tag": "agents"}],
            "collections": ["COLL1234"],
        },
    }


def _client(handler: httpx.AsyncBaseTransport | httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://127.0.0.1:23119/api/",
        transport=handler,
        follow_redirects=False,
    )


class ZoteroServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_projects_safe_fields_and_bounds_request(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[_item()], request=request)

        async with _client(httpx.MockTransport(handler)) as client:
            result = await ZoteroLocalService(client=client).search_items(
                query="agent memory", limit=5
            )
        self.assertEqual(result[0]["item_key"], "ABCD1234")
        self.assertEqual(result[0]["title"], "Agent Systems")
        self.assertNotIn("data", result[0])
        self.assertNotIn("library", result[0])
        self.assertEqual(seen[0].url.path, "/api/users/0/items")
        self.assertEqual(seen[0].url.params["limit"], "5")
        self.assertEqual(seen[0].headers["Zotero-API-Version"], "3")

    async def test_get_item_collections_annotations_and_attachment_metadata(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/collections"):
                payload = [
                    {
                        "key": "COLL1234",
                        "data": {"name": "Research", "parentCollection": False},
                    }
                ]
            elif request.url.path.endswith("/children"):
                payload = [
                    {
                        "key": "ANNO1234",
                        "data": {
                            "itemType": "annotation",
                            "parentItem": "ABCD1234",
                            "annotationType": "highlight",
                            "annotationText": "bounded note",
                            "annotationComment": "comment",
                            "annotationPageLabel": "3",
                        },
                    }
                ]
            else:
                payload = _item()
                payload["data"].update(
                    {
                        "itemType": "attachment",
                        "parentItem": "PARN1234",
                        "contentType": "application/pdf",
                        "linkMode": "imported_file",
                        "numPages": 12,
                    }
                )
            return httpx.Response(200, json=payload, request=request)

        async with _client(httpx.MockTransport(handler)) as client:
            service = ZoteroLocalService(client=client)
            item = await service.get_item(item_key="ABCD1234")
            collections = await service.list_collections(limit=5)
            annotations = await service.get_annotations(item_key="ABCD1234", limit=5)
            attachment = await service.get_attachment_metadata(item_key="ABCD1234")
        self.assertEqual(item["item_key"], "ABCD1234")
        self.assertEqual(collections, [{"collection_key": "COLL1234", "name": "Research"}])
        self.assertEqual(annotations[0]["annotation_text"], "bounded note")
        self.assertEqual(attachment["content_type"], "application/pdf")
        self.assertEqual(attachment["page_count"], 12)

    async def test_fulltext_is_bounded_and_reports_index_progress(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"content": "x" * 25_000, "indexedPages": 10, "totalPages": 12},
                request=request,
            )

        async with _client(httpx.MockTransport(handler)) as client:
            result = await ZoteroLocalService(client=client).get_fulltext(
                item_key="ABCD1234"
            )
        self.assertEqual(len(result["content"]), 20_000)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["indexed_pages"], 10)
        self.assertEqual(result["total_pages"], 12)

    async def test_errors_are_stable_and_never_include_response_body(self) -> None:
        cases = (
            (403, "zotero_local_api_disabled"),
            (404, "zotero_not_found"),
            (302, "zotero_redirect_rejected"),
        )
        for status, reason in cases:
            with self.subTest(status=status):
                def handler(request: httpx.Request, status: int = status) -> httpx.Response:
                    return httpx.Response(
                        status,
                        headers={"location": "file:///private/paper.pdf"},
                        text="private response body",
                        request=request,
                    )

                async with _client(httpx.MockTransport(handler)) as client:
                    with self.assertRaisesRegex(ZoteroServiceError, reason) as raised:
                        await ZoteroLocalService(client=client).get_item(item_key="ABCD1234")
                self.assertNotIn("private", str(raised.exception))

    async def test_rejects_non_json_and_oversized_responses(self) -> None:
        payloads = (
            httpx.Response(200, text="not json", headers={"content-type": "text/plain"}),
            httpx.Response(
                200,
                content=json.dumps({"value": "x" * 1_100_000}).encode(),
                headers={"content-length": "1100013", "content-type": "application/json"},
            ),
        )
        for response in payloads:
            with self.subTest(content_type=response.headers.get("content-type")):
                def handler(request: httpx.Request, response: httpx.Response = response) -> httpx.Response:
                    response.request = request
                    return response

                async with _client(httpx.MockTransport(handler)) as client:
                    with self.assertRaises(ZoteroServiceError):
                        await ZoteroLocalService(client=client).get_item(item_key="ABCD1234")

    async def test_rejects_offline_timeout_invalid_inputs_and_non_loopback_url(self) -> None:
        def offline(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("private endpoint detail", request=request)

        async with _client(httpx.MockTransport(offline)) as client:
            service = ZoteroLocalService(client=client)
            with self.assertRaisesRegex(ZoteroServiceError, "zotero_offline"):
                await service.get_item(item_key="ABCD1234")
            with self.assertRaises(ValueError):
                await service.search_items(query="", limit=5)
            with self.assertRaises(ValueError):
                await service.search_items(query="agent", limit=21)
            with self.assertRaises(ValueError):
                await service.get_item(item_key="../../bad")
        with self.assertRaisesRegex(ValueError, "loopback"):
            ZoteroLocalService(base_url="https://api.zotero.org")


if __name__ == "__main__":
    unittest.main()
