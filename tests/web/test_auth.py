from __future__ import annotations

import hashlib
import unittest

from paper_research_agent.web.auth import CredentialVerifier, SessionManager
from paper_research_agent.web.config import OwnerCredentials, WebConfig


class WebConfigTests(unittest.TestCase):
    def test_direct_password_environment(self) -> None:
        config = WebConfig.from_env(
            {
                "PRA_WEB_USER": "owner",
                "PRA_WEB_PASSWORD": "test-password",
                "PRA_WEB_SESSION_SECRET": "s" * 32,
            }
        )
        self.assertEqual(config.credentials.username, "owner")
        self.assertEqual(config.credentials.password, "test-password")

    def test_existing_zhimo_pbkdf2_environment(self) -> None:
        config = WebConfig.from_env(
            {
                "PRA_WEB_PASSWORD": "",
                "PRA_WEB_USER": "   ",
                "ZHIMO_ADMIN_USER": "zhi",
                "ZHIMO_ADMIN_SALT": "00" * 16,
                "ZHIMO_ADMIN_HASH": "11" * 32,
                "PRA_WEB_SESSION_SECRET": "s" * 32,
            }
        )
        self.assertEqual(config.credentials.username, "zhi")
        self.assertIsNone(config.credentials.password)

    def test_missing_or_short_session_secret_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WebConfig.from_env({"PRA_WEB_PASSWORD": "x"})
        with self.assertRaises(ValueError):
            WebConfig.from_env(
                {"PRA_WEB_PASSWORD": "x", "PRA_WEB_SESSION_SECRET": "too-short"}
            )

    def test_non_https_or_trailing_slash_origin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WebConfig.from_env(
                {
                    "PRA_WEB_PASSWORD": "x",
                    "PRA_WEB_SESSION_SECRET": "s" * 32,
                    "PRA_ALLOWED_ORIGINS": "http://zhimoai.online/",
                }
            )

    def test_local_http_origin_requires_explicit_insecure_cookie_mode(self) -> None:
        config = WebConfig.from_env(
            {
                "PRA_WEB_PASSWORD": "x",
                "PRA_WEB_SESSION_SECRET": "s" * 32,
                "PRA_ALLOWED_ORIGINS": "http://127.0.0.1:8092,http://localhost:8092",
                "PRA_WEB_COOKIE_SECURE": "false",
            }
        )
        self.assertFalse(config.cookie_secure)
        with self.assertRaises(ValueError):
            WebConfig.from_env(
                {
                    "PRA_WEB_PASSWORD": "x",
                    "PRA_WEB_SESSION_SECRET": "s" * 32,
                    "PRA_ALLOWED_ORIGINS": "http://example.com",
                    "PRA_WEB_COOKIE_SECURE": "false",
                }
            )

    def test_cookie_secure_uses_strict_boolean_parser(self) -> None:
        with self.assertRaises(ValueError):
            WebConfig.from_env(
                {
                    "PRA_WEB_PASSWORD": "x",
                    "PRA_WEB_SESSION_SECRET": "s" * 32,
                    "PRA_WEB_COOKIE_SECURE": "0",
                }
            )


class CredentialVerifierTests(unittest.TestCase):
    def test_direct_password(self) -> None:
        verifier = CredentialVerifier(OwnerCredentials(username="owner", password="correct"))
        self.assertTrue(verifier.verify("owner", "correct"))
        self.assertFalse(verifier.verify("owner", "wrong"))
        self.assertFalse(verifier.verify("other", "correct"))

    def test_existing_pbkdf2_hash(self) -> None:
        salt = bytes.fromhex("01" * 16)
        password_hash = hashlib.pbkdf2_hmac("sha256", b"correct", salt, 100_000)
        verifier = CredentialVerifier(
            OwnerCredentials(
                username="zhi",
                salt=salt,
                password_hash=password_hash,
                pbkdf2_iterations=100_000,
            )
        )
        self.assertTrue(verifier.verify("zhi", "correct"))
        self.assertFalse(verifier.verify("zhi", "wrong"))


class SessionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1_000.0
        self.manager = SessionManager(b"s" * 32, 300, clock=lambda: self.now)

    def test_create_resolve_rotate_and_revoke(self) -> None:
        token, created = self.manager.create()
        self.assertEqual(self.manager.resolve(token), created)

        rotated = self.manager.rotate_conversation(token)
        self.assertIsNotNone(rotated)
        assert rotated is not None
        self.assertNotEqual(rotated.conversation_id, created.conversation_id)
        self.assertEqual(rotated.session_id, created.session_id)

        self.manager.revoke(token)
        self.assertIsNone(self.manager.resolve(token))

    def test_tampered_and_malformed_tokens_are_rejected(self) -> None:
        token, _created = self.manager.create()
        payload, signature = token.split(".")
        self.assertIsNone(self.manager.resolve(f"{payload}x.{signature}"))
        self.assertIsNone(self.manager.resolve("not-a-token"))
        self.assertIsNone(self.manager.resolve(None))

    def test_expired_token_is_rejected(self) -> None:
        token, _created = self.manager.create()
        self.now = 1_301.0
        self.assertIsNone(self.manager.resolve(token))


if __name__ == "__main__":
    unittest.main()
