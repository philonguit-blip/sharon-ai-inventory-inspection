from __future__ import annotations

import base64
import unittest

from app.security import basic_auth_enabled, basic_auth_matches


class BasicAuthTests(unittest.TestCase):
    def test_requires_both_username_and_password(self):
        self.assertFalse(basic_auth_enabled("", "secret"))
        self.assertFalse(basic_auth_enabled("user", ""))
        self.assertTrue(basic_auth_enabled("user", "secret"))

    def test_accepts_exact_basic_credentials(self):
        encoded = base64.b64encode(b"user:secret").decode("ascii")
        self.assertTrue(basic_auth_matches(f"Basic {encoded}", "user", "secret"))

    def test_rejects_wrong_or_malformed_credentials(self):
        wrong = base64.b64encode(b"user:wrong").decode("ascii")
        self.assertFalse(basic_auth_matches(f"Basic {wrong}", "user", "secret"))
        self.assertFalse(basic_auth_matches("Bearer token", "user", "secret"))
        self.assertFalse(basic_auth_matches("Basic not-base64", "user", "secret"))


if __name__ == "__main__":
    unittest.main()
