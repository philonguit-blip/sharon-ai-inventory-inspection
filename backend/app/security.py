from __future__ import annotations

import base64
import hmac


def basic_auth_enabled(username: str, password: str) -> bool:
    return bool(username and password)


def basic_auth_matches(
    authorization: str | None,
    username: str,
    password: str,
) -> bool:
    if not basic_auth_enabled(username, password) or not authorization:
        return False
    try:
        scheme, encoded = authorization.split(" ", 1)
        if scheme.casefold() != "basic":
            return False
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    expected = f"{username}:{password}"
    return hmac.compare_digest(decoded, expected)
