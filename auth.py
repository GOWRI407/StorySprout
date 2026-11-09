"""
StorySprout - Login & Accounts
Local, file-based accounts with salted PBKDF2 password hashes and two roles:
  * "learner" - uses user_app.py
  * "creator" - uses creator_app.py (admin / content author)

The first creator account can be created freely (setup mode). After that, creating
another creator account requires the CREATOR_ACCESS_CODE value from your .env file.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_DIR = os.path.join(BASE_DIR, "data", "users")
USERS_FILE = os.path.join(USERS_DIR, "accounts.json")
os.makedirs(USERS_DIR, exist_ok=True)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_ITERATIONS = 200_000


def _load() -> Dict[str, Any]:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: Dict[str, Any]) -> None:
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, USERS_FILE)


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS).hex()


def creator_exists() -> bool:
    return any(u.get("role") == "creator" for u in _load().values())


def register(username: str, password: str, role: str = "learner", access_code: str = "") -> Tuple[bool, str]:
    """Create an account. Returns (ok, message)."""
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, . _ -"
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    if role not in ("learner", "creator"):
        return False, "Unknown role."

    if role == "creator" and creator_exists():
        expected = os.getenv("CREATOR_ACCESS_CODE", "")
        if not expected:
            return False, "Creator sign-up is closed. Set CREATOR_ACCESS_CODE in your .env file to allow more creators."
        if not hmac.compare_digest(access_code or "", expected):
            return False, "Wrong creator access code."

    users = _load()
    if username.lower() in {u.lower() for u in users}:
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "hash": _hash(password, salt),
        "role": role,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _save(users)
    return True, "Account created."


def authenticate(username: str, password: str) -> Optional[Dict[str, str]]:
    """Return {"username", "role"} on success, otherwise None."""
    users = _load()
    key = next((u for u in users if u.lower() == (username or "").strip().lower()), None)
    if key is None:
        return None
    rec = users[key]
    if hmac.compare_digest(_hash(password or "", rec["salt"]), rec["hash"]):
        return {"username": key, "role": rec.get("role", "learner")}
    return None


def list_accounts() -> Dict[str, Dict[str, str]]:
    """Public info only (no hashes)."""
    return {u: {"role": r.get("role", "learner"), "created": r.get("created", "")} for u, r in _load().items()}
