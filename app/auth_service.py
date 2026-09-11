import os
import json
import uuid
import secrets
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from app.models import UserRegisterRequest, UserLoginRequest, GuestLoginRequest, UserResponse

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
USERS_FILE = os.path.join(DATA_DIR, "users.json")

def _ensure_users_storage():
    """Ensure data directory and users.json exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load users map keyed by lowercase email."""
    _ensure_users_storage()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: Dict[str, Dict[str, Any]]):
    """Save users map to users.json."""
    _ensure_users_storage()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Securely hash a password with a random hexadecimal salt using SHA-256."""
    if not salt:
        salt = secrets.token_hex(16)
    hasher = hashlib.sha256()
    hasher.update((password + salt).encode("utf-8"))
    return hasher.hexdigest(), salt

def register_user(req: UserRegisterRequest) -> UserResponse:
    """Register a new user account with hashed password and initial token."""
    email_key = req.email.strip().lower()
    users = _load_users()

    if email_key in users:
        raise ValueError("An account with this email address already exists.")

    hashed_pw, salt = hash_password(req.password)
    user_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_data = {
        "id": user_id,
        "name": req.name.strip(),
        "email": email_key,
        "password_hash": hashed_pw,
        "salt": salt,
        "risk_profile": req.risk_profile or "Moderate",
        "token": token,
        "is_guest": False,
        "created_at": now_str
    }

    users[email_key] = user_data
    _save_users(users)

    return UserResponse(
        id=user_id,
        name=user_data["name"],
        email=user_data["email"],
        risk_profile=user_data["risk_profile"],
        token=token,
        is_guest=False,
        created_at=now_str
    )

def login_user(req: UserLoginRequest) -> UserResponse:
    """Authenticate user with email and password."""
    email_key = req.email.strip().lower()
    users = _load_users()

    if email_key not in users:
        raise ValueError("Invalid email or password.")

    user = users[email_key]
    hashed_input, _ = hash_password(req.password, user["salt"])

    if hashed_input != user["password_hash"]:
        raise ValueError("Invalid email or password.")

    # Refresh session token
    token = secrets.token_urlsafe(32)
    user["token"] = token
    users[email_key] = user
    _save_users(users)

    return UserResponse(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        risk_profile=user["risk_profile"],
        token=token,
        is_guest=user.get("is_guest", False),
        created_at=user.get("created_at", "")
    )

def login_guest(req: GuestLoginRequest) -> UserResponse:
    """Create an instant guest investor session without permanent credentials."""
    user_id = str(uuid.uuid4())
    guest_num = secrets.token_hex(2).upper()
    guest_name = req.name.strip() if req.name and req.name.strip() else f"Guest Investor #{guest_num}"
    token = secrets.token_urlsafe(32)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return UserResponse(
        id=user_id,
        name=guest_name,
        email=f"guest_{user_id[:8]}@qfinopt.app",
        risk_profile=req.risk_profile or "Moderate",
        token=token,
        is_guest=True,
        created_at=now_str
    )

def get_user_by_token(token: str) -> Optional[UserResponse]:
    """Retrieve user details for a given session token."""
    if not token:
        return None
    users = _load_users()
    for user in users.values():
        if user.get("token") == token:
            return UserResponse(
                id=user["id"],
                name=user["name"],
                email=user["email"],
                risk_profile=user.get("risk_profile", "Moderate"),
                token=token,
                is_guest=user.get("is_guest", False),
                created_at=user.get("created_at", "")
            )
    return None
