
import hashlib
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Optional

import bcrypt
import jwt

from src.api.middleware.errors import RateLimitError
from src.storage.database import get_db_manager
from src.storage.repositories import ApiKeyRepository, UserRepository

_DEFAULT_JWT_SECRET = "your-secret-key-change-in-production"

# In-memory rate limit tracking: api_key_id -> deque of request timestamps
_rate_limit_windows: dict[str, Deque[float]] = defaultdict(deque)


class AuthService:
    JWT_SECRET = os.getenv("JWT_SECRET", _DEFAULT_JWT_SECRET)
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = 24

    @classmethod
    def _ensure_jwt_secret(cls) -> None:
        if os.getenv("ENV", "development") == "production":
            if not os.getenv("JWT_SECRET") or cls.JWT_SECRET == _DEFAULT_JWT_SECRET:
                raise EnvironmentError(
                    "JWT_SECRET must be set to a secure value in production"
                )

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        if hashed.startswith("$2"):
            return bcrypt.checkpw(password.encode(), hashed.encode())
        # Legacy SHA256 hashes (pre-migration)
        legacy = hashlib.sha256(password.encode()).hexdigest()
        return legacy == hashed

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        timestamp = int(time.time())
        random_part = secrets.token_urlsafe(32)
        plain_key = f"sg_{timestamp}_{random_part}"
        hashed_key = hashlib.sha256(plain_key.encode()).hexdigest()
        prefix = plain_key[:10]
        return plain_key, hashed_key, prefix

    @staticmethod
    def register(email: str, username: str, password: str) -> Dict[str, Any]:
        db = get_db_manager()

        with db.get_session() as session:
            user_repo = UserRepository(session)
            api_key_repo = ApiKeyRepository(session)

            if user_repo.get_by_email(email):
                return {"success": False, "error": "Email already registered"}

            if user_repo.get_by_username(username):
                return {"success": False, "error": "Username already taken"}

            user = user_repo.create(
                email=email,
                username=username,
                password_hash=AuthService.hash_password(password),
                full_name="",
                avatar_url="",
                language="en",
                timezone="UTC",
                role="free",
                status="active",
                email_verified=False,
                email_verified_at=None,
                monthly_request_limit=0,
                monthly_token_limit=0,
                monthly_budget_usd=0.0,
                default_model="finetune",
                default_language="en",
                auto_correct=True,
                show_explanations=True,
                stripe_customer_id="",
                subscription_id="",
                subscription_status="",
                subscription_end_at=None,
            )

            plain_key, hashed_key, prefix = AuthService.generate_api_key()
            api_key_repo.create(
                user_id=user.id,
                key_name="Default API Key",
                key_hash=hashed_key,
                key_prefix=prefix,
                permissions={
                    "spell_check": True,
                    "grammar_check": True,
                    "style_check": True,
                    "batch_process": False,
                },
                rate_limit_per_second=0,
                rate_limit_per_minute=0,
                rate_limit_per_hour=0,
                rate_limit_per_day=0,
                total_requests=0,
                total_tokens=0,
                total_cost_usd=0.0,
                last_used_at=None,
                is_active=True,
                expires_at=None,
                revoked_at=None,
            )

            jwt_token = AuthService.generate_jwt(user.id)

            return {
                "success": True,
                "user_id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role,
                "api_key": plain_key,
                "token": jwt_token,
                "message": "Registration successful",
            }

    @staticmethod
    def login(email: str, password: str) -> Dict[str, Any]:
        db = get_db_manager()

        with db.get_session() as session:
            user_repo = UserRepository(session)

            user = user_repo.get_by_email(email)
            if not user:
                return {"success": False, "error": "Invalid email or password"}

            if not AuthService.verify_password(password, user.password_hash):
                return {"success": False, "error": "Invalid email or password"}

            jwt_token = AuthService.generate_jwt(user.id)

            return {
                "success": True,
                "user_id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role,
                "token": jwt_token,
                "message": "Login successful",
            }

    @staticmethod
    def _check_rate_limit(api_key_id: str, limits: dict[str, int]) -> None:
        now = time.time()
        window = _rate_limit_windows[api_key_id]
        window.append(now)

        # Keep only last 24h of timestamps
        cutoff = now - 86400
        while window and window[0] < cutoff:
            window.popleft()

        per_minute = limits.get("per_minute") or 0
        per_hour = limits.get("per_hour") or 0
        per_day = limits.get("per_day") or 0

        if per_minute > 0:
            minute_count = sum(1 for t in window if t >= now - 60)
            if minute_count > per_minute:
                raise RateLimitError(
                    f"Rate limit exceeded: {per_minute} requests per minute"
                )

        if per_hour > 0:
            hour_count = sum(1 for t in window if t >= now - 3600)
            if hour_count > per_hour:
                raise RateLimitError(
                    f"Rate limit exceeded: {per_hour} requests per hour"
                )

        if per_day > 0:
            day_count = sum(1 for t in window if t >= now - 86400)
            if day_count > per_day:
                raise RateLimitError(
                    f"Rate limit exceeded: {per_day} requests per day"
                )

    @staticmethod
    def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        db = get_db_manager()
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()

        with db.get_session() as session:
            api_key_repo = ApiKeyRepository(session)
            user_repo = UserRepository(session)

            api_key_obj = api_key_repo.get_by(key_hash=hashed_key)
            if not api_key_obj or not api_key_obj.is_active:
                return None

            user = user_repo.get_by_id(api_key_obj.user_id)
            if not user or user.status != "active":
                return None

            limits = {
                "per_minute": api_key_obj.rate_limit_per_minute,
                "per_hour": api_key_obj.rate_limit_per_hour,
                "per_day": api_key_obj.rate_limit_per_day,
            }
            AuthService._check_rate_limit(api_key_obj.id, limits)

            api_key_repo.update(api_key_obj.id, last_used_at=datetime.utcnow())

            return {
                "user_id": user.id,
                "api_key_id": api_key_obj.id,
                "email": user.email,
                "username": user.username,
                "role": user.role,
                "key_name": api_key_obj.key_name,
                "rate_limits": limits,
            }

    @staticmethod
    def generate_jwt(user_id: str) -> str:
        AuthService._ensure_jwt_secret()
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=AuthService.JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, AuthService.JWT_SECRET, algorithm=AuthService.JWT_ALGORITHM)

    @staticmethod
    def verify_jwt(token: str) -> Optional[Dict[str, Any]]:
        AuthService._ensure_jwt_secret()
        try:
            payload = jwt.decode(
                token, AuthService.JWT_SECRET, algorithms=[AuthService.JWT_ALGORITHM]
            )
            return {"user_id": payload["user_id"]}
        except jwt.PyJWTError:
            return None

    @staticmethod
    def change_password(
        user_id: str, old_password: str, new_password: str
    ) -> Dict[str, Any]:
        db = get_db_manager()

        with db.get_session() as session:
            user_repo = UserRepository(session)
            user = user_repo.get_by_id(user_id)

            if not user:
                return {"success": False, "error": "User not found"}

            if not AuthService.verify_password(old_password, user.password_hash):
                return {"success": False, "error": "Invalid current password"}

            user_repo.update(
                user_id, password_hash=AuthService.hash_password(new_password)
            )

            return {"success": True, "message": "Password changed successfully"}

    @staticmethod
    def revoke_api_key(api_key_id: str, user_id: str) -> Dict[str, Any]:
        db = get_db_manager()
        with db.get_session() as session:
            api_key_repo = ApiKeyRepository(session)
            api_key = api_key_repo.get_by_id(api_key_id)
            if not api_key or api_key.user_id != user_id:
                return {"success": False, "error": "API key not found"}
            api_key_repo.update(api_key_id, is_active=False)
            return {"success": True, "message": "API key revoked"}

    @staticmethod
    def list_api_keys(user_id: str) -> Dict[str, Any]:
        db = get_db_manager()
        with db.get_session() as session:
            api_key_repo = ApiKeyRepository(session)
            keys = api_key_repo.get_by_user(user_id)
            return {
                "success": True,
                "api_keys": [
                    {
                        "id": k.id,
                        "name": k.key_name,
                        "prefix": k.key_prefix,
                        "is_active": k.is_active,
                        "created_at": k.created_at.isoformat() if k.created_at else None,
                        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                        "total_requests": k.total_requests,
                    }
                    for k in keys
                ],
            }
