"""
jwt_manager.py
==============

Concrete :class:`JWTManager` implementation of :class:`TokenManager`
backed by `PyJWT <https://pyjwt.readthedocs.io/>`_.

Supports HS256 (symmetric), RS256 (RSA 2048+) and ES256 (ECDSA P-256)
signing algorithms. Implements claim validation (``exp``, ``nbf``,
``iat``, ``iss``, ``aud``), an in-memory blacklist for revocation, and
refresh-token rotation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Set

import jwt
from jwt import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidTokenError,
)

from .config import JWTConfig
from .constants import (
    AuthEvent,
    AuthStatus,
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
)
from .token_manager import TokenClaims, TokenIssueResult, TokenManager
from .utils import generate_token_id, hash_token

logger = logging.getLogger(__name__)


class JWTManager(TokenManager):
    """JWT (RFC 7519) token manager.

    Parameters
    ----------
    config:
        :class:`JWTConfig` instance carrying keys, algorithm, TTLs and
        claim defaults.
    """

    def __init__(self, config: JWTConfig) -> None:
        super().__init__(issuer=config.issuer, audience=config.audience)
        self.config = config
        if config.algorithm not in {"HS256", "RS256", "ES256"}:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        if config.algorithm == "HS256" and not config.secret_key:
            raise ValueError("HS256 requires a secret_key")
        if config.algorithm in {"RS256", "ES256"} and not (
            config.private_key and config.public_key
        ):
            raise ValueError(f"{config.algorithm} requires private_key and public_key")
        # blacklist keyed by token hash (so we never store raw tokens)
        self._blacklist: Set[str] = set()
        # refresh-token registry: subject -> set of refresh-token hashes
        self._refresh_registry: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Encoding / decoding primitives
    # ------------------------------------------------------------------
    def _signing_key(self) -> Any:
        if self.config.algorithm == "HS256":
            return self.config.secret_key
        return self.config.private_key

    def _verifying_key(self) -> Any:
        if self.config.algorithm == "HS256":
            return self.config.secret_key
        return self.config.public_key

    def encode(
        self,
        claims: Dict[str, Any],
        algorithm: Optional[str] = None,
    ) -> str:
        """Sign ``claims`` and return the encoded JWT string."""
        algo = algorithm or self.config.algorithm
        return jwt.encode(claims, self._signing_key(), algorithm=algo)

    def decode(self, token: str) -> Dict[str, Any]:
        """Verify signature & standard claims and return the payload.

        Raises :class:`jwt.InvalidTokenError` (or a subclass) on any
        validation failure.
        """
        return jwt.decode(
            token,
            self._verifying_key(),
            algorithms=[self.config.algorithm],
            issuer=self.issuer,
            audience=self.audience,
            leeway=self.config.leeway_seconds,
            options={"require": ["exp", "iat", "nbf", "iss", "aud", "sub", "jti"]},
        )

    def verify_signature(self, token: str) -> bool:
        """Return ``True`` iff the JWT signature is cryptographically valid."""
        try:
            self.decode(token)
            return True
        except InvalidTokenError as exc:
            logger.debug("JWT signature verification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # TokenManager implementation
    # ------------------------------------------------------------------
    def issue_token(
        self,
        subject: str,
        *,
        scopes: Optional[Set[str]] = None,
        ttl_seconds: int = 900,
        custom_claims: Optional[Dict[str, Any]] = None,
        token_type: str = TOKEN_TYPE_ACCESS,
    ) -> TokenIssueResult:
        now = int(time.time())
        jti = generate_token_id()
        claims: Dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": subject,
            "iat": now,
            "nbf": now,
            "exp": now + int(ttl_seconds),
            "jti": jti,
            "type": token_type,
            "scopes": sorted(scopes) if scopes else [],
        }
        if custom_claims:
            for k, v in custom_claims.items():
                if k in claims:
                    logger.warning("Refusing to override reserved claim %s", k)
                    continue
                claims[k] = v

        access_token = self.encode(claims)
        refresh_token: Optional[str] = None
        if token_type == TOKEN_TYPE_ACCESS:
            refresh_claims = dict(claims)
            refresh_claims["jti"] = generate_token_id()
            refresh_claims["type"] = TOKEN_TYPE_REFRESH
            refresh_claims["exp"] = now + int(self.config.refresh_token_ttl.total_seconds())
            refresh_token = self.encode(refresh_claims)
            self._refresh_registry.setdefault(subject, set()).add(
                hash_token(refresh_token)
            )

        token_claims = TokenClaims(
            subject=subject,
            issuer=self.issuer,
            audience=self.audience,
            issued_at=now,
            expires_at=now + int(ttl_seconds),
            not_before=now,
            token_id=jti,
            token_type=token_type,
            scopes=set(scopes or []),
            custom=dict(custom_claims or {}),
        )
        return TokenIssueResult(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=int(ttl_seconds),
            claims=token_claims,
        )

    def validate_token(self, token: str) -> Optional[TokenClaims]:
        try:
            payload = self.decode(token)
        except ExpiredSignatureError:
            logger.info("JWT expired")
            return None
        except ImmatureSignatureError:
            logger.info("JWT nbf in future")
            return None
        except (InvalidIssuerError, InvalidAudienceError) as exc:
            logger.info("JWT issuer/audience mismatch: %s", exc)
            return None
        except DecodeError:
            logger.info("JWT could not be decoded")
            return None
        except InvalidTokenError as exc:
            logger.info("JWT invalid: %s", exc)
            return None

        jti = payload.get("jti", "")
        if self.config.blacklist_enabled and hash_token(token) in self._blacklist:
            logger.info("JWT blacklisted jti=%s", jti)
            return None
        if jti and self.is_revoked(jti):
            return None

        return TokenClaims(
            subject=payload["sub"],
            issuer=payload["iss"],
            audience=payload["aud"],
            issued_at=int(payload["iat"]),
            expires_at=int(payload["exp"]),
            not_before=int(payload["nbf"]),
            token_id=jti,
            token_type=payload.get("type", TOKEN_TYPE_ACCESS),
            scopes=set(payload.get("scopes", [])),
            custom={k: v for k, v in payload.items()
                    if k not in {"iss", "aud", "sub", "iat", "nbf", "exp", "jti", "type", "scopes"}},
        )

    # ------------------------------------------------------------------
    # Refresh / blacklist
    # ------------------------------------------------------------------
    def refresh(self, refresh_token: str) -> Optional[TokenIssueResult]:
        """Rotate a refresh token, returning a fresh access+refresh pair."""
        claims = self.validate_token(refresh_token)
        if claims is None or claims.token_type != TOKEN_TYPE_REFRESH:
            logger.info("Refresh token rejected")
            return None
        subject = claims.subject
        token_hash = hash_token(refresh_token)
        registered = self._refresh_registry.get(subject, set())
        if token_hash not in registered:
            logger.warning("Refresh token not in registry (re-use detected?)")
            return None
        # rotate: invalidate old refresh token, issue new pair
        registered.discard(token_hash)
        self._blacklist.add(token_hash)
        return self.issue_token(
            subject,
            scopes=claims.scopes,
            ttl_seconds=int(self.config.access_token_ttl.total_seconds()),
            custom_claims=claims.custom or None,
        )

    def refresh_token(self, refresh_token: str) -> Optional[TokenIssueResult]:
        return self.refresh(refresh_token)

    def blacklist_token(self, token: str) -> bool:
        """Add ``token`` (by hash) to the in-memory blacklist."""
        token_hash = hash_token(token)
        if token_hash in self._blacklist:
            return False
        self._blacklist.add(token_hash)
        logger.info("Blacklisted JWT (hash=%s…)", token_hash[:12])
        return True

    def revoke_token(self, token: str) -> bool:
        """Both blacklist (by hash) and revoke (by jti) for defence in depth.

        We deliberately decode the token *first* (before blacklisting) so
        that the subsequent validate_token call inside the parent class
        does not see the just-blacklisted token and reject it as already
        invalid — that would mask the success of the revocation itself.
        """
        try:
            payload = self.decode(token)
        except InvalidTokenError as exc:
            logger.info("revoke_token: cannot decode token: %s", exc)
            return False
        token_hash = hash_token(token)
        already_blacklisted = token_hash in self._blacklist
        self._blacklist.add(token_hash)
        jti = payload.get("jti", "")
        if jti:
            self._revoked.add(jti)
        logger.info("Revoked JWT jti=%s (was_blacklisted=%s)", jti, already_blacklisted)
        return not already_blacklisted

    def audit_event_for(
        self,
        token_id: str,
        event: AuthEvent,
        status: AuthStatus,
        actor: str = "system",
    ) -> Dict[str, Any]:
        detail = super().audit_event_for(token_id, event, status, actor)
        detail["algorithm"] = self.config.algorithm
        return detail


__all__ = ["JWTManager"]
