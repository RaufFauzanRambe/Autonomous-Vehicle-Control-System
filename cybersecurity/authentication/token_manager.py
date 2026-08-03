"""
token_manager.py
================

Abstract :class:`TokenManager` and supporting dataclasses.

The concrete token implementations (``JWTManager``, opaque-token manager,
OAuth2 access tokens) all inherit from this base class so that callers
can swap token backends without touching their business logic.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from .constants import AuthEvent, AuthStatus, TOKEN_TYPE_ACCESS, TOKEN_TYPE_REFRESH

logger = logging.getLogger(__name__)


@dataclass
class TokenClaims:
    """Normalised view of a token's claims, regardless of backend."""

    subject: str
    issuer: str
    audience: str
    issued_at: int
    expires_at: int
    not_before: int
    token_id: str
    token_type: str = TOKEN_TYPE_ACCESS
    scopes: Set[str] = field(default_factory=set)
    custom: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[int] = None) -> bool:
        ts = now or int(datetime.now(tz=timezone.utc).timestamp())
        return ts >= self.expires_at

    def is_active(self, now: Optional[int] = None) -> bool:
        ts = now or int(datetime.now(tz=timezone.utc).timestamp())
        return self.not_before <= ts < self.expires_at


@dataclass
class TokenIssueResult:
    """The bundle returned to a caller when a token is issued."""

    access_token: str
    refresh_token: Optional[str]
    token_type: str = "Bearer"
    expires_in: int = 0
    claims: Optional[TokenClaims] = None


class TokenManager(abc.ABC):
    """Abstract base for all token backends.

    Concrete subclasses must implement at minimum ``issue_token`` and
    ``validate_token``. ``refresh_token`` and ``revoke_token`` have
    default no-op implementations so that backends without refresh /
    revocation support still satisfy the interface.
    """

    def __init__(self, issuer: str, audience: str) -> None:
        self.issuer = issuer
        self.audience = audience
        self._revoked: Set[str] = set()  # token_id blacklist

    # ------------------------------------------------------------------
    # Abstract operations
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def issue_token(
        self,
        subject: str,
        *,
        scopes: Optional[Set[str]] = None,
        ttl_seconds: int = 900,
        custom_claims: Optional[Dict[str, Any]] = None,
        token_type: str = TOKEN_TYPE_ACCESS,
    ) -> TokenIssueResult:
        """Issue a new token for ``subject`` and return the bundle."""

    @abc.abstractmethod
    def validate_token(self, token: str) -> Optional[TokenClaims]:
        """Validate ``token`` and return its claims, or ``None`` if invalid."""

    # ------------------------------------------------------------------
    # Optional operations
    # ------------------------------------------------------------------
    def refresh_token(self, refresh_token: str) -> Optional[TokenIssueResult]:
        """Exchange a refresh token for a fresh access/refresh pair.

        Default implementation refuses refresh; subclasses that support
        refresh (JWT, OAuth2) override this.
        """
        logger.warning("refresh_token not implemented for %s", self.__class__.__name__)
        return None

    def revoke_token(self, token: str) -> bool:
        """Revoke ``token``. Returns ``True`` if revocation was applied."""
        claims = self.validate_token(token)
        if claims is None:
            return False
        self._revoked.add(claims.token_id)
        logger.info("Revoked token jti=%s subject=%s", claims.token_id, claims.subject)
        return True

    def is_revoked(self, token_id: str) -> bool:
        return token_id in self._revoked

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def audit_event_for(
        self,
        token_id: str,
        event: AuthEvent,
        status: AuthStatus,
        actor: str = "system",
    ) -> Dict[str, Any]:
        """Build an audit ``detail`` dict describing a token event."""
        return {
            "token_id": token_id,
            "event": event.value,
            "status": status.value,
            "actor": actor,
            "manager": self.__class__.__name__,
        }


__all__ = ["TokenClaims", "TokenIssueResult", "TokenManager"]
