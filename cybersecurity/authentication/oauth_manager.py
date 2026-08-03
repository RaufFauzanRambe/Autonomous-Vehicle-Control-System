"""
oauth_manager.py
================

OAuth 2.0 / OpenID Connect client manager.

Implements the three flows most relevant to the AV platform:

* **Authorization Code** with optional **PKCE** — used by in-vehicle
  infotainment UIs that need to act on behalf of a driver.
* **Client Credentials** — used by machine-to-machine backends (e.g. an
  OTA service talking to the fleet manager).
* **Refresh Token** grant for silent renewal.

The manager does not ship a network client; callers inject an
``http_client`` callable with a ``(method, url, headers, data, timeout)``
signature so that the manager can be unit-tested with mocks.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set
from urllib.parse import urlencode

from .constants import DEFAULT_ACCESS_TOKEN_TTL
from .token_manager import TokenClaims, TokenIssueResult, TokenManager
from .utils import generate_token_id, hash_token

logger = logging.getLogger(__name__)

HTTPClient = Callable[..., "HTTPResponse"]


@dataclass
class HTTPResponse:
    """Minimal HTTP response envelope used by :class:`OAuth2Manager`."""

    status: int
    json_body: Dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class OAuthProviderConfig:
    """Discovery-style configuration for an OAuth 2.0 provider."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    revoke_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    scopes_supported: tuple = ("openid", "profile", "email")
    code_challenge_methods_supported: tuple = ("S256",)


class OAuth2Manager(TokenManager):
    """OAuth 2.0 / OIDC client.

    Parameters
    ----------
    provider:
        :class:`OAuthProviderConfig` describing the IdP endpoints.
    client_id / client_secret:
        Client credentials registered with the IdP.
    redirect_uri:
        Redirect URI registered for the authorization-code flow.
    http_client:
        Callable invoked as ``http_client(method, url, *, headers, data,
        timeout)`` returning an :class:`HTTPResponse`.
    use_pkce:
        Whether to require PKCE on the authorization-code flow.
    """

    GRANT_AUTH_CODE = "authorization_code"
    GRANT_CLIENT_CREDENTIALS = "client_credentials"
    GRANT_REFRESH = "refresh_token"

    def __init__(
        self,
        provider: OAuthProviderConfig,
        client_id: str,
        client_secret: Optional[str],
        redirect_uri: str,
        http_client: HTTPClient,
        *,
        use_pkce: bool = True,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> None:
        super().__init__(
            issuer=issuer or provider.issuer,
            audience=audience or client_id,
        )
        self.provider = provider
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.http_client = http_client
        self.use_pkce = use_pkce
        # state -> {"code_verifier": str, "subject_hint": str, "expires_at": int}
        self._pending: Dict[str, Dict[str, Any]] = {}
        # subject -> refresh-token hash set
        self._refresh_registry: Dict[str, set] = {}
        self._blacklist: set = set()

    # ------------------------------------------------------------------
    # PKCE helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_pkce_pair() -> Dict[str, str]:
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return {"verifier": verifier, "challenge": challenge, "method": "S256"}

    # ------------------------------------------------------------------
    # Authorization-code flow
    # ------------------------------------------------------------------
    def get_authorization_url(
        self,
        subject_hint: str = "",
        scopes: Optional[tuple] = None,
        state: Optional[str] = None,
    ) -> str:
        """Build an authorization URL (with PKCE if enabled).

        Returns the URL the user-agent should be redirected to. The PKCE
        verifier and state are cached internally for later correlation
        in :meth:`exchange_code`.
        """
        state = state or secrets.token_urlsafe(16)
        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
            "scope": " ".join(scopes or self.provider.scopes_supported),
        }
        pkce: Optional[Dict[str, str]] = None
        if self.use_pkce and "S256" in self.provider.code_challenge_methods_supported:
            pkce = self._make_pkce_pair()
            params["code_challenge"] = pkce["challenge"]
            params["code_challenge_method"] = pkce["method"]
        self._pending[state] = {
            "code_verifier": pkce["verifier"] if pkce else None,
            "subject_hint": subject_hint,
            "expires_at": int(time.time()) + 600,
        }
        return f"{self.provider.authorization_endpoint}?{urlencode(params)}"

    def exchange_code(
        self,
        code: str,
        state: str,
    ) -> Optional[TokenIssueResult]:
        """Exchange an authorization code for tokens."""
        pending = self._pending.pop(state, None)
        if pending is None:
            logger.warning("OAuth exchange: unknown state")
            return None
        if pending["expires_at"] < int(time.time()):
            logger.warning("OAuth exchange: state expired")
            return None
        data: Dict[str, str] = {
            "grant_type": self.GRANT_AUTH_CODE,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }
        if pending.get("code_verifier"):
            data["code_verifier"] = pending["code_verifier"]
        if self.client_secret:
            data["client_secret"] = self.client_secret
        resp = self.http_client(
            "POST",
            self.provider.token_endpoint,
            data=data,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status != 200:
            logger.info("OAuth token endpoint returned %s", resp.status)
            return None
        return self._build_result(resp.json_body, subject_hint=pending.get("subject_hint", ""))

    # ------------------------------------------------------------------
    # Client-credentials flow
    # ------------------------------------------------------------------
    def client_credentials(
        self,
        scopes: Optional[tuple] = None,
    ) -> Optional[TokenIssueResult]:
        """Obtain a token via the ``client_credentials`` grant."""
        data: Dict[str, str] = {
            "grant_type": self.GRANT_CLIENT_CREDENTIALS,
            "client_id": self.client_id,
            "scope": " ".join(scopes or self.provider.scopes_supported),
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        resp = self.http_client(
            "POST",
            self.provider.token_endpoint,
            data=data,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status != 200:
            logger.info("client_credentials grant failed: %s", resp.status)
            return None
        return self._build_result(resp.json_body)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh_token(self, refresh_token: str) -> Optional[TokenIssueResult]:
        if hash_token(refresh_token) in self._blacklist:
            return None
        resp = self.http_client(
            "POST",
            self.provider.token_endpoint,
            data={
                "grant_type": self.GRANT_REFRESH,
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                **({"client_secret": self.client_secret} if self.client_secret else {}),
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status != 200:
            logger.info("refresh_token grant failed: %s", resp.status)
            return None
        self._blacklist.add(hash_token(refresh_token))
        return self._build_result(resp.json_body)

    # ------------------------------------------------------------------
    # Token validation / userinfo
    # ------------------------------------------------------------------
    def validate_token(self, token: str) -> Optional[TokenClaims]:
        if hash_token(token) in self._blacklist:
            return None
        resp = self.http_client(
            "GET",
            self.provider.userinfo_endpoint,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=10,
        )
        if resp.status != 200:
            return None
        info = resp.json_body
        now = int(time.time())
        ttl = int(DEFAULT_ACCESS_TOKEN_TTL.total_seconds())
        return TokenClaims(
            subject=str(info.get("sub", "")),
            issuer=self.issuer,
            audience=self.audience,
            issued_at=now,
            expires_at=now + ttl,
            not_before=now,
            token_id=generate_token_id(),
            token_type="access",
            scopes=set(),
            custom=dict(info),
        )

    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Call the IdP userinfo endpoint and return the parsed response."""
        resp = self.http_client(
            "GET",
            self.provider.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=10,
        )
        if resp.status != 200:
            return None
        return resp.json_body

    # ------------------------------------------------------------------
    # TokenManager abstract op
    # ------------------------------------------------------------------
    def issue_token(
        self,
        subject: str,
        *,
        scopes: Optional[Set[str]] = None,
        ttl_seconds: int = 900,
        custom_claims: Optional[Dict[str, Any]] = None,
        token_type: str = "access",
    ) -> TokenIssueResult:
        return self.client_credentials(scopes=tuple(scopes) if scopes else None) \
            or TokenIssueResult(access_token="", refresh_token=None, expires_in=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        body: Dict[str, Any],
        subject_hint: str = "",
    ) -> TokenIssueResult:
        access = body.get("access_token", "")
        refresh = body.get("refresh_token")
        expires_in = int(body.get("expires_in", 0))
        now = int(time.time())
        subject = subject_hint or body.get("sub", "") or self.client_id
        if refresh and subject:
            self._refresh_registry.setdefault(subject, set()).add(hash_token(refresh))
        claims = TokenClaims(
            subject=subject,
            issuer=self.issuer,
            audience=self.audience,
            issued_at=now,
            expires_at=now + expires_in,
            not_before=now,
            token_id=generate_token_id(),
            token_type="access",
            scopes=set(body.get("scope", "").split()) if body.get("scope") else set(),
        )
        return TokenIssueResult(
            access_token=access,
            refresh_token=refresh,
            token_type=body.get("token_type", "Bearer"),
            expires_in=expires_in,
            claims=claims,
        )


__all__ = ["OAuth2Manager", "OAuthProviderConfig", "HTTPResponse"]
