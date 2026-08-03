"""
multi_factor_auth.py
====================

Multi-Factor Authentication (MFA) orchestrator.

Factors supported:
  * **password**  — handled by :class:`UserAuthenticator` (passed in as a
    callable verifier so this module stays decoupled).
  * **OTP / TOTP** — RFC 6238, implemented in :mod:`utils`.
  * **hardware token** — FIDO2 / WebAuthn-style. The actual signature
    verification is delegated to an injected ``hardware_verifier`` callable.
  * **biometric** — delegates to :class:`BiometricAuthenticator`.
  * **push notification** — out-of-band approval; an injected
    ``push_dispatcher`` is invoked and the user must call
    :meth:`approve_push` to confirm.

The orchestrator issues an :class:`MFAChallenge`, tracks the satisfied
factors, and either completes (issuing a token via the supplied
``token_manager``) or fails / times out.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .biometric_auth import BiometricAuthenticator, BiometricModality
from .constants import (
    AuthEvent,
    AuthMethod,
    AuthStatus,
    DEFAULT_MFA_CHALLENGE_TTL,
)
from .token_manager import TokenManager
from .utils import generate_totp_secret, verify_totp

logger = logging.getLogger(__name__)


@dataclass
class MFAChallenge:
    """An in-flight MFA challenge for a subject."""

    challenge_id: str
    subject_id: str
    required_factors: Set[AuthMethod]
    satisfied_factors: Set[AuthMethod] = field(default_factory=set)
    created_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: int = 0
    totp_secret: Optional[str] = None
    push_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[int] = None) -> bool:
        return (now or int(time.time())) >= self.expires_at

    def is_complete(self) -> bool:
        return self.required_factors.issubset(self.satisfied_factors)


# Type aliases for injected callbacks
PasswordVerifier = Callable[[str, str], bool]  # (subject_id, password) -> bool
HardwareVerifier = Callable[[str, str, bytes], bool]  # (subject_id, credential_id, assertion) -> bool
BiometricVerifier = Callable[[str, BiometricModality, List[float]], bool]
PushDispatcher = Callable[[str, str], None]  # (subject_id, push_token) -> None


class MFAManager:
    """Multi-factor authentication orchestrator."""

    def __init__(
        self,
        *,
        token_manager: TokenManager,
        totp_secrets: Optional[Dict[str, str]] = None,
        password_verifier: Optional[PasswordVerifier] = None,
        hardware_verifier: Optional[HardwareVerifier] = None,
        biometric_authenticator: Optional[BiometricAuthenticator] = None,
        push_dispatcher: Optional[PushDispatcher] = None,
        challenge_ttl: int = int(DEFAULT_MFA_CHALLENGE_TTL.total_seconds()),
    ) -> None:
        self.token_manager = token_manager
        self._totp_secrets: Dict[str, str] = dict(totp_secrets or {})
        self.password_verifier = password_verifier
        self.hardware_verifier = hardware_verifier
        self.biometric = biometric_authenticator
        self.push_dispatcher = push_dispatcher
        self.challenge_ttl = challenge_ttl
        self._lock = threading.RLock()
        self._challenges: Dict[str, MFAChallenge] = {}

    # ------------------------------------------------------------------
    # Factor enrolment
    # ------------------------------------------------------------------
    def enroll_factor(
        self,
        subject_id: str,
        factor: AuthMethod,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enroll a new factor for ``subject_id``.

        Returns a dict containing any data the caller needs to complete
        enrolment (e.g. the TOTP secret + otpauth URI).
        """
        if factor == AuthMethod.OTP_TOTP:
            secret = kwargs.get("secret") or generate_totp_secret()
            self._totp_secrets[subject_id] = secret
            issuer = kwargs.get("issuer", "AVCS")
            uri = (
                f"otpauth://totp/{issuer}:{subject_id}?secret={secret}"
                f"&issuer={issuer}&algorithm=SHA1&digits=6&period=30"
            )
            logger.info("Enrolled TOTP for %s", subject_id)
            return {"secret": secret, "otpauth_uri": uri}
        if factor == AuthMethod.PUSH_NOTIFICATION:
            # nothing to store; push destinations are managed by the dispatcher
            logger.info("Enrolled push-notification factor for %s", subject_id)
            return {"enrolled": True}
        if factor == AuthMethod.HARDWARE_TOKEN:
            # in real life the caller registers a FIDO2 credential; here we
            # just acknowledge it.
            logger.info("Enrolled hardware-token factor for %s", subject_id)
            return {"enrolled": True, "credential_id": kwargs.get("credential_id")}
        if factor == AuthMethod.BIOMETRIC:
            if self.biometric is None:
                raise RuntimeError("No biometric_authenticator configured")
            modality = kwargs["modality"]
            embedding = kwargs["embedding"]
            template = self.biometric.enroll(subject_id, modality, embedding)
            return {"template_id": template.template_id}
        raise ValueError(f"Unsupported factor for enrolment: {factor}")

    # ------------------------------------------------------------------
    # Challenge lifecycle
    # ------------------------------------------------------------------
    def initiate_auth(
        self,
        subject_id: str,
        required_factors: Iterable[AuthMethod],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MFAChallenge:
        """Create a new MFA challenge requiring ``required_factors``."""
        now = int(time.time())
        challenge = MFAChallenge(
            challenge_id=secrets.token_urlsafe(24),
            subject_id=subject_id,
            required_factors=set(required_factors),
            expires_at=now + self.challenge_ttl,
            totp_secret=self._totp_secrets.get(subject_id),
            metadata=dict(metadata or {}),
        )
        if not challenge.required_factors:
            raise ValueError("At least one required factor must be specified")
        with self._lock:
            # cancel any previous in-flight challenge for the same subject
            stale = [cid for cid, c in self._challenges.items() if c.subject_id == subject_id]
            for cid in stale:
                self._challenges.pop(cid, None)
            self._challenges[challenge.challenge_id] = challenge
        logger.info(
            "Initiated MFA challenge %s for %s requiring %s",
            challenge.challenge_id, subject_id,
            [f.value for f in challenge.required_factors],
        )
        # Auto-dispatch push challenges if applicable
        if AuthMethod.PUSH_NOTIFICATION in challenge.required_factors and self.push_dispatcher:
            challenge.push_token = secrets.token_urlsafe(16)
            try:
                self.push_dispatcher(subject_id, challenge.push_token)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Push dispatch failed: %s", exc)
        return challenge

    def verify_factor(
        self,
        challenge_id: str,
        factor: AuthMethod,
        **kwargs: Any,
    ) -> bool:
        """Verify a single factor against an in-flight challenge."""
        with self._lock:
            challenge = self._challenges.get(challenge_id)
        if challenge is None:
            logger.info("MFA verify: unknown challenge %s", challenge_id)
            return False
        if challenge.is_expired():
            logger.info("MFA verify: challenge %s expired", challenge_id)
            self._challenges.pop(challenge_id, None)
            return False
        if factor not in challenge.required_factors:
            logger.info("MFA verify: factor %s not required for %s", factor.value, challenge_id)
            return False
        if factor in challenge.satisfied_factors:
            return True

        ok = self._verify_factor(challenge, factor, **kwargs)
        if ok:
            with self._lock:
                challenge.satisfied_factors.add(factor)
            logger.info(
                "MFA factor %s verified for challenge %s (%d/%d)",
                factor.value, challenge_id,
                len(challenge.satisfied_factors), len(challenge.required_factors),
            )
        else:
            logger.info("MFA factor %s FAILED for challenge %s", factor.value, challenge_id)
        return ok

    def complete_auth(self, challenge_id: str) -> Optional[Any]:
        """If all required factors are satisfied, issue tokens and clear."""
        with self._lock:
            challenge = self._challenges.get(challenge_id)
            if challenge is None:
                return None
            if challenge.is_expired():
                self._challenges.pop(challenge_id, None)
                return None
            if not challenge.is_complete():
                return None
            self._challenges.pop(challenge_id, None)
        result = self.token_manager.issue_token(challenge.subject_id)
        logger.info(
            "MFA complete for challenge %s subject %s",
            challenge_id, challenge.subject_id,
        )
        return result

    def cancel_challenge(self, challenge_id: str) -> bool:
        with self._lock:
            return self._challenges.pop(challenge_id, None) is not None

    # ------------------------------------------------------------------
    # Push approval helper
    # ------------------------------------------------------------------
    def approve_push(self, challenge_id: str, push_token: str) -> bool:
        """Approve a push-notification factor out-of-band."""
        with self._lock:
            challenge = self._challenges.get(challenge_id)
        if challenge is None or challenge.push_token is None:
            return False
        from .utils import safe_compare
        if not safe_compare(challenge.push_token, push_token):
            return False
        return self.verify_factor(challenge_id, AuthMethod.PUSH_NOTIFICATION)

    # ------------------------------------------------------------------
    # Internal: per-factor verification logic
    # ------------------------------------------------------------------
    def _verify_factor(self, challenge: MFAChallenge, factor: AuthMethod, **kwargs: Any) -> bool:
        if factor == AuthMethod.PASSWORD:
            if self.password_verifier is None:
                return False
            return self.password_verifier(challenge.subject_id, kwargs.get("password", ""))
        if factor == AuthMethod.OTP_TOTP:
            secret = challenge.totp_secret or self._totp_secrets.get(challenge.subject_id)
            if not secret:
                return False
            return verify_totp(kwargs.get("code", ""), secret)
        if factor == AuthMethod.HARDWARE_TOKEN:
            if self.hardware_verifier is None:
                return False
            return self.hardware_verifier(
                challenge.subject_id,
                kwargs.get("credential_id", ""),
                kwargs.get("assertion", b""),
            )
        if factor == AuthMethod.BIOMETRIC:
            if self.biometric is None:
                return False
            modality = kwargs.get("modality", BiometricModality.FACE)
            embedding = kwargs.get("embedding", [])
            result = self.biometric.authenticate(
                modality, embedding, restrict_to_subject=challenge.subject_id,
            )
            return result.status == AuthStatus.SUCCESS
        if factor == AuthMethod.PUSH_NOTIFICATION:
            # Push is approved out-of-band via :meth:`approve_push`. Direct
            # verify_factor calls for push always fail — the caller must
            # approve via the dispatcher callback path.
            return False
        logger.warning("Unsupported MFA factor: %s", factor)
        return False


__all__ = ["MFAChallenge", "MFAManager"]
