"""
session_manager.py
==================

In-memory session registry with idle / absolute timeout enforcement and
per-user session-count limits.

Sessions track the subject, the originating IP, the device fingerprint
and a cache of permissions granted at login time, so that the
authorisation layer can short-circuit the common case without round-
tripping through the RBAC store on every API call.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .config import SessionConfig
from .constants import AuthEvent, AuthStatus, Role
from .utils import generate_session_id

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A live user/device session."""

    session_id: str
    user_id: str
    created_at: int
    expires_at: int
    last_activity: int
    ip_address: Optional[str] = None
    device_fingerprint: Optional[str] = None
    user_agent: Optional[str] = None
    roles: Set[Role] = field(default_factory=set)
    permissions_cache: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    revoked: bool = False

    def touch(self, now: Optional[int] = None) -> None:
        self.last_activity = now or int(time.time())

    def is_expired(self, now: Optional[int] = None) -> bool:
        ts = now or int(time.time())
        return ts >= self.expires_at or self.revoked

    def is_idle_expired(self, idle_timeout: int, now: Optional[int] = None) -> bool:
        ts = now or int(time.time())
        return ts - self.last_activity >= idle_timeout


class SessionManager:
    """Thread-safe in-memory session store."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._by_user: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def create_session(
        self,
        user_id: str,
        *,
        ip_address: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        user_agent: Optional[str] = None,
        roles: Optional[Iterable[Role]] = None,
        permissions_cache: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_override: Optional[int] = None,
    ) -> Session:
        now = int(time.time())
        ttl = ttl_override or int(self.config.session_ttl.total_seconds())
        expires_at = now + ttl
        session = Session(
            session_id=generate_session_id(),
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            device_fingerprint=device_fingerprint,
            user_agent=user_agent,
            roles=set(roles or []),
            permissions_cache=set(permissions_cache or []),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._enforce_max_sessions(user_id)
            self._sessions[session.session_id] = session
            self._by_user.setdefault(user_id, set()).add(session.session_id)
        logger.info(
            "Created session id=%s user=%s expires_at=%s",
            session.session_id, user_id, expires_at,
        )
        return session

    def get_session(self, session_id: str, touch: bool = True) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired():
                self._evict(session_id)
                return None
            idle_t = int(self.config.idle_timeout.total_seconds())
            if session.is_idle_expired(idle_t):
                logger.info("Session %s idle-expired", session_id)
                self._evict(session_id)
                return None
            if touch:
                session.touch()
            return session

    def destroy_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.revoked = True
            self._evict(session_id)
        logger.info("Destroyed session %s", session_id)
        return True

    def destroy_all_for_user(self, user_id: str) -> int:
        with self._lock:
            ids = list(self._by_user.get(user_id, set()))
        for sid in ids:
            self.destroy_session(sid)
        return len(ids)

    def list_user_sessions(self, user_id: str) -> List[Session]:
        with self._lock:
            ids = list(self._by_user.get(user_id, set()))
        sessions: List[Session] = []
        for sid in ids:
            sess = self.get_session(sid, touch=False)
            if sess is not None:
                sessions.append(sess)
        return sessions

    def cleanup_expired(self) -> int:
        """Evict every expired / idle-expired session. Returns evicted count."""
        evicted = 0
        now = int(time.time())
        idle_t = int(self.config.idle_timeout.total_seconds())
        with self._lock:
            to_check = list(self._sessions.keys())
        for sid in to_check:
            with self._lock:
                sess = self._sessions.get(sid)
                if sess is None:
                    continue
                if sess.is_expired(now) or sess.is_idle_expired(idle_t, now):
                    self._evict(sid)
                    evicted += 1
        if evicted:
            logger.info("Cleaned up %d expired sessions", evicted)
        return evicted

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def update_permissions_cache(
        self, session_id: str, permissions: Iterable[str]
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.permissions_cache = set(permissions)
            return True

    def has_cached_permission(self, session_id: str, permission: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            return permission in session.permissions_cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _enforce_max_sessions(self, user_id: str) -> None:
        ids = self._by_user.get(user_id, set())
        if len(ids) < self.config.max_sessions_per_user:
            return
        # evict oldest sessions first
        sessions = sorted(
            (self._sessions[sid] for sid in ids if sid in self._sessions),
            key=lambda s: s.created_at,
        )
        excess = len(sessions) - self.config.max_sessions_per_user + 1
        for sess in sessions[:excess]:
            logger.info("Evicting oldest session %s for user %s", sess.session_id, user_id)
            self._evict(sess.session_id)

    def _evict(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        user_set = self._by_user.get(session.user_id)
        if user_set is not None:
            user_set.discard(session_id)
            if not user_set:
                self._by_user.pop(session.user_id, None)

    # ------------------------------------------------------------------
    # Audit-event helper
    # ------------------------------------------------------------------
    def session_audit_detail(self, session: Session, event: AuthEvent, status: AuthStatus) -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "event": event.value,
            "status": status.value,
            "ip_address": session.ip_address,
            "device_fingerprint": session.device_fingerprint,
        }


__all__ = ["Session", "SessionManager"]
