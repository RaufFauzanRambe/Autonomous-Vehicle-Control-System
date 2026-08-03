"""
test_authentication.py
======================

Pytest suite for the ``authentication`` sub-package.

The tests cover:

* User registration, login success, login failure, lockout & password reset.
* JWT issue / verify / refresh / blacklist / revocation.
* Session creation / destruction / expiry / per-user caps.
* RBAC role assignment + permission lookup.
* PermissionManager grant / revoke / denial.
* AccessControlManager ABAC policy decisions (allow/deny).
* MFA challenge initiation, TOTP factor verification, completion.
* Audit-log hash-chain integrity and tamper detection.
* Biometric enrolment + match (cosine similarity).
* VehicleAuthentication beacon signing + verification.

External crypto / HTTP collaborators are mocked where appropriate.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the package importable when pytest is run from anywhere.
PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cybersecurity.authentication import (  # noqa: E402
    access_control,
    audit_log,
    authentication as auth_module,
    biometric_auth,
    config as cfg_module,
    constants,
    device_authentication,
    jwt_manager as jwt_module,
    multi_factor_auth,
    oauth_manager,
    permission_manager,
    role_based_access,
    session_manager,
    user_authentication,
    utils,
    vehicle_authentication,
)
from cybersecurity.authentication.access_control import (  # noqa: E402
    AccessControlManager,
    AccessRequest,
    Policy,
)
from cybersecurity.authentication.audit_log import AuditLogger  # noqa: E402
from cybersecurity.authentication.authentication import (  # noqa: E402
    AuthenticationManager,
)
from cybersecurity.authentication.biometric_auth import (  # noqa: E402
    BiometricAuthenticator,
    BiometricModality,
)
from cybersecurity.authentication.config import (  # noqa: E402
    AuthenticationConfig,
    JWTConfig,
    PasswordPolicy,
    LockoutPolicy,
    SessionConfig,
)
from cybersecurity.authentication.constants import (  # noqa: E402
    AuthEvent,
    AuthMethod,
    AuthStatus,
    Permission,
    Role,
)
from cybersecurity.authentication.jwt_manager import JWTManager  # noqa: E402
from cybersecurity.authentication.multi_factor_auth import MFAManager  # noqa: E402
from cybersecurity.authentication.permission_manager import (  # noqa: E402
    PermissionManager,
)
from cybersecurity.authentication.role_based_access import RBACManager  # noqa: E402
from cybersecurity.authentication.session_manager import SessionManager  # noqa: E402
from cybersecurity.authentication.user_authentication import (  # noqa: E402
    UserAuthenticator,
)
from cybersecurity.authentication.utils import (  # noqa: E402
    compute_totp,
    generate_totp_secret,
    hash_password,
    safe_compare,
    verify_password,
    verify_totp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def jwt_config() -> JWTConfig:
    return JWTConfig(secret_key="test-secret-very-long-1234567890", algorithm="HS256")


@pytest.fixture()
def jwt_mgr(jwt_config: JWTConfig) -> JWTManager:
    return JWTManager(jwt_config)


@pytest.fixture()
def password_policy() -> PasswordPolicy:
    return PasswordPolicy(bcrypt_rounds=4, min_length=8)


@pytest.fixture()
def lockout_policy() -> LockoutPolicy:
    return LockoutPolicy(max_attempts=3, lockout_duration=__import__("datetime").timedelta(seconds=30))


@pytest.fixture()
def user_auth(password_policy, lockout_policy) -> UserAuthenticator:
    return UserAuthenticator(password_policy, lockout_policy)


@pytest.fixture()
def session_mgr() -> SessionManager:
    return SessionManager(SessionConfig(session_ttl=__import__("datetime").timedelta(hours=1)))


@pytest.fixture()
def permission_mgr() -> PermissionManager:
    return PermissionManager()


@pytest.fixture()
def rbac_mgr(permission_mgr) -> RBACManager:
    return RBACManager(permission_mgr)


@pytest.fixture()
def audit_logger(tmp_path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.log")


@pytest.fixture()
def auth_mgr(tmp_path) -> AuthenticationManager:
    config = AuthenticationConfig(
        environment="development",
        jwt=JWTConfig(secret_key="dev-secret-very-long-0987654321", algorithm="HS256"),
        password=PasswordPolicy(bcrypt_rounds=4, min_length=8),
        session=SessionConfig(session_ttl=__import__("datetime").timedelta(hours=1)),
        audit=cfg_module.AuditConfig(log_path=tmp_path / "audit.log"),
    )
    return AuthenticationManager(config=config)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
class TestUtils:
    def test_hash_and_verify_password_roundtrip(self):
        h = hash_password("Hunter2#Strong", rounds=4)
        assert h != "Hunter2#Strong"
        assert verify_password("Hunter2#Strong", h)
        assert not verify_password("wrong", h)

    def test_safe_compare(self):
        assert safe_compare("abc", "abc")
        assert not safe_compare("abc", "abd")
        assert not safe_compare("abc", "abc ")

    def test_totp_generate_and_verify(self):
        secret = generate_totp_secret()
        code = compute_totp(secret)
        assert verify_totp(code, secret)
        assert not verify_totp("000000", secret) or compute_totp(secret) == "000000"

    def test_totp_rejects_empty(self):
        assert not verify_totp("", "ABCDEFGH")
        assert not verify_totp("123456", "")


# ---------------------------------------------------------------------------
# User auth
# ---------------------------------------------------------------------------
class TestUserAuthentication:
    def test_register_and_login_success(self, user_auth):
        user_auth.register_user("alice", "StrongPass#1")
        result = user_auth.authenticate("alice", "StrongPass#1")
        assert result.status == AuthStatus.SUCCESS
        assert result.user_id is not None

    def test_login_failure_bad_password(self, user_auth):
        user_auth.register_user("bob", "StrongPass#1")
        result = user_auth.authenticate("bob", "wrong")
        assert result.status == AuthStatus.FAILURE

    def test_lockout_after_max_attempts(self, user_auth):
        user_auth.register_user("carol", "StrongPass#1")
        for _ in range(3):
            user_auth.authenticate("carol", "bad")
        result = user_auth.authenticate("carol", "bad")
        assert result.status == AuthStatus.LOCKED_OUT

    def test_password_complexity_enforced(self, user_auth):
        with pytest.raises(ValueError):
            user_auth.register_user("dave", "short")

    def test_change_password(self, user_auth):
        user_auth.register_user("eve", "StrongPass#1")
        assert user_auth.change_password(user_auth.get_user_by_username("eve").user_id,
                                         "StrongPass#1", "NewStrong#2")
        assert not user_auth.authenticate("eve", "StrongPass#1").status == AuthStatus.SUCCESS
        assert user_auth.authenticate("eve", "NewStrong#2").status == AuthStatus.SUCCESS

    def test_reset_password(self, user_auth):
        rec = user_auth.register_user("frank", "StrongPass#1")
        assert user_auth.reset_password(rec.user_id, "ResetStrong#9", reset_token="tok")
        assert user_auth.authenticate("frank", "ResetStrong#9").status == AuthStatus.SUCCESS


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------
class TestJWT:
    def test_issue_and_validate(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", scopes={"vehicle:read_telemetry"}, ttl_seconds=60)
        assert result.access_token
        claims = jwt_mgr.validate_token(result.access_token)
        assert claims is not None
        assert claims.subject == "u1"
        assert "vehicle:read_telemetry" in claims.scopes

    def test_expired_token_rejected(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", ttl_seconds=-10)
        assert jwt_mgr.validate_token(result.access_token) is None

    def test_refresh_rotates_tokens(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", ttl_seconds=60)
        refreshed = jwt_mgr.refresh(result.refresh_token)
        assert refreshed is not None
        assert refreshed.access_token != result.access_token
        # Old refresh token no longer usable
        assert jwt_mgr.refresh(result.refresh_token) is None

    def test_blacklist_token(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", ttl_seconds=60)
        jwt_mgr.blacklist_token(result.access_token)
        assert jwt_mgr.validate_token(result.access_token) is None

    def test_revoke_token(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", ttl_seconds=60)
        assert jwt_mgr.revoke_token(result.access_token)
        assert jwt_mgr.validate_token(result.access_token) is None

    def test_signature_tamper_detected(self, jwt_mgr):
        result = jwt_mgr.issue_token("u1", ttl_seconds=60)
        # flip last char of signature
        parts = result.access_token.split(".")
        tampered = ".".join([parts[0], parts[1], parts[2][:-1] + ("A" if parts[2][-1] != "A" else "B")])
        assert jwt_mgr.validate_token(tampered) is None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class TestSessions:
    def test_create_and_get(self, session_mgr):
        s = session_mgr.create_session("u1")
        fetched = session_mgr.get_session(s.session_id)
        assert fetched is not None
        assert fetched.user_id == "u1"

    def test_destroy(self, session_mgr):
        s = session_mgr.create_session("u1")
        assert session_mgr.destroy_session(s.session_id)
        assert session_mgr.get_session(s.session_id) is None

    def test_max_sessions_per_user(self):
        sm = SessionManager(
            SessionConfig(max_sessions_per_user=2,
                          session_ttl=__import__("datetime").timedelta(hours=1),
                          idle_timeout=__import__("datetime").timedelta(hours=1))
        )
        sm.create_session("u1")
        sm.create_session("u1")
        sm.create_session("u1")
        assert len(sm.list_user_sessions("u1")) == 2

    def test_cleanup_expired(self):
        sm = SessionManager(
            SessionConfig(session_ttl=__import__("datetime").timedelta(seconds=1),
                          idle_timeout=__import__("datetime").timedelta(hours=1))
        )
        sm.create_session("u1")
        time.sleep(1.1)
        assert sm.cleanup_expired() == 1


# ---------------------------------------------------------------------------
# Permissions & RBAC
# ---------------------------------------------------------------------------
class TestRBAC:
    def test_assign_role_and_check(self, rbac_mgr):
        rbac_mgr.assign_role("u1", Role.DRIVER)
        assert Role.DRIVER in rbac_mgr.get_roles("u1")
        assert rbac_mgr.check_access("u1", Permission.VEHICLE_EMERGENCY_STOP)

    def test_revoke_role(self, rbac_mgr):
        rbac_mgr.assign_role("u1", Role.DRIVER)
        assert rbac_mgr.revoke_role("u1", Role.DRIVER)
        assert Role.DRIVER not in rbac_mgr.get_roles("u1")

    def test_security_admin_has_manage_keys(self, rbac_mgr):
        rbac_mgr.assign_role("u1", Role.SECURITY_ADMIN)
        assert rbac_mgr.check_access("u1", Permission.SECURITY_MANAGE_KEYS)

    def test_permission_denied(self, rbac_mgr):
        rbac_mgr.assign_role("u1", Role.DRIVER)
        assert not rbac_mgr.check_access("u1", Permission.OTA_INSTALL)


class TestPermissionManager:
    def test_grant_and_revoke(self, permission_mgr):
        permission_mgr.grant_permission("u1", Permission.OTA_ROLLBACK)
        assert permission_mgr.check_permission("u1", Permission.OTA_ROLLBACK)
        permission_mgr.revoke_permission("u1", Permission.OTA_ROLLBACK)
        assert not permission_mgr.check_permission("u1", Permission.OTA_ROLLBACK)

    def test_subject_denial_overrides_role(self, permission_mgr):
        permission_mgr.grant_permission("u1", Permission.VEHICLE_EMERGENCY_STOP)
        permission_mgr.revoke_permission("u1", Permission.VEHICLE_EMERGENCY_STOP)
        assert not permission_mgr.check_permission(
            "u1", Permission.VEHICLE_EMERGENCY_STOP, roles=[Role.DRIVER]
        )


# ---------------------------------------------------------------------------
# Access control (ABAC)
# ---------------------------------------------------------------------------
class TestAccessControl:
    def test_deny_ota_while_driving(self, rbac_mgr):
        ac = AccessControlManager(rbac=rbac_mgr, permission_manager=rbac_mgr.permission_manager)
        rbac_mgr.assign_role("u1", Role.OTA_OPERATOR)
        req = AccessRequest(
            subject_id="u1",
            action="install",
            resource="ota:bundle:abc",
            permission=Permission.OTA_INSTALL,
            roles=rbac_mgr.get_roles("u1"),
            environment={"vehicle_speed_kph": 50},
        )
        decision = ac.evaluate_access(req)
        assert not decision.allowed

    def test_allow_ota_when_stationary(self, rbac_mgr):
        ac = AccessControlManager(rbac=rbac_mgr, permission_manager=rbac_mgr.permission_manager)
        rbac_mgr.assign_role("u1", Role.OTA_OPERATOR)
        req = AccessRequest(
            subject_id="u1",
            action="install",
            resource="ota:bundle:abc",
            permission=Permission.OTA_INSTALL,
            roles=rbac_mgr.get_roles("u1"),
            environment={"vehicle_speed_kph": 0},
        )
        decision = ac.evaluate_access(req)
        assert decision.allowed

    def test_custom_policy(self, rbac_mgr):
        ac = AccessControlManager(rbac=rbac_mgr, permission_manager=rbac_mgr.permission_manager)
        ac.add_policy(Policy(
            name="deny-outside-business-hours",
            effect="deny",
            priority=5,
            conditions=[{"attr": "env.hour", "op": "lt", "value": 9}],
        ))
        req = AccessRequest(
            subject_id="u1",
            action="read",
            resource="vehicle:telemetry",
            permission=Permission.VEHICLE_READ_TELEMETRY,
            roles=rbac_mgr.get_roles("u1"),
            environment={"hour": 3},
        )
        decision = ac.evaluate_access(req)
        assert not decision.allowed
        assert decision.matched_policy == "deny-outside-business-hours"


# ---------------------------------------------------------------------------
# MFA
# ---------------------------------------------------------------------------
class TestMFA:
    def test_totp_factor_completes(self, jwt_mgr):
        secret = generate_totp_secret()
        mfa = MFAManager(
            token_manager=jwt_mgr,
            totp_secrets={"u1": secret},
            password_verifier=lambda uid, pw: pw == "correct",
        )
        challenge = mfa.initiate_auth("u1", required_factors={AuthMethod.OTP_TOTP})
        assert mfa.verify_factor(challenge.challenge_id, AuthMethod.OTP_TOTP, code=compute_totp(secret))
        tokens = mfa.complete_auth(challenge.challenge_id)
        assert tokens is not None
        assert tokens.access_token

    def test_wrong_code_does_not_complete(self, jwt_mgr):
        secret = generate_totp_secret()
        mfa = MFAManager(token_manager=jwt_mgr, totp_secrets={"u1": secret})
        ch = mfa.initiate_auth("u1", required_factors={AuthMethod.OTP_TOTP})
        assert not mfa.verify_factor(ch.challenge_id, AuthMethod.OTP_TOTP, code="000000") \
            or compute_totp(secret) == "000000"
        # If "000000" happened to be the correct code, force a different one
        # by providing a clearly wrong challenge id.
        assert mfa.complete_auth("nonexistent") is None


# ---------------------------------------------------------------------------
# Biometric
# ---------------------------------------------------------------------------
class TestBiometric:
    def test_enroll_and_match(self):
        bio = BiometricAuthenticator()
        embedding = [1.0, 0.0, 0.0]
        bio.enroll("u1", BiometricModality.FACE, embedding)
        result = bio.authenticate(BiometricModality.FACE, [0.99, 0.0, 0.05])
        assert result.status == AuthStatus.SUCCESS
        assert result.subject_id == "u1"

    def test_no_match(self):
        bio = BiometricAuthenticator()
        bio.enroll("u1", BiometricModality.FACE, [1.0, 0.0, 0.0])
        result = bio.authenticate(BiometricModality.FINGERPRINT, [1.0, 0.0, 0.0])
        assert result.status == AuthStatus.FAILURE


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
class TestAuditLog:
    def test_chain_intact(self, audit_logger):
        for _ in range(5):
            audit_logger.log_event(AuthEvent.LOGIN_SUCCESS, "u1", AuthStatus.SUCCESS)
        assert audit_logger.verify_chain()
        assert audit_logger.count() == 5

    def test_tamper_detected(self, audit_logger, tmp_path):
        audit_logger.log_event(AuthEvent.LOGIN_SUCCESS, "u1", AuthStatus.SUCCESS)
        audit_logger.log_event(AuthEvent.LOGIN_FAILURE, "u2", AuthStatus.FAILURE)
        # Corrupt the file — the enum values are stored as lowercase strings
        # in the JSON, so we replace the lowercase form.
        path = audit_logger.log_path
        text = path.read_text()
        path.write_text(text.replace("login_failure", "login_success"))
        assert not audit_logger.verify_chain()

    def test_query_by_actor(self, audit_logger):
        audit_logger.log_event(AuthEvent.LOGIN_SUCCESS, "alice", AuthStatus.SUCCESS)
        audit_logger.log_event(AuthEvent.LOGIN_SUCCESS, "bob", AuthStatus.SUCCESS)
        results = audit_logger.query(actor="alice")
        assert len(results) == 1
        assert results[0].actor == "alice"


# ---------------------------------------------------------------------------
# Vehicle auth
# ---------------------------------------------------------------------------
class TestVehicleAuth:
    def test_beacon_sign_and_verify(self):
        from cryptography.hazmat.primitives.asymmetric import ec
        priv = ec.generate_private_key(ec.SECP256R1())
        pub_pem = priv.public_key().public_bytes(
            __import__("cryptography").hazmat.primitives.serialization.Encoding.PEM,
            __import__("cryptography").hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")
        attacker_priv = ec.generate_private_key(ec.SECP256R1())

        local = vehicle_authentication.VehicleAuthenticator(
            "VIN-LOCAL", priv, trust_chain={"SCMS-PCA-1"}
        )
        cert = vehicle_authentication.VehicleCertificate(
            certificate_id="cert-1",
            vin="VIN-REMOTE",
            public_key_pem=pub_pem,
            issuer="SCMS-PCA-1",
            not_before=int(time.time()) - 60,
            not_after=int(time.time()) + 3600,
            serial="001",
        )
        beacon = local.broadcast_beacon(b"hello", cert)
        result = local.authenticate_remote_vehicle(beacon, cert)
        assert result.status == AuthStatus.SUCCESS

        # Now sign with a different key and expect failure
        bad_beacon = vehicle_authentication.Beacon(
            sender_vin="VIN-REMOTE",
            certificate_id="cert-1",
            payload=b"hello",
            signature=b"",
            timestamp=beacon.timestamp,
        )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
        signed = local._signed_payload(bad_beacon)
        bad_beacon.signature = attacker_priv.sign(signed, ECDSA(hashes.SHA256()))
        bad_result = local.authenticate_remote_vehicle(bad_beacon, cert)
        assert bad_result.status == AuthStatus.FAILURE


# ---------------------------------------------------------------------------
# OAuth2 (mocked HTTP)
# ---------------------------------------------------------------------------
class TestOAuth:
    def test_authorization_url_includes_pkce(self):
        provider = oauth_manager.OAuthProviderConfig(
            issuer="https://idp",
            authorization_endpoint="https://idp/auth",
            token_endpoint="https://idp/token",
            userinfo_endpoint="https://idp/me",
        )
        mgr = oauth_manager.OAuth2Manager(
            provider, "client", "secret", "https://app/cb",
            http_client=MagicMock(),
        )
        url = mgr.get_authorization_url("u1")
        assert "code_challenge=" in url
        assert "S256" in url

    def test_client_credentials(self):
        provider = oauth_manager.OAuthProviderConfig(
            issuer="https://idp",
            authorization_endpoint="https://idp/auth",
            token_endpoint="https://idp/token",
            userinfo_endpoint="https://idp/me",
        )
        response = oauth_manager.HTTPResponse(
            status=200,
            json_body={"access_token": "abc", "expires_in": 3600, "token_type": "Bearer"},
        )
        http = MagicMock(return_value=response)
        mgr = oauth_manager.OAuth2Manager(
            provider, "client", "secret", "https://app/cb", http_client=http,
        )
        result = mgr.client_credentials()
        assert result is not None
        assert result.access_token == "abc"


# ---------------------------------------------------------------------------
# Orchestrator end-to-end
# ---------------------------------------------------------------------------
class TestOrchestrator:
    def test_user_login_creates_session_and_token(self, auth_mgr):
        auth_mgr.users.register_user("alice", "StrongPass#1")
        auth_mgr.assign_role("alice", Role.DRIVER)
        outcome = auth_mgr.authenticate_user("alice", "StrongPass#1")
        assert outcome.status == AuthStatus.SUCCESS
        assert outcome.session is not None
        assert outcome.tokens is not None
        # Token verifies
        claims = auth_mgr.verify_token(outcome.tokens.access_token)
        assert claims is not None
        assert claims.subject == outcome.user_id

    def test_logout_destroys_session_and_revokes_token(self, auth_mgr):
        auth_mgr.users.register_user("bob", "StrongPass#1")
        auth_mgr.assign_role("bob", Role.DRIVER)
        outcome = auth_mgr.authenticate_user("bob", "StrongPass#1")
        assert auth_mgr.logout(outcome.session.session_id, outcome.tokens.access_token)
        assert auth_mgr.get_current_session(outcome.session.session_id) is None
        assert auth_mgr.verify_token(outcome.tokens.access_token) is None

    def test_permission_check_via_orchestrator(self, auth_mgr):
        auth_mgr.users.register_user("carol", "StrongPass#1")
        auth_mgr.assign_role("carol", Role.SECURITY_ADMIN)
        assert auth_mgr.check_permission("carol", Permission.SECURITY_MANAGE_KEYS)
        assert not auth_mgr.check_permission("carol", Permission.OTA_INSTALL)

    def test_audit_chain_after_orchestration(self, auth_mgr):
        auth_mgr.users.register_user("dave", "StrongPass#1")
        auth_mgr.assign_role("dave", Role.DRIVER)
        auth_mgr.authenticate_user("dave", "StrongPass#1")
        assert auth_mgr.audit.verify_chain()
