"""Configuration dataclasses and YAML loader for vehicle_security.

``VehicleSecurityConfig`` is the single configuration object consumed by the
``VehicleSecuritySystem`` orchestrator. It is constructed either from a YAML
file via :func:`load_config` or programmatically. Defaults mirror the values
defined in :mod:`.constants`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyYAML is the standard YAML loader; fall back gracefully if missing.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore

from .constants import (
    DEFAULT_BASE_DIR,
    DEFAULT_BASELINE_DB_PATH,
    DEFAULT_CVE_INDEX_PATH,
    DEFAULT_FIREWALL_RULES_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SECURE_STORE_PATH,
    DEFAULT_TPM_DEVICE,
    POLL_INTERVAL_INTEGRITY,
    POLL_INTERVAL_MONITOR,
    POLL_INTERVAL_SAFETY,
    POLL_INTERVAL_TAMPER,
    POLL_INTERVAL_VULN,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class FirewallConfig:
    """Configuration for the CAN/Ethernet firewall."""

    rules_path: str = DEFAULT_FIREWALL_RULES_PATH
    can_allowlist: List[int] = field(default_factory=list)
    can_denylist: List[int] = field(default_factory=list)
    ip_allowlist: List[str] = field(default_factory=list)
    ip_denylist: List[str] = field(default_factory=list)
    blocked_ports: List[int] = field(default_factory=list)
    rate_limit_per_second: int = 500
    dpi_enabled: bool = True
    stateful_tracking: bool = True


@dataclass
class StorageConfig:
    """Configuration for the encrypted secrets store."""

    store_path: str = DEFAULT_SECURE_STORE_PATH
    master_key_id: str = "avcs-hardware-root"
    auto_rotate_days: int = 90
    kdf_iterations: int = 200_000


@dataclass
class MonitorConfig:
    """Polling intervals for background monitors."""

    monitor_interval: float = POLL_INTERVAL_MONITOR
    integrity_interval: float = POLL_INTERVAL_INTEGRITY
    tamper_interval: float = POLL_INTERVAL_TAMPER
    vuln_interval: float = POLL_INTERVAL_VULN
    safety_interval: float = POLL_INTERVAL_SAFETY
    max_events_per_second: int = 1000


@dataclass
class BootConfig:
    """Secure-boot configuration."""

    manifest_path: str = DEFAULT_MANIFEST_PATH
    tpm_device: str = DEFAULT_TPM_DEVICE
    trusted_pubkeys_path: str = "/etc/avcs/security/trusted_pubkeys.pem"
    require_attestation: bool = True


@dataclass
class UpdateConfig:
    """OTA update configuration."""

    server_url: str = "https://ota.avcs-internal.example/api/v2"
    staging_dir: str = "/var/lib/avcs/ota_staging"
    active_slot: str = "A"
    inactive_slot: str = "B"
    chunk_size: int = 1 << 20  # 1 MiB
    max_retries: int = 5


@dataclass
class LockdownConfig:
    """Emergency lockdown configuration."""

    disable_network: bool = True
    disable_ota: bool = True
    isolate_critical_ecus: bool = True
    force_safe_stop: bool = True
    admin_pin_hash_path: str = "/etc/avcs/security/admin_pin.hash"


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass
class VehicleSecurityConfig:
    """Top-level configuration object for the vehicle security system."""

    base_dir: str = DEFAULT_BASE_DIR
    log_path: str = DEFAULT_LOG_PATH
    baseline_db_path: str = DEFAULT_BASELINE_DB_PATH
    cve_index_path: str = DEFAULT_CVE_INDEX_PATH
    enable_firewall: bool = True
    enable_monitor: bool = True
    enable_secure_boot: bool = True
    enable_secure_update: bool = True
    enable_secure_storage: bool = True
    enable_firmware_validation: bool = True
    enable_integrity_monitor: bool = True
    enable_safety_monitor: bool = True
    enable_tamper_detection: bool = True
    enable_vuln_scanner: bool = True
    enable_event_logger: bool = True
    enable_emergency_lockdown: bool = True
    enable_incident_response: bool = True
    enable_diagnostics: bool = True

    firewall: FirewallConfig = field(default_factory=FirewallConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    boot: BootConfig = field(default_factory=BootConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    lockdown: LockdownConfig = field(default_factory=LockdownConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _from_dict(cls_type: type, data: Dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if not hasattr(cls_type, "__dataclass_fields__"):
        return data
    field_names = set(cls_type.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls_type(**filtered)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> VehicleSecurityConfig:
    """Load a :class:`VehicleSecurityConfig` from YAML.

    Args:
        path: Path to a YAML file. If ``None`` or missing, defaults are used.
        overrides: Optional dict of overrides merged on top of the YAML data.

    Returns:
        A fully populated :class:`VehicleSecurityConfig`.
    """
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files")
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {path} must contain a mapping at top level")
        logger.info("loaded config from %s", path)
        data = loaded
    else:
        logger.debug("no config file provided; using defaults")

    if overrides:
        _deep_update(data, overrides)

    cfg = VehicleSecurityConfig()
    if not data:
        return cfg

    cfg.base_dir = data.get("base_dir", cfg.base_dir)
    cfg.log_path = data.get("log_path", cfg.log_path)
    cfg.baseline_db_path = data.get("baseline_db_path", cfg.baseline_db_path)
    cfg.cve_index_path = data.get("cve_index_path", cfg.cve_index_path)

    for flag in (
        "enable_firewall",
        "enable_monitor",
        "enable_secure_boot",
        "enable_secure_update",
        "enable_secure_storage",
        "enable_firmware_validation",
        "enable_integrity_monitor",
        "enable_safety_monitor",
        "enable_tamper_detection",
        "enable_vuln_scanner",
        "enable_event_logger",
        "enable_emergency_lockdown",
        "enable_incident_response",
        "enable_diagnostics",
    ):
        if flag in data:
            setattr(cfg, flag, bool(data[flag]))

    if "firewall" in data and isinstance(data["firewall"], dict):
        cfg.firewall = _from_dict(FirewallConfig, data["firewall"])
    if "storage" in data and isinstance(data["storage"], dict):
        cfg.storage = _from_dict(StorageConfig, data["storage"])
    if "monitor" in data and isinstance(data["monitor"], dict):
        cfg.monitor = _from_dict(MonitorConfig, data["monitor"])
    if "boot" in data and isinstance(data["boot"], dict):
        cfg.boot = _from_dict(BootConfig, data["boot"])
    if "update" in data and isinstance(data["update"], dict):
        cfg.update = _from_dict(UpdateConfig, data["update"])
    if "lockdown" in data and isinstance(data["lockdown"], dict):
        cfg.lockdown = _from_dict(LockdownConfig, data["lockdown"])

    return cfg


def save_config(cfg: VehicleSecurityConfig, path: str) -> None:
    """Persist a config back to YAML (used by diagnostics/admin tooling)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to save config files")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, default_flow_style=False, sort_keys=False)
    logger.info("config written to %s", path)
