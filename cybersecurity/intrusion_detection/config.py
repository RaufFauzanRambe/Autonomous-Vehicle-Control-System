"""Configuration dataclasses and YAML loader for the intrusion detection module.

The :class:`IDSConfig` dataclass aggregates all configuration knobs used by the
various detectors, the alert manager, the response engine and the forensic
tools. The :func:`load_config` helper parses a YAML file (or dictionary) into a
validated :class:`IDSConfig` instance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # PyYAML is optional at runtime; we degrade gracefully if absent.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only when PyYAML missing
    yaml = None  # type: ignore

from .constants import (
    DEFAULT_ALERT_TTL_SEC,
    DEFAULT_BASELINE_TRAINING_SAMPLES,
    DEFAULT_CAN_INTERFACE,
    DEFAULT_DEDUP_WINDOW_SEC,
    DEFAULT_ESCALATION_TIME_SEC,
    DEFAULT_INCIDENT_RETENTION_DAYS,
    DEFAULT_NETWORK_INTERFACE,
    DEFAULT_PCAP_SNAPLEN,
    DEFAULT_PCAP_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Configuration for packet / CAN frame / log capture."""

    can_interface: str = DEFAULT_CAN_INTERFACE
    network_interface: str = DEFAULT_NETWORK_INTERFACE
    pcap_snaplen: int = DEFAULT_PCAP_SNAPLEN
    pcap_timeout_ms: int = DEFAULT_PCAP_TIMEOUT_MS
    promiscuous: bool = True
    bpf_filter: str = ""
    can_bitrate: int = 500_000
    log_paths: List[str] = field(
        default_factory=lambda: [
            "/var/log/syslog",
            "/var/log/auth.log",
            "/var/log/journal",
        ]
    )


@dataclass
class RuleConfig:
    """Configuration for signatures, rules, and YARA sources."""

    signature_db_path: str = "/etc/avcs/ids/signatures.json"
    yara_rules_dir: str = "/etc/avcs/ids/yara_rules"
    rule_paths: List[str] = field(
        default_factory=lambda: ["/etc/avcs/ids/rules/default.rules"]
    )
    auto_update: bool = True
    update_interval_sec: int = 3600


@dataclass
class AnomalyConfig:
    """Configuration for anomaly / behavioral baselining."""

    baseline_model_path: str = "/var/lib/avcs/ids/baseline.pkl"
    training_samples: int = DEFAULT_BASELINE_TRAINING_SAMPLES
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    retrain_interval_sec: int = 6 * 3600
    isolation_forest_contamination: float = 0.05


@dataclass
class AlertConfig:
    """Configuration for the alert manager."""

    retention_sec: int = DEFAULT_ALERT_TTL_SEC
    dedup_window_sec: float = DEFAULT_DEDUP_WINDOW_SEC
    escalation_sec: int = DEFAULT_ESCALATION_TIME_SEC
    enable_correlation: bool = True
    correlation_window_sec: float = 120.0
    siem_webhook_url: Optional[str] = None
    siem_api_key: Optional[str] = None


@dataclass
class ResponseConfig:
    """Configuration for automated response actions."""

    enabled: bool = True
    dry_run: bool = False
    max_actions_per_min: int = 30
    allowed_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    cooldown_sec: int = 60


@dataclass
class ForensicConfig:
    """Configuration for forensic evidence collection."""

    evidence_dir: str = "/var/lib/avcs/ids/evidence"
    max_evidence_size_mb: int = 4096
    compress_evidence: bool = True
    hash_algorithm: str = "sha256"
    retain_days: int = 90


@dataclass
class IncidentLogConfig:
    """Configuration for the tamper-evident incident log."""

    log_path: str = "/var/lib/avcs/ids/incidents.chain"
    hash_algorithm: str = "sha256"
    retention_days: int = DEFAULT_INCIDENT_RETENTION_DAYS
    sign_entries: bool = False
    signing_key_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


@dataclass
class IDSConfig:
    """Top-level configuration for the entire intrusion detection subsystem."""

    instance_id: str = "avcs-ids-default"
    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    forensic: ForensicConfig = field(default_factory=ForensicConfig)
    incident_log: IncidentLogConfig = field(default_factory=IncidentLogConfig)

    # ------------------------------------------------------------------
    # Validation & (de)serialization
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the configuration; return a list of human-readable errors."""
        errors: List[str] = []
        if not self.instance_id:
            errors.append("instance_id must be a non-empty string")
        if self.anomaly.zscore_threshold <= 0:
            errors.append("anomaly.zscore_threshold must be > 0")
        if self.anomaly.iqr_multiplier <= 0:
            errors.append("anomaly.iqr_multiplier must be > 0")
        if self.anomaly.isolation_forest_contamination <= 0 or \
                self.anomaly.isolation_forest_contamination >= 1:
            errors.append("anomaly.isolation_forest_contamination must be in (0, 1)")
        if self.alerts.dedup_window_sec < 0:
            errors.append("alerts.dedup_window_sec must be >= 0")
        if self.alerts.escalation_sec < 0:
            errors.append("alerts.escalation_sec must be >= 0")
        if self.response.max_actions_per_min <= 0:
            errors.append("response.max_actions_per_min must be > 0")
        if self.forensic.max_evidence_size_mb <= 0:
            errors.append("forensic.max_evidence_size_mb must be > 0")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dict (suitable for YAML/JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IDSConfig":
        """Construct an :class:`IDSConfig` from a (possibly partial) dict."""
        if not isinstance(data, dict):
            raise TypeError("IDSConfig.from_dict requires a dict")

        def _sub(dclass, sub_data: Optional[Dict[str, Any]]):
            if not sub_data:
                return dclass()
            valid = {k: v for k, v in sub_data.items() if k in dclass.__dataclass_fields__}
            return dclass(**valid)

        return cls(
            instance_id=data.get("instance_id", "avcs-ids-default"),
            enabled=data.get("enabled", True),
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
            capture=_sub(CaptureConfig, data.get("capture")),
            rules=_sub(RuleConfig, data.get("rules")),
            anomaly=_sub(AnomalyConfig, data.get("anomaly")),
            alerts=_sub(AlertConfig, data.get("alerts")),
            response=_sub(ResponseConfig, data.get("response")),
            forensic=_sub(ForensicConfig, data.get("forensic")),
            incident_log=_sub(IncidentLogConfig, data.get("incident_log")),
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(source: Any) -> IDSConfig:
    """Load an :class:`IDSConfig` from a path, file-like object, or dict.

    Parameters
    ----------
    source:
        Either a path string / :class:`~pathlib.Path` to a YAML file, an
        already-opened file-like object, or a dictionary.
    """
    data: Dict[str, Any]
    if isinstance(source, IDSConfig):
        return source
    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load config from a YAML file; "
                "install it or pass a dict instead"
            )
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif hasattr(source, "read"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML from a file object")
        data = yaml.safe_load(source.read()) or {}
    else:
        raise TypeError(f"Unsupported config source type: {type(source).__name__}")

    config = IDSConfig.from_dict(data)
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error("Config validation error: %s", err)
        raise ValueError(f"Invalid IDS configuration: {'; '.join(errors)}")
    logger.info("Loaded IDS configuration for instance '%s'", config.instance_id)
    return config


def default_config() -> IDSConfig:
    """Return a default, valid :class:`IDSConfig` instance."""
    return IDSConfig()


__all__ = [
    "CaptureConfig",
    "RuleConfig",
    "AnomalyConfig",
    "AlertConfig",
    "ResponseConfig",
    "ForensicConfig",
    "IncidentLogConfig",
    "IDSConfig",
    "load_config",
    "default_config",
]
