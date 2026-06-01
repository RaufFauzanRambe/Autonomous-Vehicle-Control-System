"""
Version information for the Autonomous Vehicle Control System AI Engine.

Provides package version, API version, dependency version checks,
and compatibility validation utilities.

Usage:
    from utils.version import __version__, get_version_info, check_compatibility

    print(__version__)
    info = get_version_info()
    check_compatibility(min_python=(3, 8), min_numpy="1.21.0")
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Version definitions
# ---------------------------------------------------------------------------

# Semantic versioning: MAJOR.MINOR.PATCH[-PRERELEASE]+BUILD
__version__: str = "2.4.1"
__version_tuple__: Tuple[int, int, int] = (2, 4, 1)

# API version – incremented when the public API changes incompatibly
API_VERSION: str = "v2"
API_VERSION_MAJOR: int = 2
API_VERSION_MINOR: int = 1

# Build metadata
BUILD_COMMIT: str = "unknown"
BUILD_BRANCH: str = "unknown"
BUILD_DATE: str = "unknown"


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

class Version:
    """Semantic version parser and comparator.

    Follows SemVer 2.0.0 specification: MAJOR.MINOR.PATCH[-PRERELEASE]

    Usage::

        v1 = Version("2.4.1")
        v2 = Version("2.5.0")
        assert v1 < v2
        assert v1.is_compatible(v2, policy="minor")
    """

    def __init__(self, version_string: str) -> None:
        self._raw = version_string.strip()
        self.major: int = 0
        self.minor: int = 0
        self.patch: int = 0
        self.prerelease: str = ""
        self._parse(self._raw)

    def _parse(self, version_string: str) -> None:
        """Parse a version string into components."""
        # Strip leading 'v' if present
        s = version_string.lstrip("vV")

        # Split off prerelease
        if "-" in s:
            s, self.prerelease = s.split("-", 1)

        # Split into major.minor.patch
        parts = s.split(".")
        try:
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            raise ValueError(f"Invalid version string: {version_string}")

    @property
    def tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def is_prerelease(self) -> bool:
        return bool(self.prerelease)

    def is_compatible(self, other: "Version", policy: str = "minor") -> bool:
        """Check compatibility with another version.

        Args:
            other: Version to compare against.
            policy: Compatibility policy:
                - ``"exact"``: versions must match exactly.
                - ``"patch"``: MAJOR and MINOR must match.
                - ``"minor"``: MAJOR must match (default).
                - ``"major"``: always compatible.

        Returns:
            True if compatible according to the policy.
        """
        if policy == "exact":
            return self.tuple == other.tuple and self.prerelease == other.prerelease
        elif policy == "patch":
            return self.major == other.major and self.minor == other.minor
        elif policy == "minor":
            return self.major == other.major
        elif policy == "major":
            return True
        else:
            raise ValueError(f"Unknown compatibility policy: {policy}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.tuple == other.tuple and self.prerelease == other.prerelease

    def __lt__(self, other: "Version") -> bool:
        if self.tuple != other.tuple:
            return self.tuple < other.tuple
        # Prerelease versions are lower than release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return self.prerelease < other.prerelease

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.tuple, self.prerelease))

    def __repr__(self) -> str:
        return f"Version('{self}')"

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        return s


# ---------------------------------------------------------------------------
# Dependency version checking
# ---------------------------------------------------------------------------

@dataclass
class DependencyInfo:
    """Version information for a single dependency."""
    name: str
    installed_version: str
    minimum_version: str
    is_compatible: bool


def _get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a Python package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except (importlib.metadata.PackageNotFoundError, ImportError):
        pass
    # Fallback: try importing and checking __version__
    try:
        mod = __import__(package_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def check_dependency(
    package_name: str,
    min_version: str,
    policy: str = "minor",
) -> DependencyInfo:
    """Check if a dependency meets the minimum version requirement.

    Args:
        package_name: Name of the Python package.
        min_version: Minimum required version string.
        policy: Compatibility policy for the check.

    Returns:
        :class:`DependencyInfo` with compatibility status.
    """
    installed = _get_package_version(package_name) or "0.0.0"
    try:
        installed_v = Version(installed)
        required_v = Version(min_version)
        is_compat = installed_v >= required_v
    except ValueError:
        is_compat = False

    return DependencyInfo(
        name=package_name,
        installed_version=installed,
        minimum_version=min_version,
        is_compatible=is_compat,
    )


# ---------------------------------------------------------------------------
# System compatibility check
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityReport:
    """Full compatibility check report."""
    python_compatible: bool
    python_version: str
    python_min: str
    dependencies: List[DependencyInfo]
    platform_info: Dict[str, str]
    all_compatible: bool

    def summary(self) -> str:
        lines = [
            f"Compatibility Report",
            f"{'=' * 50}",
            f"Python: {self.python_version} (min: {self.python_min}) {'OK' if self.python_compatible else 'FAIL'}",
            f"Platform: {self.platform_info.get('system', 'unknown')}",
            f"",
            f"Dependencies:",
        ]
        for dep in self.dependencies:
            status = "OK" if dep.is_compatible else "FAIL"
            lines.append(f"  {dep.name}: {dep.installed_version} (min: {dep.minimum_version}) [{status}]")
        lines.append(f"\nOverall: {'COMPATIBLE' if self.all_compatible else 'INCOMPATIBLE'}")
        return "\n".join(lines)


# Minimum dependency versions for the AV AI engine
_MIN_DEPENDENCIES = {
    "numpy": "1.21.0",
    "scipy": "1.7.0",
    "opencv-python": "4.5.0",
}


def check_compatibility(
    min_python: Tuple[int, ...] = (3, 8),
    extra_dependencies: Optional[Dict[str, str]] = None,
) -> CompatibilityReport:
    """Perform a full compatibility check of the runtime environment.

    Args:
        min_python: Minimum required Python version as a tuple.
        extra_dependencies: Additional dependency version requirements.

    Returns:
        A :class:`CompatibilityReport` with detailed results.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_compat = sys.version_info >= min_python

    deps = {**_MIN_DEPENDENCIES, **(extra_dependencies or {})}
    dep_results = [
        check_dependency(name, version)
        for name, version in deps.items()
    ]

    all_compat = python_compat and all(d.is_compatible for d in dep_results)

    return CompatibilityReport(
        python_compatible=python_compat,
        python_version=python_version,
        python_min=".".join(str(x) for x in min_python),
        dependencies=dep_results,
        platform_info={
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
        },
        all_compatible=all_compat,
    )


# ---------------------------------------------------------------------------
# Version info aggregation
# ---------------------------------------------------------------------------

def get_version_info() -> Dict[str, Any]:
    """Return a comprehensive version information dictionary.

    Includes package version, API version, Python version, and key
    dependency versions.
    """
    info: Dict[str, Any] = {
        "package_version": __version__,
        "package_version_tuple": __version_tuple__,
        "api_version": API_VERSION,
        "api_version_detail": f"{API_VERSION_MAJOR}.{API_VERSION_MINOR}",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "build": {
            "commit": BUILD_COMMIT,
            "branch": BUILD_BRANCH,
            "date": BUILD_DATE,
        },
        "dependencies": {},
    }

    # Collect dependency versions
    for dep in _MIN_DEPENDENCIES:
        version = _get_package_version(dep)
        info["dependencies"][dep] = version or "not installed"

    return info


def format_version_banner() -> str:
    """Return a human-readable version banner string."""
    info = get_version_info()
    return (
        f"Autonomous Vehicle Control System - AI Engine\n"
        f"  Version:  {info['package_version']}\n"
        f"  API:      {info['api_version_detail']}\n"
        f"  Python:   {info['python_version']}\n"
        f"  Platform: {info['platform']}"
    )
