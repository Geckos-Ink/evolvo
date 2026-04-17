"""Runtime acceleration settings for Evolvo execution backends.

Edit these values to tune GPU/CPU execution flow globally.
Environment variables with the same intent can still override defaults.
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(max(minimum, int(default)))
    try:
        parsed = int(raw)
    except Exception:
        return int(max(minimum, int(default)))
    return int(max(minimum, parsed))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(max(minimum, min(maximum, float(default))))
    try:
        parsed = float(raw)
    except Exception:
        parsed = float(default)
    return float(max(minimum, min(maximum, parsed)))


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = os.environ.get(name)
    candidate = str(default if raw is None else raw).strip().lower()
    if candidate not in allowed:
        return str(default).strip().lower()
    return candidate


# High-level backend selection.
DEFAULT_COMPUTE_BACKEND = _env_choice(
    "EVOLVO_EXECUTOR_BACKEND",
    "auto",
    {"auto", "cpu", "kompute", "kompute-sim"},
)
DEFAULT_KOMPUTE_RUNTIME_MODE = _env_choice(
    "EVOLVO_KOMPUTE_RUNTIME_MODE",
    "native",
    {"native", "simulated", "auto"},
)

# Kompute behavior/safety.
DEFAULT_KOMPUTE_WARN_ON_FALLBACK = _env_bool("EVOLVO_KOMPUTE_WARN_ON_FALLBACK", True)
DEFAULT_KOMPUTE_FAIL_HARD = _env_bool("EVOLVO_KOMPUTE_FAIL_HARD", False)
DEFAULT_KOMPUTE_KEEP_VRAM_STATE = _env_bool("EVOLVO_KOMPUTE_KEEP_VRAM_STATE", True)

# Coverage policy to avoid costly hybrid sync traffic when GPU coverage is weak.
DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_COUNT = _env_int(
    "EVOLVO_KOMPUTE_MIN_NATIVE_STAGE_COUNT",
    1,
    minimum=0,
)
DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_SHARE = _env_float(
    "EVOLVO_KOMPUTE_MIN_NATIVE_STAGE_SHARE",
    0.0,
    minimum=0.0,
    maximum=1.0,
)
DEFAULT_KOMPUTE_MAX_UNSUPPORTED_SHARE = _env_float(
    "EVOLVO_KOMPUTE_MAX_UNSUPPORTED_SHARE",
    1.0,
    minimum=0.0,
    maximum=1.0,
)
DEFAULT_KOMPUTE_MAX_UNSUPPORTED_COUNT = _env_int(
    "EVOLVO_KOMPUTE_MAX_UNSUPPORTED_COUNT",
    -1,
    minimum=-1,
)
DEFAULT_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE = _env_bool(
    "EVOLVO_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE",
    False,
)

# Native shader family gates (debug/perf tuning).
DEFAULT_KOMPUTE_NATIVE_ENABLE_DECIMAL = _env_bool(
    "EVOLVO_KOMPUTE_NATIVE_ENABLE_DECIMAL",
    True,
)
DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE = _env_bool(
    "EVOLVO_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE",
    True,
)
DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC = _env_bool(
    "EVOLVO_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC",
    True,
)
DEFAULT_KOMPUTE_NATIVE_ENABLE_LIST_QUERY = _env_bool(
    "EVOLVO_KOMPUTE_NATIVE_ENABLE_LIST_QUERY",
    True,
)


__all__ = [
    "DEFAULT_COMPUTE_BACKEND",
    "DEFAULT_KOMPUTE_RUNTIME_MODE",
    "DEFAULT_KOMPUTE_WARN_ON_FALLBACK",
    "DEFAULT_KOMPUTE_FAIL_HARD",
    "DEFAULT_KOMPUTE_KEEP_VRAM_STATE",
    "DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_COUNT",
    "DEFAULT_KOMPUTE_MIN_NATIVE_STAGE_SHARE",
    "DEFAULT_KOMPUTE_MAX_UNSUPPORTED_SHARE",
    "DEFAULT_KOMPUTE_MAX_UNSUPPORTED_COUNT",
    "DEFAULT_KOMPUTE_FORCE_CPU_ON_PARTIAL_COVERAGE",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_DECIMAL",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_COMPARE",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_BOOLEAN_LOGIC",
    "DEFAULT_KOMPUTE_NATIVE_ENABLE_LIST_QUERY",
]
