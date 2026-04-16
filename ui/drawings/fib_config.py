from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RETRACEMENT_PRESETS = [0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8]
DEFAULT_EXTENSION_PRESETS = [0.618, 1.0, 1.272, 1.618, 2.0]

KEY_RETRACEMENT_ENABLED = "drawing/fib/retracement_enabled"
KEY_RETRACEMENT_CUSTOM = "drawing/fib/retracement_custom"
KEY_EXTENSION_ENABLED = "drawing/fib/extension_enabled"
KEY_EXTENSION_CUSTOM = "drawing/fib/extension_custom"


def _normalize_levels(levels: list[float]) -> list[float]:
    normalized: list[float] = []
    for level in levels:
        value = float(level)
        if value < 0:
            raise ValueError("Fibonacci levels must be non-negative")
        normalized.append(value)
    return sorted(set(normalized))


def _coerce_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if value == "":
        return []
    return [float(value)]


def merge_fib_levels(enabled_levels: list[float], custom_levels_text: str) -> list[float]:
    custom_levels: list[float] = []
    for raw in custom_levels_text.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            custom_levels.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid Fibonacci level: {token}") from exc
    return _normalize_levels([*enabled_levels, *custom_levels])


@dataclass(frozen=True)
class FibLevelsConfig:
    enabled_levels: list[float]
    custom_levels_text: str = ""

    @property
    def effective_levels(self) -> list[float]:
        return merge_fib_levels(self.enabled_levels, self.custom_levels_text)


@dataclass(frozen=True)
class FibSettings:
    retracement: FibLevelsConfig
    extension: FibLevelsConfig


def default_fib_settings() -> FibSettings:
    return FibSettings(
        retracement=FibLevelsConfig(enabled_levels=list(DEFAULT_RETRACEMENT_PRESETS)),
        extension=FibLevelsConfig(enabled_levels=list(DEFAULT_EXTENSION_PRESETS)),
    )


def save_fib_settings(settings, fib_settings: FibSettings) -> None:
    settings.setValue(KEY_RETRACEMENT_ENABLED, fib_settings.retracement.enabled_levels)
    settings.setValue(KEY_RETRACEMENT_CUSTOM, fib_settings.retracement.custom_levels_text)
    settings.setValue(KEY_EXTENSION_ENABLED, fib_settings.extension.enabled_levels)
    settings.setValue(KEY_EXTENSION_CUSTOM, fib_settings.extension.custom_levels_text)
    settings.sync()


def load_fib_settings(settings) -> FibSettings:
    defaults = default_fib_settings()
    retracement_enabled = _coerce_float_list(settings.value(KEY_RETRACEMENT_ENABLED, defaults.retracement.enabled_levels))
    extension_enabled = _coerce_float_list(settings.value(KEY_EXTENSION_ENABLED, defaults.extension.enabled_levels))
    return FibSettings(
        retracement=FibLevelsConfig(
            enabled_levels=retracement_enabled or list(DEFAULT_RETRACEMENT_PRESETS),
            custom_levels_text=settings.value(KEY_RETRACEMENT_CUSTOM, "", type=str),
        ),
        extension=FibLevelsConfig(
            enabled_levels=extension_enabled or list(DEFAULT_EXTENSION_PRESETS),
            custom_levels_text=settings.value(KEY_EXTENSION_CUSTOM, "", type=str),
        ),
    )
