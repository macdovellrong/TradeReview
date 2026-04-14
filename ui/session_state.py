from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SETTINGS_GROUP = "session"
KEY_DB_PATH = f"{SETTINGS_GROUP}/db_path"
KEY_CENTER_TIME = f"{SETTINGS_GROUP}/center_time"


@dataclass(frozen=True)
class SessionState:
    db_path: str
    center_time: pd.Timestamp


def save_session_state(settings, state: SessionState) -> None:
    settings.setValue(KEY_DB_PATH, state.db_path)
    settings.setValue(KEY_CENTER_TIME, pd.Timestamp(state.center_time).isoformat())
    settings.sync()


def load_session_state(settings) -> SessionState | None:
    db_path = settings.value(KEY_DB_PATH, "", type=str)
    center_time_raw = settings.value(KEY_CENTER_TIME, "", type=str)
    if not db_path or not center_time_raw:
        return None

    try:
        center_time = pd.Timestamp(center_time_raw)
    except Exception:
        return None

    if pd.isna(center_time):
        return None

    return SessionState(db_path=db_path, center_time=center_time)
