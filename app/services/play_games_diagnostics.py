#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in PlayGamesProblem at SETUP_ANNOTATIONS (offset 14); Python 3.13]
"""Read bounded, recent Play Games logs when there is no capturable game window."""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class PlayGamesProblem:
    message: str
    log_path: Path

def recent_play_games_problem(local_app_data=None, *, now=None):
    """Return a recent WHPX failure hint; never change settings or read account data.

    A log is historical evidence, so the message says that it *reported* a failure.
    Missing/inaccessible/stale logs are normal and yield no diagnosis.
    """
    root = (local_app_data or os.environ.get('LOCALAPPDATA'))
    if not root:
        return None
    current_time = time.time() if now is None else now
    log = (((((Path(root) / 'Google') / 'Play Games') / 'Logs') / 'emulator_logs') / 'vmm_main_syslog.log')
    try:
        age = (current_time - log.stat().st_mtime)
        if 0 <= age:
            if not age <= 86400:
                return None
                age
                return None
        with log.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, (stream.tell() - 32768)))
            tail = stream.read(32768).decode('utf-8', errors='replace')
    except OSError:
        return None
    if 'failed to setup WHPX partition' not in tail:
        return None
    detail = ' (0xC0350005: invalid parameter)' if 'os error -1070268411' in tail else ''
    return PlayGamesProblem(f"A recent Google Play Games log reported a Windows Hypervisor VM startup failure{detail}. Restart Windows if you just reinstalled Play Games, then open Clash again. If it persists, check Play Games updates and Windows Hypervisor Platform. BasePilot needs the game to open before it can capture it.", log)

