'''Profile preferences (JSON) under LOCALAPPDATA\\BasePilot, next to player_list.json.'''
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from app.utils.common import ensure_dir, get_user_app_data_dir

EARTHQUAKE_METHOD_CURVE = 'Curve Placement'
EARTHQUAKE_METHOD_RANDOM = 'Random Placement'
EARTHQUAKE_METHOD_OPTIONS = (EARTHQUAKE_METHOD_CURVE, EARTHQUAKE_METHOD_RANDOM)
SETTINGS_FILENAME = 'settings.json'
WALL_UPGRADE_THRESHOLD_M_DEFAULT = 3
WALL_UPGRADE_THRESHOLD_M_MAX = 20
RESERVE_BUILDERS_DEFAULT = 1
RESERVE_BUILDERS_MAX = 5
UPGRADE_ORDER_PRICIEST = 'priciest'
UPGRADE_ORDER_CHEAPEST = 'cheapest'
UPGRADE_ORDER_OPTIONS = (UPGRADE_ORDER_PRICIEST, UPGRADE_ORDER_CHEAPEST)

LOOT_MATCH_ANY = 'any'
LOOT_MATCH_ALL = 'all'
LOOT_MATCH_OPTIONS = (LOOT_MATCH_ANY, LOOT_MATCH_ALL)

MIN_GOLD_K_DEFAULT = 0
MIN_GOLD_K_MAX = 2500
MIN_ELIXIR_K_DEFAULT = 0
MIN_ELIXIR_K_MAX = 2500
MIN_DARK_ELIXIR_DEFAULT = 0
MIN_DARK_ELIXIR_MAX = 25000
MAX_NEXT_SKIPS_DEFAULT = 50
MAX_NEXT_SKIPS_MAX = 200

BATTLE_END_DELAY_DEFAULT = 20
BATTLE_END_DELAY_MAX = 180

@dataclass
class ProfileSettings:
    earthquake_method: str = EARTHQUAKE_METHOD_CURVE
    wall_upgrade_threshold_m: int = WALL_UPGRADE_THRESHOLD_M_DEFAULT
    reserve_builders: int = RESERVE_BUILDERS_DEFAULT
    upgrade_order: str = UPGRADE_ORDER_PRICIEST
    min_gold_k: int = MIN_GOLD_K_DEFAULT
    min_elixir_k: int = MIN_ELIXIR_K_DEFAULT
    min_dark_elixir: int = MIN_DARK_ELIXIR_DEFAULT
    max_next_skips: int = MAX_NEXT_SKIPS_DEFAULT
    battle_end_delay_s: int = BATTLE_END_DELAY_DEFAULT
    loot_match_condition: str = LOOT_MATCH_ANY
    filter_dark_elixir: bool = False


def get_settings_path():
    dest = get_user_app_data_dir() / SETTINGS_FILENAME
    ensure_dir(dest.parent)
    return dest


def _normalize_earthquake_method(raw):
    if raw == EARTHQUAKE_METHOD_RANDOM or raw == EARTHQUAKE_METHOD_CURVE:
        return str(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == 'random placement':
            return EARTHQUAKE_METHOD_RANDOM
        if s == 'curve placement':
            return EARTHQUAKE_METHOD_CURVE
    return EARTHQUAKE_METHOD_CURVE


def _normalize_wall_threshold_m(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return WALL_UPGRADE_THRESHOLD_M_DEFAULT
    return max(0, min(WALL_UPGRADE_THRESHOLD_M_MAX, value))


def _normalize_upgrade_order(raw):
    if isinstance(raw, str) and raw.strip().lower() in UPGRADE_ORDER_OPTIONS:
        return raw.strip().lower()
    return UPGRADE_ORDER_PRICIEST


def _normalize_reserve_builders(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return RESERVE_BUILDERS_DEFAULT
    return max(0, min(RESERVE_BUILDERS_MAX, value))


def _normalize_min_gold_k(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MIN_GOLD_K_DEFAULT
    return max(0, min(MIN_GOLD_K_MAX, value))


def _normalize_min_elixir_k(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MIN_ELIXIR_K_DEFAULT
    return max(0, min(MIN_ELIXIR_K_MAX, value))


def _normalize_min_dark_elixir(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MIN_DARK_ELIXIR_DEFAULT
    return max(0, min(MIN_DARK_ELIXIR_MAX, value))


def _normalize_max_next_skips(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MAX_NEXT_SKIPS_DEFAULT
    return max(1, min(MAX_NEXT_SKIPS_MAX, value))


def _normalize_battle_end_delay_s(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return BATTLE_END_DELAY_DEFAULT
    return max(0, min(BATTLE_END_DELAY_MAX, value))


def _normalize_loot_match_condition(raw):
    if isinstance(raw, str) and raw.strip().lower() in LOOT_MATCH_OPTIONS:
        return raw.strip().lower()
    return LOOT_MATCH_ANY


def load_profile_settings():
    path = get_settings_path()
    if not path.is_file():
        return ProfileSettings()

    try:
        raw = json.loads(path.read_text(encoding = 'utf-8'))
        if not isinstance(raw, dict):
            return ProfileSettings()
        return ProfileSettings(
            earthquake_method = _normalize_earthquake_method(raw.get('earthquake_method')),
            wall_upgrade_threshold_m = _normalize_wall_threshold_m(raw.get('wall_upgrade_threshold_m', WALL_UPGRADE_THRESHOLD_M_DEFAULT)),
            reserve_builders = _normalize_reserve_builders(raw.get('reserve_builders', RESERVE_BUILDERS_DEFAULT)),
            upgrade_order = _normalize_upgrade_order(raw.get('upgrade_order', UPGRADE_ORDER_PRICIEST)),
            min_gold_k = _normalize_min_gold_k(raw.get('min_gold_k', MIN_GOLD_K_DEFAULT)),
            min_elixir_k = _normalize_min_elixir_k(raw.get('min_elixir_k', MIN_ELIXIR_K_DEFAULT)),
            min_dark_elixir = _normalize_min_dark_elixir(raw.get('min_dark_elixir', MIN_DARK_ELIXIR_DEFAULT)),
            max_next_skips = _normalize_max_next_skips(raw.get('max_next_skips', MAX_NEXT_SKIPS_DEFAULT)),
            battle_end_delay_s = _normalize_battle_end_delay_s(raw.get('battle_end_delay_s', BATTLE_END_DELAY_DEFAULT)),
            loot_match_condition = _normalize_loot_match_condition(raw.get('loot_match_condition', LOOT_MATCH_ANY)),
            filter_dark_elixir = bool(raw.get('filter_dark_elixir', False)))
    except (json.JSONDecodeError, OSError):
        return ProfileSettings()


def save_profile_settings(settings):
    path = get_settings_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    normalized = ProfileSettings(
        earthquake_method = _normalize_earthquake_method(settings.earthquake_method),
        wall_upgrade_threshold_m = _normalize_wall_threshold_m(getattr(settings, 'wall_upgrade_threshold_m', WALL_UPGRADE_THRESHOLD_M_DEFAULT)),
        reserve_builders = _normalize_reserve_builders(getattr(settings, 'reserve_builders', RESERVE_BUILDERS_DEFAULT)),
        upgrade_order = _normalize_upgrade_order(getattr(settings, 'upgrade_order', UPGRADE_ORDER_PRICIEST)),
        min_gold_k = _normalize_min_gold_k(getattr(settings, 'min_gold_k', MIN_GOLD_K_DEFAULT)),
        min_elixir_k = _normalize_min_elixir_k(getattr(settings, 'min_elixir_k', MIN_ELIXIR_K_DEFAULT)),
        min_dark_elixir = _normalize_min_dark_elixir(getattr(settings, 'min_dark_elixir', MIN_DARK_ELIXIR_DEFAULT)),
        max_next_skips = _normalize_max_next_skips(getattr(settings, 'max_next_skips', MAX_NEXT_SKIPS_DEFAULT)),
        battle_end_delay_s = _normalize_battle_end_delay_s(getattr(settings, 'battle_end_delay_s', BATTLE_END_DELAY_DEFAULT)),
        loot_match_condition = _normalize_loot_match_condition(getattr(settings, 'loot_match_condition', LOOT_MATCH_ANY)),
        filter_dark_elixir = bool(getattr(settings, 'filter_dark_elixir', False)))
    payload = asdict(normalized)
    path.write_text(json.dumps(payload, indent = 2), encoding = 'utf-8')
