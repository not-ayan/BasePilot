#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in UpgradePolicy at SETUP_ANNOTATIONS (offset 14); Python 3.13]
"""Choose funded upgrades while respecting the Town Hall policy and builder reserve.

Dark-elixir jobs get first claim. Other jobs follow the configured priciest or
cheapest order. Rusher prioritizes a funded Town Hall; Maxer excludes it. Unknown
balances only exclude purchases in that currency, and gold keeps its attack buffer.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
from app.core.upgrade_menu import RESOURCE_DARK, RESOURCE_ELIXIR, RESOURCE_GOLD, SECTION_IN_PROGRESS, SECTION_OTHER, SECTION_SUGGESTED, UpgradeRow, is_town_hall_row, is_wall_row
from app.utils.logger import setup_logger
logger = setup_logger('UpgradePolicy')
MODE_MAXER = 'maxer'
MODE_RUSHER = 'rusher'
ORDER_PRICIEST = 'priciest'
ORDER_CHEAPEST = 'cheapest'
UPGRADE_ORDERS = (ORDER_PRICIEST, ORDER_CHEAPEST)
@dataclass(frozen=True)
class UpgradePolicy:
    mode: str = MODE_MAXER
    reserve_builders: int = 1
    gold_attack_buffer: int = 50000
    order: str = ORDER_PRICIEST

def _available_for(resource, resources, policy):
    gold, elixir, dark = resources
    if resource == RESOURCE_GOLD:
        if gold is not None:
            return max(0, (gold - policy.gold_attack_buffer))
        return 0
    if resource == RESOURCE_ELIXIR:
        if elixir is not None:
            return elixir
        return 0
    if resource == RESOURCE_DARK:
        if dark is not None:
            return dark
        return 0
    return 0

def can_pay(row, resources, policy):
    """True when the row has a parsed cost/resource and the OCR'd resources cover it
    (after the gold attack buffer)."""
    if row.cost is None or row.resource is None:
        return False
    return row.cost <= _available_for(row.resource, resources, policy)

def choose_upgrade(rows, resources, free_builders, policy):
    """Pick the upgrade to start now, or ``None``.

    ``rows``: merged :class:`UpgradeRow` list from the builder menu scan.
    ``resources``: OCR'd ``(gold, elixir, dark)`` HUD triplet.
    ``free_builders``: from the builder chip; ``None`` means unknown → do nothing.
    """
    if free_builders is None or free_builders <= policy.reserve_builders:
        return None
    candidates = [r for r in rows if r.section != SECTION_IN_PROGRESS if r.affordable if not (is_wall_row(r)) if can_pay(r, resources, policy)]
    if not candidates:
        return None
    th_rows = [r for r in candidates if is_town_hall_row(r)]
    if policy.mode == MODE_RUSHER and th_rows:
        pick = min(th_rows, key=(lambda r: r.cost))
        logger.info('Policy(rusher): Town Hall affordable — picking it (%s, cost %s)', pick.label, pick.cost)
        return pick
    if policy.mode != MODE_RUSHER:
        candidates = [r for r in candidates if not (is_town_hall_row(r))]
    if not candidates:
        return None
    de_rows = [r for r in candidates if r.resource == RESOURCE_DARK]
    pool = de_rows or candidates
    chooser = min if policy.order == ORDER_CHEAPEST else max
    pick = chooser(pool, key=(lambda r: r.cost))
    logger.info('Policy(%s/%s): picking %s (cost %s %s)', policy.mode, policy.order, pick.label, pick.cost, pick.resource)
    return pick

