"""Confirmed raid earnings, independent of storage balances and purchases.

Only the battle result's central 'You got' panel is an earnings source. The
ledger records each bot-started raid once, preserves unknown resources, and
writes an append-only local history so totals can be checked after a run.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from uuid import uuid4
import cv2
from app.core.ui_navigation import find_ui_label
from app.services.vision import VisionService
from app.utils.common import get_template_path
from app.utils.logger import setup_logger
logger = setup_logger('RaidLoot')
RESOURCE_NAMES = ('gold', 'elixir', 'dark_elixir')
UNKNOWN = (None, None, None)
RESULT_LIMITS = (50000000, 50000000, 1000000)

def _currency_present(crop, index):
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bounds = (((12, 90, 110), (42, 255, 255)), ((125, 80, 80), (175, 255, 255)), ((115, 20, 15), (179, 255, 190)))
    lower, upper = bounds[index]
    return ((cv2.countNonZero(cv2.inRange(hsv, lower, upper)) / crop.shape[0]) / crop.shape[1]) >= 0.12

def _read_amount(crop, limit, vision):
    if crop.size == 0:
        return None
    votes = []
    aliases = str.maketrans({'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'})
    for floor, scale in ((210, 3), (180, 4), (210, 4)):
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ink = cv2.inRange(big, (floor, floor, floor), (255, 255, 255))
        padded = cv2.copyMakeBorder((255 - ink), 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        words = vision.find_words_ocr(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), preprocess=False, tesseract_config='--psm 7')
        if not words:
            continue
        if any((((not math.isfinite(float(w.confidence))) or w.confidence < 60) for w in words)):
            continue
        tokens = [w.text.strip().replace(',', '') for w in sorted(words, key=(lambda w: w.left))]
        text = ''.join(tokens)
        if not re.fullmatch('[+]?[0-9OIlSB]+', text):
            continue
        value = int(text.lstrip('+').translate(aliases))
        if 0 <= value:
            if not value <= limit:
                continue
        else:
            continue
        votes.append(value)
    if not votes:
        return None
    value = max(set(votes), key=votes.count)
    if votes.count(value) >= 2 and (votes.count(value) * 2) > len(votes):
        return value

def read_raid_result(frame, vision=VisionService):
    """Three optional amounts from a recognized Home Village result, or None.

    Available enemy loot, top-right storage balances, troop levels, and seasonal
    reward panels are outside the anchored number regions. Missing rows stay
    unknown; a visible, confidently read zero is a legitimate result.
    """
    if (frame is None or not getattr(frame, 'size', 0)):
        return None
    h, w = frame.shape[:2]
    footer = (0, (h // 2), w, (h - (h // 2)))
    if not any(((get_template_path(name).exists() and vision.find_template(frame, name, region=footer)[0] is not None) for name in ('returnhome.png', 'returnhome2.png', 'chestclaim.png'))):
        return None
    cx, cy = find_ui_label(frame, vision, 'You got', region=(round((w * 0.25)), round((h * 0.25)), round((w * 0.5)), round((h * 0.4))))
    if cx is None:
        return None
    amounts = []
    for index, limit in enumerate(RESULT_LIMITS):
        top = round((cy + ((0.018 + (index * 0.066)) * h)))
        bottom = (top + round((h * 0.058)))
        left, right = round((cx - (h * 0.28))), round((cx + (h * 0.045)))
        icon_left, icon_right = round((cx + (h * 0.049))), round((cx + (h * 0.115)))
        if (min(top, left) < 0 or bottom > h or icon_right > w):
            return None
        if not _currency_present(frame[top:bottom, icon_left:icon_right], index):
            amounts.append(None)
        else:
            amounts.append(_read_amount(frame[top:bottom, left:right], limit, vision))
    return tuple(amounts)

def read_raid_bonus(frame, vision=VisionService):
    """Read the separate percentage Bonus panel on an already verified result.

    The caller must first recognize the battle result. An absent or not-yet-
    visible panel stays unknown; it never means a guessed zero bonus.
    """
    if (frame is None or not getattr(frame, 'size', 0)):
        return None
    h, w = frame.shape[:2]
    roi = (round((w * 0.66)), round((h * 0.32)), round((w * 0.25)), round((h * 0.35)))
    anchor = None
    for floor in (235, 210, 180):
        words = (vision.find_words_ocr(frame, region=roi, white_text=True, brightness_floor=floor, roi_upscale=3, tesseract_config='--psm 11') or [])
        labels = [word for word in words if word.text.lower().strip(':') == 'bonus' if math.isfinite(float(word.confidence)) if word.confidence >= 70]
        candidates = []
        for label in labels:
            percentages = [
                word for word in words
                if re.fullmatch(r'[0-9]{1,3}%', word.text)
                and word.confidence >= 70
                and 0 <= int(word.text[:-1]) <= 100
                and abs(word.top - label.top) <= (max(word.height, label.height) * 0.6)
                and 0 <= ((label.left - word.left) - word.width) <= (label.height * 1.5)
            ]
            if len(percentages) != 1:
                continue
            candidates.append((round(label.left + (label.width / 2)), round(label.top + (label.height / 2))))
        if not len(candidates) == 1:
            continue
        anchor = candidates[0]
        break
    if anchor is None:
        return None
    cx, cy = anchor
    amounts = []
    for index, limit in enumerate(RESULT_LIMITS):
        top = round((cy + ((0.028 + (index * 0.052)) * h)))
        bottom = (top + round((h * 0.04)))
        left, right = round((cx - (h * 0.135))), round((cx + (h * 0.025)))
        icon_left, icon_right = round((cx + (h * 0.026))), round((cx + (h * 0.069)))
        if (min(left, top) < 0 or icon_right > w or bottom > h):
            return None
        if not _currency_present(frame[top:bottom, icon_left:icon_right], index):
            amounts.append(None)
        else:
            amounts.append(_read_amount(frame[top:bottom, left:right], limit, vision))
    return tuple(amounts)

def corroborate_results(reads):
    """Require agreement across consecutive captures for each resource separately.

    A later contradictory read revokes earlier agreement, which avoids accepting
    an early, still-animating number. An unreadable resource cannot erase another
    resource's valid amount or manufacture a zero.
    """
    previous = [None] * 3
    confirmed = [None] * 3
    for read in reads:
        for index, value in enumerate(read if read is not None else UNKNOWN):
            if value is not None and confirmed[index] is not None and value != confirmed[index]:
                confirmed[index] = None
            if value is not None and value == previous[index]:
                confirmed[index] = value
            previous[index] = value
    return tuple(confirmed)

class LootLedger:
    """One session; each started raid can be resolved at most once."""
    def __init__(self, history_dir=None):
        self.session_id = ((datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-') + uuid4().hex[:8])
        if history_dir is not None:
            pass
        else:
            pass
        self.history_path = None
        self.totals = [0, 0, 0]
        self.missing = [0, 0, 0]
        self.known = [0, 0, 0]
        self.finished = 0
        self.complete = 0
        self.started = 0
        self.pending = None
        self._history_failed = False
        self._last_inventory = None

    def _append(self, event):
        if self.history_path is None or self._history_failed:
            return None
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(schema=1, session_id=self.session_id, at=datetime.now(timezone.utc).isoformat(), **event)
            with self.history_path.open('a', encoding='utf-8') as stream:
                stream.write((json.dumps(record, ensure_ascii=False) + """
"""))
        except OSError:
            self._history_failed = True
            logger.warning('Raid history could not be written; in-memory tracking continues', exc_info=True)

    def begin(self, *, mode='farming', account=None):
        if self.pending is not None:
            self.finish(reason='next raid started before the previous result was recorded')
        self.started += 1
        self.pending = dict(raid_id=self.started, mode=mode, account=account)
        self._append(dict(event='raid_started', **self.pending))
        return self.started

    def finish(self, amounts=UNKNOWN, *, bonus=UNKNOWN, reason='battle result'):
        if self.pending is None:
            return False
        if (len(amounts) != 3 or len(bonus) != 3):
            raise ValueError('A raid must have exactly three resource fields')
        def valid(values):
            return tuple((None for value, limit in zip(values, RESULT_LIMITS) if type(value) is int and 0 <= value <= limit))

        loot, bonus = valid(amounts), valid(bonus)
        confirmed = tuple((None for pair in zip(loot, bonus) if any((value is not None for value in pair))))
        fields_complete = tuple(((a is not None and b is not None) for a, b in zip(loot, bonus)))
        raid = self.pending
        self.pending = None
        self.finished += 1
        self.complete += all(fields_complete)
        for index, value in enumerate(confirmed):
            if not fields_complete[index]:
                self.missing[index] += 1
            if value is None:
                continue
            self.known[index] += 1
            self.totals[index] += value
        self._append(dict(event='raid_result', **raid, source='battle_result' if reason == 'battle result' else reason, raid_loot=dict(zip(RESOURCE_NAMES, loot)), league_bonus=dict(zip(RESOURCE_NAMES, bonus)), resources=dict(zip(RESOURCE_NAMES, confirmed)), complete_fields=dict(zip(RESOURCE_NAMES, fields_complete)), complete=all(fields_complete)))
        logger.info('Raid #%s confirmed loot+bonus: %s (G/E/DE); totals %s; incomplete results %s', raid['raid_id'], '/'.join(((('' if complete else '>=') + str(v)) for v, complete in zip(confirmed, fields_complete) if v is None)), '/'.join(map(str, self.totals)), '/'.join(map(str, self.missing)))
        return True

    def record_inventory(self, amounts, *, account=None):
        snapshot = (account, tuple(amounts))
        if snapshot != self._last_inventory:
            self._last_inventory = snapshot
            self._append(dict(event='inventory_snapshot', account=account, resources=dict(zip(RESOURCE_NAMES, amounts))))
            return None

    def summary(self):
        if self.history_path and not self._history_failed:
            return dict(finished=self.finished, complete=self.complete, missing=list(self.missing), known=list(self.known), pending=self.pending is not None, history=str(self.history_path))
        return dict(finished=self.finished, complete=self.complete, missing=list(self.missing), known=list(self.known), pending=self.pending is not None, history=None)

