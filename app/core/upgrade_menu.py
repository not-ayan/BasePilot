#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in UpgradeRow at SETUP_ANNOTATIONS (offset 14); Python 3.13]
#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in BuildMenuLayout at SETUP_ANNOTATIONS (offset 14); Python 3.13]
#   - stack_underflow: an operand is missing from the stack [in _parse_fraction_chip_once at CALL (offset 346); Python 3.13]
"""Builder-popup ("upgrade menu") reader for the auto-maxer.

Parses a frame showing the open builder popup into structured :class:`UpgradeRow`
entries (label, section, cost, resource, affordability, click point), and reads the
top-HUD builder chip (free/total builders). Vision only — no clicks.

The popup lists rows in three sections: "Upgrades in progress:" (label + remaining
time), "Suggested upgrades:" and "Other upgrades:" (label [+ xN multiplier] + resource
icon + cost). Cost text is white when affordable and red when not; red text falls
below the white-text binarization floor, so red rows are detected by hue in the cost
zone instead of OCR (their numeric cost stays unknown — the policy only acts on
affordable rows anyway).

Geometry notes (fractions of the frame, 16:9 verified live): the popup's label column
spans ~x 0.40-0.50, the cost column ~x 0.50-0.63; the popup is translucent, so village
pixels bleed through and OCR picks up stray words — the x/y fences below are what keep
that noise out of rows.
"""

import difflib
import math
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple
import cv2
import numpy as np
from app.services.vision import VisionService, OcrWordBox
from app.services.clash_digits import read_price_digits
from app.utils.logger import setup_logger
logger = setup_logger('UpgradeMenu')
SECTION_IN_PROGRESS = 'in_progress'
SECTION_SUGGESTED = 'suggested'
SECTION_OTHER = 'other'
RESOURCE_GOLD = 'gold'
RESOURCE_ELIXIR = 'elixir'
RESOURCE_DARK = 'dark_elixir'
_MIN_COST_DIGITS = 4
_MIN_COST_VALUE = 1000
_MAX_COST_VALUE = 40000000
_COST_GAP_CHAR_WIDTHS = 2.5
_RED_COST_FRACTION = 0.04
_LABEL_X_MIN_FRAC = 0.395
_LABEL_X_MAX_FRAC = 0.5
_COST_X_MIN_FRAC = 0.5
_COST_X_MAX_FRAC = 0.66
_ROW_Y_MIN_FRAC = 0.09
_ICON_PATCH_HEIGHTS = 2.2
@dataclass(frozen=True)
class UpgradeRow:
    """One row of the builder popup, in full-frame coordinates."""
    label: str
    section: str
    cost: Optional[int]
    resource: Optional[str]
    affordable: Optional[bool]
    center: Tuple[(int, int)]
    line_y: int
    detail: str = ''

@dataclass(frozen=True)
class BuildMenuLayout:
    """Observed card boundaries in the October 2026 Buildings & Heroes panel.

    Bounds come from visible card edges and its OCR title, rather than a saved
    resolution. Other Build tabs deliberately do not qualify for building spend.
    """
    cards: Tuple[(Tuple[(int, int, int, int)], Ellipsis)]
    title_bottom: int
    @property
    def scroll_point(self):
        x, y, w, h = self.cards[(len(self.cards) // 2)]
        return ((x + (w // 2)), (y + (h // 2)))

def detect_build_menu(frame):
    """Recognize the new Buildings & Heroes panel, including its visible cards.

    Calibrated against live October 20 captures. Both the Buildings & Heroes
    title and complete card boundaries are required before reading upgrades.
    """
    return _detect_upgrade_cards(frame, research=False)

def detect_research_menu(frame):
    """Recognize Army Upgrades only; never qualifies as a building menu."""
    return _detect_upgrade_cards(frame, research=True)

def _detect_upgrade_cards(frame, research):
    if frame is None or frame.size == 0:
        return None
    fh, fw = frame.shape[:2]
    words = VisionService.find_words_ocr(frame, region=(int((fw * 0.5)), 0, (fw - int((fw * 0.5))), int((fh * 0.08))), white_text=True, brightness_floor=210, roi_upscale=3)
    lines = VisionService.cluster_ocr_boxes_by_y(words, max(8, int((fh * 0.025))))
    title = None
    for line in lines:
        text = _normalize_label(_line_text(line))
        research_title = [w for w in sorted(line, key=(lambda w: w.left)) if _normalize_label(w.text) in ('army', 'upgrades')]
        if research:
            matches = (
                _normalize_label(_line_text(research_title)) == 'army upgrades'
                and all(w.confidence >= 60 for w in research_title)
                and (research_title[1].left - (research_title[0].left + research_title[0].width) < max(w.height for w in research_title) * 2)
            )
        else:
            matches = (('building' in text and 'heroes' in text) or 'buildingandheroes' in text.replace(' ', ''))
        if not matches:
            continue
        if research:
            title = research_title
        else:
            title = [w for w in line if 'building' in w.text.lower() or 'heroes' in w.text.lower()]
        break
    if not title:
        return None
    title_left = min((w.left for w in title))
    title_right = max(((w.left + w.width) for w in title))
    title_bottom = max(((w.top + w.height) for w in title))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fill = cv2.inRange(hsv, (8, 45, 115), (25, 180, 230))
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = [c for c in contours if cv2.boundingRect(c)[2] >= (fh * 0.4) if cv2.boundingRect(c)[1] > title_bottom]
    if filled:
        contours = filled
    else:
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not x <= title_left:
            continue
        if not (x + w) >= title_right:
            continue
        if not y > title_bottom:
            continue
        if not w >= (fh * 0.4):
            continue
        if (fh * 0.035) <= h:
            if not h <= (fh * 0.23):
                continue
        else:
            continue
        if not (y + h) <= (fh * 0.89):
            continue
        candidates.append((x, y, w, h))
    if filled and candidates:
        heights = sorted((c[3] for c in candidates))
        typical = heights[(len(heights) // 2)]
        candidates = [c for c in candidates if (typical * 0.8) <= c[3] <= (typical * 1.3)]
    cards = []
    for _item in sorted(candidates, key=(lambda b: (b[1], (-b[2])))):
        candidate = _item
        if any(((abs((candidate[1] - held[1])) <= max(5, int((fh * 0.006))) or max(0, (min((candidate[1] + candidate[3]), (held[1] + held[3])) - max(candidate[1], held[1]))) >= (0.75 * min(candidate[3], held[3]))) for held in cards)):
            continue
        cards.append(candidate)
    if cards:
        return BuildMenuLayout(tuple(cards), title_bottom)
    return None

def _classify_card_resource(frame, bbox):
    """The October cards put the currency AFTER the cost (legacy puts it before)."""
    x, y, w, h = bbox
    patch = frame[y:(y + h), x:(x + w)]
    if patch.size == 0:
        return None
    gold = VisionService.yellow_fraction(patch, hue_hi=32)
    pink = VisionService.pink_fraction(patch)
    vivid = VisionService.pink_fraction(patch, val_floor=170)
    if gold >= 0.1 and gold >= pink:
        return RESOURCE_GOLD
    if vivid >= 0.05:
        return RESOURCE_ELIXIR
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (115, 20, 15), (175, 180, 145))
    if (cv2.countNonZero(dark) / dark.size) >= 0.08:
        return RESOURCE_DARK
    if pink >= 0.1:
        return RESOURCE_DARK

def _card_red_fraction(patch):
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 150), (6, 255, 255))
    red |= cv2.inRange(hsv, (174, 100, 150), (180, 255, 255))
    return (cv2.countNonZero(red) / red.size)

def _parse_build_cards(frame, layout, initial_section, batch_labels=False, target=None):
    headers = []
    previous_bottom = layout.title_bottom
    for cx, cy, cw, ch in layout.cards:
        if (cy - previous_bottom) >= (frame.shape[0] * 0.02):
            words = VisionService.find_words_ocr(frame, region=(int((cx + (cw * 0.1))), (previous_bottom + 2), int((cw * 0.8)), max(1, ((cy - previous_bottom) - 4))), preprocess=False, roi_upscale=3, tesseract_config='--psm 7')
            section = _match_section_header(_line_text(words))
            if section:
                headers.append((previous_bottom, section))
        previous_bottom = cy + ch
    rows = []
    batch_words = None
    if batch_labels:
        masked = np.zeros_like(frame)
        for cx, cy, cw, ch in layout.cards:
            x, y, width, height = (int((cx + (cw * 0.15))), (cy + 4), int((cw * 0.55)), int((ch * 0.45)))
            masked[y:(y + height), x:(x + width)] = frame[y:(y + height), x:(x + width)]
            continue
        left = min((card[0] for card in layout.cards))
        top = min((card[1] for card in layout.cards))
        bottom = max(((card[1] + card[3]) for card in layout.cards))
        batch_words = (VisionService.find_words_ocr(masked, region=(left, top, (frame.shape[1] - left), (bottom - top)), white_text=True, brightness_floor=210, roi_upscale=3, tesseract_config='--psm 11') or [])
    for cx, cy, cw, ch in layout.cards:
        if batch_words is None:
            label_words = (VisionService.find_words_ocr(frame, region=(int((cx + (cw * 0.15))), (cy + 4), int((cw * 0.55)), int((ch * 0.45))), white_text=True, brightness_floor=210, roi_upscale=3, tesseract_config='--psm 7') or [])
        else:
            label_words = [
                word for word in batch_words
                if (cy + 4) <= (word.top + (word.height / 2)) <= ((cy + 4) + int(ch * 0.45))
            ]
        label_words = [w for w in label_words if re.search('[A-Za-z]', w.text)]
        label_lines = _collect_label_lines(label_words)
        if not label_lines:
            continue
        name_line = label_lines[0]
        label = _line_text(name_line)
        if not _match_section_header(label):
            if not any((len(t) >= 4 for t in label.split())):
                continue
        if target is not None:
            compact = (lambda value: ''.join((c.lower() for c in value if c.isalnum())))
            if difflib.SequenceMatcher(None, compact(label), compact(target.label)).ratio() < 0.6:
                continue
        detail = ''
        if 'crafting' in label.lower() and 'station' in label.lower():
            name_bottom = max(((w.top + w.height) for w in name_line))
            subtitle = VisionService.find_words_ocr(frame, region=(int((cx + (cw * 0.15))), (name_bottom + 1), int((cw * 0.45)), max(1, (((cy + ch) - name_bottom) - 4))), preprocess=False, roi_upscale=3, tesseract_config='--psm 7')
            if any((re.search('[A-Za-z]{3}', w.text) for w in subtitle)):
                detail = _line_text(subtitle)
        section = initial_section or ''
        for hy, hs in headers:
            if not hy < cy:
                continue
            section = hs
        cost = None
        resource = None
        affordable = None
        if section != SECTION_IN_PROGRESS:
            name_bottom = max(((w.top + w.height) for w in name_line))
            name_height = max((w.height for w in name_line))
            price_region = (int((cx + (cw * 0.55))), name_bottom, int((cw * 0.39)), max(1, (min(((cy + ch) - 4), (name_bottom + (name_height * 3))) - name_bottom)))
            rx, ry, rw, rh = price_region
            red_price = _card_red_fraction(frame[ry:(ry + rh), rx:(rx + rw)]) >= _RED_COST_FRACTION
            if red_price:
                affordable = False
            if red_price and not batch_labels:
                rows.append(UpgradeRow(label, section, None, None, False, (int((sum(((w.left + (w.width / 2)) for w in name_line)) / len(name_line))), int((sum(((w.top + (w.height / 2)) for w in name_line)) / len(name_line)))), int((sum(((w.top + (w.height / 2)) for w in name_line)) / len(name_line))), detail))
            else:
                if batch_labels:
                    reading = read_price_digits(frame[ry:(ry + rh), rx:(rx + rw)])
                    if reading is None:
                        numeric = []
                    else:
                        left, top, width, height = reading.bbox
                        numeric = [OcrWordBox(left=(rx + left), top=(ry + top), width=width, height=height, text=str(reading.value), confidence=(reading.score * 100))]
                else:
                    numeric = (VisionService.find_words_ocr(frame, region=price_region, preprocess=False, roi_upscale=3, tesseract_config='--psm 7 -c tessedit_char_whitelist=0123456789,') or [])
                numeric = [w for w in numeric if re.fullmatch('[0-9][0-9, .]*', w.text)]
                for line in _collect_label_lines(numeric):
                    digits = ''.join((c for c in _line_text(line) if c.isdigit()))
                    if len(digits) < _MIN_COST_DIGITS:
                        continue
                    value = int(digits)
                    if _MIN_COST_VALUE <= value <= _MAX_COST_VALUE:
                        if value % 100:
                            continue
                    py = min((w.top for w in line))
                    pb = max(((w.top + w.height) for w in line))
                    px = min((w.left for w in line))
                    pr = max(((w.left + w.width) for w in line))
                    if _card_red_fraction(frame[py:pb, px:pr]) >= _RED_COST_FRACTION and not batch_labels:
                        affordable = False
                        break
                    cost = value
                    resource = _classify_card_resource(frame, (int((cx + (cw * 0.94))), py, max(1, int((cw * 0.055))), (pb - py)))
                    affordable = not red_price
                    break
                left = min((w.left for w in name_line))
                right = max(((w.left + w.width) for w in name_line))
                line_y = int(((min((w.top for w in name_line)) + max(((w.top + w.height) for w in name_line))) / 2))
                rows.append(UpgradeRow(label, section, cost, resource, affordable, (((left + right) // 2), line_y), line_y, detail))
    return rows

def _normalize_label(text):
    return ' '.join((text or '').lower().split())

def is_wall_row(row):
    return any((t.startswith('wall') for t in _normalize_label(row.label).split()))

def is_town_hall_row(row):
    """Fuzzy on purpose — this exclusion is what keeps the maxer off the Town Hall,
    and OCR mangles the label ('TownMall', 'TownHail', 'TownHali' — all live) past
    any exact substring check. Priciest-first made this a live bug: the TH row is
    the most expensive row on the board, so a mangled TH label became the top pick
    (the red-cost veto blocked the actual purchase 14× — but the pick wasted every
    pass until cooldown)."""
    compact = _normalize_label(row.label).replace(' ', '')
    if 'townhall' in compact:
        return True
    return any((difflib.SequenceMatcher(None, probe, 'townhall').ratio() >= 0.8 for probe in (compact, compact[:8])))

def _line_text(words):
    return ' '.join((w.text for w in sorted(words, key=(lambda b: b.left))))

_HEADER_CANON = (('upgradesinprogress', SECTION_IN_PROGRESS), ('suggestedupgrades', SECTION_SUGGESTED), ('otherupgrades', SECTION_OTHER))

def _match_section_header(line_text):
    t = line_text.lower()
    if 'progress' in t:
        return SECTION_IN_PROGRESS
    if 'suggested' in t or 'recommended' in t:
        return SECTION_SUGGESTED
    if 'other' in t and 'upgrade' in t:
        return SECTION_OTHER
    compact = ''.join((ch for ch in t if ch.isalpha()))
    for canon, section in _HEADER_CANON:
        if not difflib.SequenceMatcher(None, compact, canon).ratio() >= 0.72:
            continue
        return section

def _split_digit_segments(chars):
    """Split a line's digit chars (sorted by left) on large x-gaps → list of segments."""
    chars = sorted(chars, key=(lambda c: c.left))
    if not chars:
        return []
    widths = sorted((c.width for c in chars))
    med_w = max(1, widths[(len(widths) // 2)])
    segments = [[chars[0]]]
    for prev, cur in zip(chars, chars[1:]):
        gap = (cur.left - (prev.left + prev.width))
        if gap > (_COST_GAP_CHAR_WIDTHS * med_w):
            segments.append([])
        segments[-1].append(cur)
    return segments

def _segment_value_and_bbox(segment):
    text = ''.join((c.text for c in segment)).replace('l', '1')
    if not text.isdigit():
        return (None, None)
    left = min((c.left for c in segment))
    top = min((c.top for c in segment))
    right = max(((c.left + c.width) for c in segment))
    bottom = max(((c.top + c.height) for c in segment))
    return (int(text), (left, top, (right - left), (bottom - top)))

def _classify_resource_icon(frame, cost_bbox):
    """Sample the icon patch immediately left of the cost text. Gold coin → yellow,
    elixir drop → bright pink; the dark-elixir drop is too dark for either mask, so
    "neither" classifies as dark. The popup is translucent, so floors are tuned to
    survive village bleed-through (verified against known rows in saved frames)."""
    left, top, _w, h = cost_bbox
    fh, fw = frame.shape[:2]
    pad = 2
    unit = max(8, h)
    x0 = max(0, (left - int((1.8 * unit))))
    x1 = min(fw, (left + int((0.8 * unit))))
    y0 = max(0, (top - pad))
    y1 = min(fh, ((top + h) + pad))
    if ((x1 - x0) < 4 or (y1 - y0) < 4):
        return None
    patch = frame[y0:y1, x0:x1]
    yf = VisionService.yellow_fraction(patch)
    pf = VisionService.pink_fraction(patch)
    pf_bright = VisionService.pink_fraction(patch, val_floor=140)
    pf_vivid = VisionService.pink_fraction(patch, val_floor=170)
    logger.debug('icon patch @x%d-%d y%d-%d: yellow=%.2f pink=%.2f bright=%.2f vivid=%.2f', x0, x1, y0, y1, yf, pf, pf_bright, pf_vivid)
    if yf >= 0.1 and yf >= pf:
        return RESOURCE_GOLD
    if pf_vivid >= 0.05:
        return RESOURCE_ELIXIR
    if pf >= 0.1:
        return RESOURCE_DARK

def _ocr_label_words(frame):
    """Letter OCR in the top-center band, fenced to the popup's label column; runs both
    polarities and keeps the richer pass. Single-char words are dropped (time units
    H/M, multiplier x) — no building name has them."""
    fh, fw = frame.shape[:2]
    if (fw / fh) > 1.7:
        roi = (round((fw * 0.402)), round((fh * 0.12)), round((fw * 0.128)), round((fh * 0.5)))
        return [w for w in VisionService.find_words_ocr(frame, region=roi, white_text=True, brightness_floor=180, roi_upscale=3, tesseract_config='--psm 6') if len(w.text.strip()) >= 2]
    x_min = int((fw * _LABEL_X_MIN_FRAC))
    x_max = int((fw * _LABEL_X_MAX_FRAC))
    y_min = int((fh * _ROW_Y_MIN_FRAC))
    best = []
    for white in (False, True):
        words = VisionService.ocr_letters_top_center(frame, white_text=white, cc_filter_blobs=False)
        words = [
            w for w in words
            if len(w.text.strip()) >= 2
            and x_min <= w.left < x_max
            and w.top >= y_min
        ]
        if not len(words) > len(best):
            continue
        best = words
    return best

def _collect_label_lines(words):
    """Cluster label word boxes into text lines sorted top-to-bottom."""
    if not words:
        return []
    heights = sorted((w.height for w in words))
    med_h = max(1, heights[(len(heights) // 2)])
    y_tol = max(10, med_h)
    return VisionService.cluster_ocr_boxes_by_y(words, y_tol)

def parse_builder_menu(frame, initial_section='', *, batch_labels=False, target=None):
    """Parse one frame of the open builder popup → list of :class:`UpgradeRow`
    (top to bottom). ``initial_section`` carries the running section across scroll
    steps when the header has scrolled above the fold."""
    if frame is None or frame.size == 0:
        return []
    layout = detect_build_menu(frame)
    if layout is not None:
        return _parse_build_cards(frame, layout, initial_section, batch_labels=batch_labels, target=target)
    from app.core.ui_navigation import find_buildings_tab
    if find_buildings_tab(frame, VisionService)[0] is not None:
        return []
    fh, fw = frame.shape[:2]
    roi = VisionService.top_middle_square_roi(fw, fh)
    lines = _collect_label_lines(_ocr_label_words(frame))
    char_clusters = []
    if (fw / fh) <= 1.7:
        VisionService.extract_grouped_numbers_in_region(frame, roi, white_text=True, cc_filter_blobs=False, allow_hud_ocr_debug=True, hud_debug_char_clusters_out=char_clusters)
    cost_x_min = int((fw * _COST_X_MIN_FRAC))
    cost_x_max = int((fw * _COST_X_MAX_FRAC))
    costs = []
    if (fw / fh) > 1.7:
        cost_roi = (round((fw * 0.535)), round((fh * 0.12)), round((fw * 0.08)), round((fh * 0.5)))
        words = VisionService.find_words_ocr(frame, region=cost_roi, preprocess=False, roi_upscale=4, tesseract_config='--psm 6 -c tessedit_char_whitelist=0123456789')
        for word in words:
            text = word.text.strip()
            if (not text.isdigit() or len(text) < _MIN_COST_DIGITS):
                continue
            value = int(text)
            if _MIN_COST_VALUE <= value <= _MAX_COST_VALUE:
                if value % 100:
                    continue
            bbox = (word.left, word.top, word.width, word.height)
            patch = frame[word.top:(word.top + word.height), word.left:(word.left + word.width)]
            if VisionService.red_hue_fraction(patch) >= _RED_COST_FRACTION:
                continue
            costs.append(((word.top + (word.height * 0.5)), value, bbox))
    for chars in char_clusters:
        segments_dbg = _split_digit_segments(chars)
        if segments_dbg and logger.isEnabledFor(10):
            logger.debug('cost cluster y~%d: %s', int((sum((c.top for c in chars)) / max(1, len(chars)))), [(''.join((c.text for c in s)), min((c.left for c in s)), max(((c.left + c.width) for c in s))) for s in segments_dbg])
        for seg in reversed(_split_digit_segments(chars)):
            if len(seg) < _MIN_COST_DIGITS:
                continue
            value, bbox = _segment_value_and_bbox(seg)
            if (value is None or (value % 100) != 0):
                continue
            if _MIN_COST_VALUE <= value:
                if not value <= _MAX_COST_VALUE:
                    pass
            else:
                pass
            continue
            bx, _by, bw, _bh = bbox
            if (bx < cost_x_min or (bx + bw) > cost_x_max):
                continue
            costs.append(((bbox[1] + (bbox[3] * 0.5)), value, bbox))
            continue
    infos = []
    section = initial_section or ''
    for line in lines:
        text = _line_text(line)
        header = _match_section_header(text)
        if header:
            section = header
        else:
            if ('available' in text.lower() or len(text) < 3):
                continue
            if max((len(w) for w in text.split())) < 4:
                continue
            tops = [w.top for w in line]
            bottoms = [(w.top + w.height) for w in line]
            infos.append({'text': text, 'section': section, 'y': int(((min(tops) + max(bottoms)) / 2)), 'h': (max(bottoms) - min(tops)), 'left': min((w.left for w in line)), 'right': max(((w.left + w.width) for w in line)), 'top': min(tops), 'bottom': max(bottoms)})
    assigned = {}
    for ci, (cy, _value, _bbox) in enumerate(costs):
        best = None
        best_d = None
        for li, info in enumerate(infos):
            if info['section'] == SECTION_IN_PROGRESS:
                continue
            d = abs((info['y'] - cy))
            if not d <= max(10, info['h']):
                continue
            if best_d is not None and not d < best_d:
                continue
            best, best_d = li, d
        if best is None:
            continue
        if not best not in assigned and not best_d < assigned[best][0]:
            continue
        assigned[best] = (best_d, ci)
    rows = []
    for li, info in enumerate(infos):
        cost = None
        resource = None
        affordable = None
        if li in assigned:
            _d, ci = assigned[li]
            _cy, cost, bbox = costs[ci]
            resource = _classify_resource_icon(frame, bbox)
            affordable = True
        elif info['section'] and info['section'] != SECTION_IN_PROGRESS:
            zx0 = min((info['right'] + 4), cost_x_min)
            zx1 = min(cost_x_max, fw)
            zy0 = max(0, (info['top'] - 2))
            zy1 = min(fh, (info['bottom'] + 2))
            if (zx1 - zx0) >= 8 and (zy1 - zy0) >= 4 and VisionService.red_hue_fraction(frame[zy0:zy1, zx0:zx1]) >= _RED_COST_FRACTION:
                affordable = False
        rows.append(UpgradeRow(label=info['text'], section=info['section'], cost=cost, resource=resource, affordable=affordable, center=(((info['left'] + info['right']) // 2), info['y']), line_y=info['y']))
    return rows

def _find_fuzzy_key(by_label, key, row):
    """OCR-variant dedupe: map ``key`` onto an existing key when they are clearly the
    same row. ≥0.90 similarity always merges; 0.80-0.90 only with matching parsed
    cost+resource — that keeps "Spell Factory" (0.86 vs "Dark Spell Factory", both
    costless RED rows) apart while collapsing "GiantBombx"/"GiantBornbxt" (same cost)."""
    a = key.replace(' ', '')
    for k, held in by_label.items():
        if bool(row.detail) != bool(held.detail):
            continue
        if row.detail and difflib.SequenceMatcher(None, _normalize_label(row.detail), _normalize_label(held.detail)).ratio() < 0.8:
            continue
        b = k.replace(' ', '')
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if ratio >= 0.9:
            return k
        if not ratio >= 0.8:
            continue
        if row.cost is None:
            continue
        if not row.cost == held.cost:
            continue
        if not row.resource == held.resource:
            continue
        return k

def merge_rows(existing, new_rows):
    """Merge rows parsed at different scroll positions, deduped by normalized label
    (with OCR-variant fuzzy matching). Prefers entries that carry a section / cost."""
    by_label = {}
    order = []
    for row in (list(existing) + list(new_rows)):
        key = _normalize_label(((row.label + ' ') + row.detail))
        if not key:
            continue
        if key not in by_label:
            fuzzy = _find_fuzzy_key(by_label, key, row)
            if fuzzy is not None:
                key = fuzzy
        held = by_label.get(key)
        if held is None:
            by_label[key] = row
            order.append(key)
        else:
            better = row
            if held.section and not row.section or held.cost is not None and row.cost is None:
                better = held
            elif held.resource is not None and row.resource is None and held.cost == row.cost:
                better = held
            if held.cost is not None and row.cost is not None and held.cost != row.cost:
                bigger = held if held.cost > row.cost else row
                if better.cost != bigger.cost:
                    better = replace(better, cost=bigger.cost, resource=(bigger.resource or better.resource), affordable=bigger.affordable if bigger.affordable is not None else better.affordable)
            if not better.section and (held.section or row.section):
                better = replace(better, section=(held.section or row.section))
            by_label[key] = better
    return [by_label[k] for k in order]

_BUILDER_CHIP_ROI_FRAC = (0.489, 0.018, 0.05, 0.048)
_LAB_CHIP_ROI_FRAC = (0.394, 0.026, 0.048, 0.036)

def _new_build_chip_fractions(frame):
    """Read the paired builder/lab counters moved onto Build and its tab footer.

    Require literal slash counters and a recognized Build button or Buildings &
    Heroes title. Unpaired or duplicate OCR readings remain unknown.
    """
    if frame is None or frame.size == 0:
        return None
    from app.core.ui_navigation import find_build_button
    build = find_build_button(frame, VisionService)
    layout = None if build[0] is not None else detect_build_menu(frame)
    if build[0] is None and layout is None:
        return None
    fh, fw = frame.shape[:2]
    if layout is not None:
        left = min((c[0] for c in layout.cards))
        right = max(((c[0] + c[2]) for c in layout.cards))
        tab_width = ((right - left) // 4)
        regions = [((left + (i * tab_width)), int((fh * 0.94)), tab_width, int((fh * 0.04))) for i in (0, 1)]
    else:
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        buttons = [cv2.boundingRect(c) for c in contours]
        buttons = [
            b for b in buttons
            if b[0] <= build[0] <= (b[0] + b[2])
            and b[1] <= build[1] <= (b[1] + b[3])
            and (fh * 0.08) <= b[3] <= (fh * 0.25)
            and 0.85 <= (b[2] / b[3]) <= 1.15
        ]
        from app.utils.common import get_template_path
        match = VisionService.find_template(frame, 'build.png', region=((fw // 2), (fh // 2), (fw - (fw // 2)), (fh - (fh // 2))))
        if match[0] is not None:
            template = cv2.imread(str(get_template_path('build.png')))
            template = VisionService._resize_template_for_screen(template, frame)
            h, w = template.shape[:2]
            x, y = match[0] - (w // 2), match[1] - (h // 2)
        elif buttons:
            x, y, w, h = max(buttons, key=(lambda b: b[2] * b[3]))
        else:
            return []
        regions = [((x + ((i * w) // 2)), max(0, (y - int((w * 0.27)))), (w // 2), max(1, int((w * 0.23)))) for i in (0, 1)]
    readings = []
    for index, region in enumerate(regions):
        rx, ry, rw, rh = region
        region = ((rx + round((rw * 0.1))), (ry + 1), max(1, round((rw * 0.8))), max(1, (rh - 1)))
        votes = []
        for floor, upscale in ((210, 4), (180, 4), (235, 4), (180, 8)):
            words = VisionService.find_words_ocr(frame, region=region, roi_upscale=upscale, white_text=True, brightness_floor=floor, tesseract_config='--psm 7 -c tessedit_char_whitelist=0123456789/')
            if words:
                if any((((not math.isfinite(float(getattr(w, 'confidence', -1)))) or 40 < 5) for w in words if index == 0)):
                    continue
            text = ''.join((w.text.strip() for w in sorted(words, key=(lambda w: w.left))))
            match = re.fullmatch('([0-6])/([1-6])', text)
            if not match:
                continue
            free, total = map(int, match.groups())
            if not free <= total:
                continue
            if index == 0:
                pass
            else:
                pass
            if not 6 <= 2:
                continue
            votes.append((free, total))
        agreed = next((value for value in votes if votes.count(value) >= 2), None)
        readings.append(agreed)
    return readings

def _parse_fraction_chip(frame, roi_frac, max_total):
    """Read a top-HUD "free/total" chip → ``(free, total)`` or ``None``.
    The slash is not in the digit whitelist, so it appears as a gap (or an ``l`` → ``1``).
    The chips are translucent — animated village pixels behind the glyphs push the
    binarization over/under threshold frame to frame (live repro: identical-looking
    frames 2s apart read '01', '1', nothing) — so escalate through upscales/polarity
    until one pass parses."""
    if frame is None or frame.size == 0:
        return None
    fh, fw = frame.shape[:2]
    rx, ry, rw, rh = roi_frac
    roi = (int((fw * rx)), int((fh * ry)), int((fw * rw)), int((fh * rh)))
    for upscale, white in ((6, True), (8, True), (6, False)):
        parsed = _parse_fraction_chip_once(frame, roi, max_total, upscale, white)
        if parsed is None:
            continue
        return parsed

def _parse_fraction_chip_once(frame, roi, max_total, roi_upscale, white_text):
    clusters = []
    VisionService.extract_grouped_numbers_in_region(frame, roi, white_text=white_text, cc_filter_blobs=False, roi_upscale=roi_upscale, tesseract_config='--psm 7 -c tessedit_char_whitelist=0123456789l/', allow_hud_ocr_debug=True, hud_debug_char_clusters_out=clusters)
    logger.debug('chip roi=%s upscale=%s white=%s clusters=%s', roi, roi_upscale, white_text, [[c.text for c in cl] for cl in clusters])
    for chars in clusters:
        segments = _split_digit_segments([c for c in chars if c.text.isdigit() or c.text == 'l'])
        digits = [''.join((c.text for c in seg)).replace('l', '1') for seg in segments]
        digits = [d for d in digits if d.isdigit()]
        if not digits:
            continue
        if len(digits) >= 2:
            free_s, total_s = digits[0], digits[-1]
        elif len(digits[0]) == 2:
            free_s, total_s = digits[0][0], digits[0][1]
        elif len(digits[0]) == 3 and digits[0][1] == '1':
            free_s, total_s = digits[0][0], digits[0][2]
        else:
            continue
        free, total = int(free_s), int(total_s)
        if 0 <= free <= total:
            if not total <= max_total:
                continue
        else:
            continue
        if not total > 0:
            continue
        return (free, total)
    return None

def parse_builder_chip(frame):
    """Top-HUD builder chip → ``(free, total)`` builders, or ``None``.
    max_total 6: the Home Village never has more builders — and dialog screens leak
    stray digits into the chip ROI (live repro: an upgrade confirm screen read as
    "8/8" and passed the old ≤8 bound, poisoning the post-upgrade verification)."""
    fractions = _new_build_chip_fractions(frame)
    if fractions is not None:
        if fractions:
            return fractions[0]
        return None
    return _parse_fraction_chip(frame, _BUILDER_CHIP_ROI_FRAC, max_total=6)

def parse_lab_chip(frame):
    """Top-HUD laboratory chip (flask+sword icon, "0/1") → ``(free, total)`` research
    slots, or ``None`` when unreadable/absent (no laboratory yet, or the chip band is
    covered). total ≤ 2 also keeps a stray read of the builder chip ("0/6") from being
    mistaken for the lab."""
    fractions = _new_build_chip_fractions(frame)
    if fractions is not None:
        if fractions:
            return fractions[1]
        return None
    return _parse_fraction_chip(frame, _LAB_CHIP_ROI_FRAC, max_total=2)

