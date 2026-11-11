"""Recognize visible navigation labels after the October 2026 HUD rework.

The zones below describe where to *search*, never where to click. Click targets
come from OCR boxes. English labels are a fallback for the existing templates;
unknown, low-confidence, and ambiguous labels remain unrecognized.
"""

import math
import re
from app.utils.common import get_template_path
import cv2
import numpy as np

def _normalized(text):
    aliases = {'battie': 'battle', 'matcuh': 'match', 'confinm': 'confirm'}
    return ' '.join((aliases.get(token, token) for token in re.findall('[a-z0-9]+', str(text).lower())))

def _zone(frame, left=0, top=0, right=1, bottom=1):
    height, width = frame.shape[:2]
    x, y = round((width * left)), round((height * top))
    return (x, y, (round((width * right)) - x), (round((height * bottom)) - y))

def _segments(words, region, min_confidence):
    """Group nearby words on a line without joining separate menu buttons."""
    x, y, width, height = region
    valid = []
    for _item in (words or ()):
        word = _item
        cx = (word.left + (word.width / 2))
        cy = (word.top + (word.height / 2))
        if not _normalized(word.text):
            continue
        if x <= cx:
            if not cx < (x + width):
                continue
        else:
            continue
        if y <= cy:
            if not cy < (y + height):
                continue
        else:
            continue
        valid.append(word)
    lines = []
    for _item in sorted(valid, key=(lambda w: ((w.top + (w.height / 2)), w.left))):
        word = _item
        cy = (word.top + (word.height / 2))
        line = next((line for line in lines if abs(cy - (line[0].top + line[0].height / 2)) <= max(word.height, line[0].height) * 0.6), None)
        if line is None:
            lines.append([word])
        else:
            line.append(word)
    result = []
    for line in lines:
        segment = []
        for _item in sorted(line, key=(lambda w: w.left)):
            word = _item
            if segment and (word.left - (segment[-1].left + segment[-1].width)) > (min(word.height, segment[-1].height) * 1.25):
                result.append(segment)
                segment = []
            segment.append(word)
        if not segment:
            continue
        result.append(segment)
    return result

def find_ui_label(frame, vision, labels, region=None, min_confidence=70):
    """Center of one complete, confidently read label, or ``(None, None)``.

    Unlike username OCR, matching is exact: "Ranked Battle" cannot match
    "Battle", and "Use gems" cannot match "Use". Confidence is checked here
    because the shared OCR service retains all words regardless of its argument.
    """
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    if isinstance(labels, str):
        labels = (labels,)
    wanted = {_normalized(label) for label in labels}
    region = (region or _zone(frame))
    for white, floor in ((True, 235), (True, 210), (True, 190), (False, None)):
        words = vision.find_words_ocr(frame, region=region, min_confidence=min_confidence, preprocess=True, white_text=white, brightness_floor=floor, roi_upscale=3, tesseract_config='--psm 11')
        hits = []
        for segment in _segments(words, region, min_confidence):
            if _normalized(' '.join((word.text for word in segment))) not in wanted:
                continue
            if any((((not math.isfinite(float(getattr(word, 'confidence', -1)))) or float(getattr(word, 'confidence', -1)) < min_confidence) for word in segment)):
                continue
            left = min((word.left for word in segment))
            top = min((word.top for word in segment))
            right = max(((word.left + word.width) for word in segment))
            bottom = max(((word.top + word.height) for word in segment))
            hits.append((((left + right) // 2), ((top + bottom) // 2)))
        if len(hits) == 1:
            return hits[0]
        if not len(hits) > 1:
            continue
        return (None, None)
    return (None, None)

def find_battle_button(frame, vision):
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    return find_ui_label(frame, vision, 'Battle', _zone(frame, 0, 0.8, 0.25, 1))

def find_build_button(frame, vision):
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    region = _zone(frame, 0.5, 0.5, 1, 1)
    if get_template_path('build.png').is_file():
        point = vision.find_template(frame, 'build.png', region=region)
        if point[0] is not None:
            return point
    return find_ui_label(frame, vision, 'Build', region)

def find_buildings_tab(frame, vision):
    """Find the Buildings & Heroes footer icon, including when another tab is active."""
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    if not get_template_path('buildings_tab.png').is_file():
        return (None, None)
    return vision.find_template(frame, 'buildings_tab.png', region=_zone(frame, 0.5, 0.88, 1, 1))

def find_research_tab(frame, vision):
    """Army Upgrades footer icon; the panel title is verified after navigation."""
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    if not get_template_path('research_tab.png').is_file():
        return (None, None)
    return vision.find_template(frame, 'research_tab.png', region=_zone(frame, 0.5, 0.88, 1, 1))

def find_profile_button(frame, vision):
    """Recognize the profile card from its border, level badge, and player name."""
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    label = find_ui_label(frame, vision, 'Profile', _zone(frame, 0, 0, 0.5, 0.5))
    if label[0] is not None:
        return label
    if not is_village_hud(frame, vision):
        return (None, None)
    if get_template_path('profile.png').is_file():
        point = vision.find_template(frame, 'profile.png', region=_zone(frame, 0, 0, 0.3, 0.25))
        if point[0] is not None:
            return point
    height, width = frame.shape[:2]
    roi = frame[:round((height * 0.25)), :round((width * 0.3))]
    edges = cv2.Canny(roi, 80, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.75 <= (w / max(1, h)) <= 1.3:
            if (height * 0.05) <= h:
                if not h <= (height * 0.2):
                    pass
            else:
                pass
        continue
        pad = max(2, round((h * 0.3)))
        badge = frame[max(0, (y - pad)):min(height, (y + pad)), max(0, (x - pad)):min(width, (x + round((w * 0.5))))]
        hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
        cyan = cv2.inRange(hsv, np.array([80, 110, 130]), np.array([110, 255, 255]))
        if (np.count_nonzero(cyan) / max(1, cyan.size)) < 0.04:
            continue
        name_x = x + w
        name_region = (name_x, (y + round((h * 0.15))), min((width - name_x), (w * 3)), max(1, round((h * 0.5))))
        names = vision.find_words_ocr(frame, region=name_region, white_text=True, brightness_floor=235, roi_upscale=4, tesseract_config='--psm 7')
        if not any(((len(re.sub('[^a-zA-Z]', '', word.text)) >= 3 and float(getattr(word, 'confidence', -1)) >= 40) for word in (names or ()))):
            continue
        point = ((x + (w // 2)), (y + (h // 2)))
        if any(((abs((point[0] - other[0])) <= (h * 0.1) and abs((point[1] - other[1])) <= (h * 0.1)) for other in candidates)):
            continue
        candidates.append(point)
    if len(candidates) == 1:
        return candidates[0]
    return (None, None)

def find_settings_button(frame, vision):
    return find_ui_label(frame, vision, 'Settings')

def is_village_hud(frame, vision):
    """Both primary buttons are visible; does not infer which village is open."""
    return (find_battle_button(frame, vision)[0] is not None and find_build_button(frame, vision)[0] is not None)

def find_navigation_fallback(frame, vision, template, region=None):
    if (frame is None or getattr(frame, 'size', 0) == 0):
        return (None, None)
    if template == 'attack.png':
        if find_build_button(frame, vision)[0] is None:
            return (None, None)
        return find_battle_button(frame, vision)
    if template in ('farmbattle.png', 'rankedbattle.png'):
        if find_ui_label(frame, vision, 'Multiplayer', _zone(frame, 0, 0, 1, 0.16))[0] is None:
            return (None, None)
        titles = _zone(frame, 0, 0.15, 1, 0.8)
        farm = find_ui_label(frame, vision, 'Battle', titles)
        ranked = find_ui_label(frame, vision, 'Ranked Battle', titles)
        if farm[0] is None or ranked[0] is None or farm[0] >= ranked[0]:
            return (None, None)
        distance = ranked[0] - farm[0]
        divider = ((farm[0] + ranked[0]) // 2)
        left = divider if template == 'rankedbattle.png' else max(0, (farm[0] - (distance // 2)))
        right = min(frame.shape[1], (ranked[0] + (distance // 2))) if template == 'rankedbattle.png' else divider
        title = ranked if template == 'rankedbattle.png' else farm
        return find_ui_label(frame, vision, 'Find a Match', (left, (title[1] + round((frame.shape[0] * 0.04))), (right - left), round((frame.shape[0] * 0.05))), min_confidence=40)
    labels = {'train.png': 'Edit Army', 'attack2.png': 'Attack', 'rankedattackconfirm.png': 'Attack', 'findnow.png': 'Find Now', 'savedrecipes.png': 'Saved Recipes', 'use.png': 'Use', 'returnhome.png': 'Return Home', 'returnhome2.png': 'Return Home', 'breturnhome.png': 'Return Home', 'breturnhome1920.png': 'Return Home', 'surrender.png': 'Surrender', 'endbattle.png': 'End Battle'}
    label = labels.get(template)
    if label is None:
        return (None, None)
    return find_ui_label(frame, vision, label, region=region)

