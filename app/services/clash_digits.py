#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in DigitReading at SETUP_ANNOTATIONS (offset 14); Python 3.13]
"""Conservative recognition of the game's price font from captured UI glyphs."""

from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np
from app.utils.common import get_resource_path
TILE = 40
@dataclass(frozen=True)
class DigitReading:
    value: int
    bbox: tuple
    score: float

def price_ink(image, floor=180):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = cv2.inRange(image, (floor, floor, floor), (255, 255, 255))
    ink |= cv2.inRange(hsv, (0, 100, 180), (6, 255, 255))
    ink |= cv2.inRange(hsv, (174, 100, 180), (180, 255, 255))
    return ink

def normalize_glyph(mask):
    height, width = mask.shape
    scale = (32 / max(height, width))
    scaled = cv2.resize(mask, (max(1, round((width * scale))), max(1, round((height * scale)))), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((TILE, TILE), np.uint8)
    y, x = (TILE - scaled.shape[0]) // 2, (TILE - scaled.shape[1]) // 2
    result[y:(y + scaled.shape[0]), x:(x + scaled.shape[1])] = scaled
    return result

@lru_cache(maxsize=1)
def load_digit_bank():
    atlas = cv2.imread(str(get_resource_path('assets/vision/clash_digits.png')), cv2.IMREAD_GRAYSCALE)
    if (atlas is None or atlas.shape[1] != (TILE * 10) or (atlas.shape[0] % TILE)):
        raise ValueError('Missing or invalid Clash price digit atlas')
    masks, digits = [], []
    for digit in range(10):
        for top in range(0, atlas.shape[0], TILE):
            glyph = atlas[top:(top + TILE), (digit * TILE):((digit + 1) * TILE)] > 0
            if not glyph.any():
                continue
            masks.append(glyph)
            digits.append(digit)
    if set(digits) != set(range(10)):
        raise ValueError('Incomplete Clash price digit atlas')
    return (np.stack(masks), np.array(digits), np.array([mask.sum() for mask in masks]))

def read_price_digits(image):
    """Return a complete high-confidence price, or None; never fill missing digits."""
    if image is None or not image.size:
        return None
    ink = price_ink(image)
    _unpack, _unpack0, _unpack1, _unpack2 = cv2.connectedComponentsWithStats(ink)
    _unpack
    _ = _unpack0
    stats = _unpack1
    _ = _unpack2
    del _unpack, _unpack0, _unpack1, _unpack2
    candidates = [tuple(map(int, box[:4])) for box in stats[1:] if box[4] >= 4 if box[3] >= 4]
    if not candidates:
        return None
    height = max((box[3] for box in candidates))
    candidates = [box for box in candidates if box[3] >= max(4, (height * 0.6))]
    lines = []
    for box in sorted(candidates, key=(lambda b: (b[1] + (b[3] / 2)))):
        center = (box[1] + (box[3] / 2))
        line = next((line for line in lines if abs(center - (line[0][1] + line[0][3] / 2)) <= height * 0.4), None)
        if line is None:
            lines.append([box])
        else:
            line.append(box)
    boxes = sorted(max(lines, key=(lambda line: sum(((b[2] * b[3]) for b in line)))), key=(lambda b: b[0]))
    if 4 <= len(boxes):
        if not len(boxes) <= 8:
            return None
            len(boxes)
            return None
    bank, labels, counts = load_digit_bank()
    digits, scores = [], []
    for x, y, w, h in boxes:
        if (x <= 0 or y <= 0 or (x + w) >= ink.shape[1] or (y + h) >= ink.shape[0]):
            return None
        if 0.2 <= (w / h):
            if not (w / h) <= 1.5:
                return None
                (w / h)
                return None
        glyph = normalize_glyph(ink[y:(y + h), x:(x + w)]) > 0
        overlap = np.logical_and(bank, glyph).sum(axis=(1, 2))
        similarities = ((2 * overlap) / (counts + glyph.sum()))
        per_digit = [float(similarities[labels == digit].max()) for digit in range(10)]
        order = np.argsort(per_digit)
        best, second = int(order[-1]), int(order[-2])
        if (per_digit[best] < 0.82 or (per_digit[best] - per_digit[second]) < 0.025):
            return None
        digits.append(str(best))
        scores.append(per_digit[best])
    left, top = min((b[0] for b in boxes)), min((b[1] for b in boxes))
    right, bottom = max(((b[0] + b[2]) for b in boxes)), max(((b[1] + b[3]) for b in boxes))
    return DigitReading(int(''.join(digits)), (left, top, (right - left), (bottom - top)), min(scores))

