'''Digit OCR and battle raid loot recognition using Clash of Clans font templates with icon-anchored precision.'''
from __future__ import annotations
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.utils.common import get_resource_path
from app.utils.logger import setup_logger

logger = setup_logger('LootOCR')

class LootOCR:
    '''Custom glyph-matching OCR engine specialized for Clash of Clans numbers.'''

    def __init__(self, font_path = None):
        if font_path is None:
            self.font_path = str(get_resource_path('assets/Clash_Regular.otf.ttf'))
        else:
            self.font_path = font_path
        self.gold_icon_path = str(get_resource_path('assets/gold.png'))
        self.elixir_icon_path = str(get_resource_path('assets/elixir.png'))
        self.digit_templates = {}
        self.gold_icon_base = None
        self.elixir_icon_base = None
        self.generated = False

    def generate_templates(self):
        if self.generated:
            return
        if not os.path.exists(self.font_path):
            logger.warning(f'Font not found: {self.font_path}')
            return

        # Load icons
        if os.path.exists(self.gold_icon_path):
            self.gold_icon_base = cv2.imread(self.gold_icon_path)
        if os.path.exists(self.elixir_icon_path):
            self.elixir_icon_base = cv2.imread(self.elixir_icon_path)

        sizes = [30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 100, 110, 120]
        for size in sizes:
            try:
                font = ImageFont.truetype(self.font_path, size)
                self.digit_templates[size] = {}
                for digit in '0123456789':
                    left, top, right, bottom = font.getbbox(digit)
                    w = right - left
                    h = bottom - top
                    img = Image.new('L', (w + 10, h + 10), 255)
                    draw = ImageDraw.Draw(img)
                    draw.text((-left + 5, -top + 5), digit, font = font, fill = 0)
                    img_np = np.array(img)
                    _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
                    coords = cv2.findNonZero(255 - thresh)
                    if coords is not None:
                        x, y, cw, ch = cv2.boundingRect(coords)
                        crop = thresh[y:y + ch, x:x + cw]
                        self.digit_templates[size][digit] = crop
            except Exception as e:
                logger.debug(f'Error generating template for size {size}: {e}')

        self.generated = True

    def preprocess_image(self, img_input):
        '''Convert PIL or numpy BGR/RGB image to scaled thresholded binary mask (black text on white).'''
        if isinstance(img_input, Image.Image):
            img_np = np.array(img_input)
        else:
            img_np = img_input

        if img_np is None or img_np.size == 0:
            return None

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np

        scaled = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(scaled, 170, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def recognize_digits(self, binary_img):
        '''Recognize digits from thresholded binary image (black text on white).'''
        if binary_img is None or binary_img.size == 0:
            return ''
        self.generate_templates()
        if not self.digit_templates:
            return ''

        inv_img = 255 - binary_img
        contours, _ = cv2.findContours(inv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 20 or w < 5:
                continue

            roi = binary_img[y:y + h, x:x + w]
            best_score = 0.0
            best_digit = ''

            for size, templates in self.digit_templates.items():
                for digit, template in templates.items():
                    t_h, t_w = template.shape[:2]
                    if abs((w / h) - (t_w / t_h)) > 0.45:
                        continue
                    try:
                        resized_roi = cv2.resize(roi, (t_w, t_h))
                        res = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                        score = float(res[0][0])
                        if score > best_score:
                            best_score = score
                            best_digit = digit
                    except Exception:
                        pass

            if best_score > 0.45:
                digit_candidates.append((x, y, h, best_digit, best_score))

        if not digit_candidates:
            return ''

        # Group candidates by similar baseline Y to reject vertical noise
        y_groups = []
        for c in digit_candidates:
            placed = False
            for g in y_groups:
                if abs(g[0][1] - c[1]) <= 25:
                    g.append(c)
                    placed = True
                    break
            if not placed:
                y_groups.append([c])

        best_group = max(y_groups, key=lambda g: len(g))
        best_group.sort(key = lambda k: k[0])
        return ''.join([d[3] for d in best_group])

    def parse_number(self, text):
        try:
            text = (text.replace('$', '5').replace('O', '0').replace('o', '0')
                    .replace('S', '5').replace('l', '1').replace('I', '1')
                    .replace('i', '1').replace('B', '8').replace('Z', '2').replace('?', '7'))
            digits = ''.join(re.findall(r'\d+', text))
            return int(digits) if digits else 0
        except Exception:
            return 0

    def read_battle_loot(self, screen_img):
        '''
        Extracts available raid loot (Gold, Elixir, Dark Elixir) from top-left during battle matchmaking.
        Uses icon-anchored template matching as primary, with geometric relative fallback.
        '''
        if screen_img is None or screen_img.size == 0:
            return 0, 0, 0

        self.generate_templates()
        h_s, w_s = screen_img.shape[:2]

        # Top-left search region (0% to 35% height, 0% to 32% width)
        top_left = screen_img[0:int(0.35 * h_s), 0:int(0.32 * w_s)]

        # 1. Try Icon-Anchored Detection
        gold_val, elixir_val, de_val = 0, 0, 0
        if self.gold_icon_base is not None and self.elixir_icon_base is not None:
            try:
                scale = w_s / 1920.0
                g_icon = cv2.resize(self.gold_icon_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                e_icon = cv2.resize(self.elixir_icon_base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                g_res = cv2.matchTemplate(top_left, g_icon, cv2.TM_CCOEFF_NORMED)
                _, g_max, _, g_max_loc = cv2.minMaxLoc(g_res)

                e_res = cv2.matchTemplate(top_left, e_icon, cv2.TM_CCOEFF_NORMED)
                _, e_max, _, e_max_loc = cv2.minMaxLoc(e_res)

                if g_max > 0.70 and e_max > 0.70:
                    gx = g_max_loc[0] + g_icon.shape[1] + int(4 * scale)
                    gy = max(0, g_max_loc[1] - int(2 * scale))
                    gw = int(220 * scale)
                    gh = g_icon.shape[0] + int(6 * scale)

                    ex = e_max_loc[0] + e_icon.shape[1] + int(4 * scale)
                    ey = max(0, e_max_loc[1] - int(2 * scale))
                    ew = int(220 * scale)
                    eh = e_icon.shape[0] + int(6 * scale)

                    dy = ey + (ey - gy)
                    dx = ex
                    dw = ew
                    dh = eh

                    g_crop = top_left[gy:gy+gh, gx:gx+gw]
                    e_crop = top_left[ey:ey+eh, ex:ex+ew]
                    d_crop = top_left[dy:dy+dh, dx:dx+dw]

                    gold_val = self.parse_number(self.recognize_digits(self.preprocess_image(g_crop)))
                    elixir_val = self.parse_number(self.recognize_digits(self.preprocess_image(e_crop)))
                    de_val = self.parse_number(self.recognize_digits(self.preprocess_image(d_crop)))

                    if gold_val > 0 or elixir_val > 0:
                        return gold_val, elixir_val, de_val
            except Exception as e:
                logger.debug(f'Icon-anchored detection error: {e}')

        # 2. Geometric Relative Fallback
        gx1, gy1, gw, gh = int(165 * w_s / 1920), int(136 * h_s / 1080), int(200 * w_s / 1920), int(44 * h_s / 1080)
        ex1, ey1, ew, eh = int(165 * w_s / 1920), int(176 * h_s / 1080), int(200 * w_s / 1920), int(44 * h_s / 1080)
        dx1, dy1, dw, dh = int(165 * w_s / 1920), int(216 * h_s / 1080), int(200 * w_s / 1920), int(44 * h_s / 1080)

        g_bin = self.preprocess_image(screen_img[gy1:gy1+gh, gx1:gx1+gw])
        e_bin = self.preprocess_image(screen_img[ey1:ey1+eh, ex1:ex1+ew])
        d_bin = self.preprocess_image(screen_img[dy1:dy1+dh, dx1:dx1+dw])

        gold_val = self.parse_number(self.recognize_digits(g_bin))
        elixir_val = self.parse_number(self.recognize_digits(e_bin))
        de_val = self.parse_number(self.recognize_digits(d_bin))

        # 3. VisionService / Tesseract fallback if still 0
        if gold_val == 0 and elixir_val == 0:
            try:
                from app.services.vision import VisionService
                roi = (int(0.06 * w_s), int(0.10 * h_s), int(0.18 * w_s), int(0.18 * h_s))
                grouped = VisionService.extract_grouped_numbers_in_region(screen_img, roi, white_text = True)
                if len(grouped) >= 1:
                    gold_val = self.parse_number(grouped[0].text)
                if len(grouped) >= 2:
                    elixir_val = self.parse_number(grouped[1].text)
                if len(grouped) >= 3:
                    de_val = self.parse_number(grouped[2].text)
            except Exception as e:
                logger.debug(f'VisionService fallback error: {e}')

        return gold_val, elixir_val, de_val
