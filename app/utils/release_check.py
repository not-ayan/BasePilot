"""Exercise a packaged release without capturing a game or sending input."""

from __future__ import annotations
import importlib
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback

def run_release_check(report_path):
    """Write a machine-readable report and return a process exit code.

    Called before GUI/logger imports. All user data and Qt settings are isolated,
    and the normal startup/display-restore/autostart paths are never entered.
    """
    report_path = Path(report_path).resolve()
    report = {'ok': False, 'frozen': bool(getattr(sys, 'frozen', False)), 'checks': {}}
    try:
        with ExitStack() as cleanup:
            scratch = cleanup.enter_context(tempfile.TemporaryDirectory(prefix='basepilot-release-check-'))
            cleanup.callback(logging.shutdown)
            os.environ['LOCALAPPDATA'] = scratch
            os.environ['BASEPILOT_PROFILE'] = 'release-check'
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            from app import __version__
            from app.utils.common import get_resource_path
            from app.utils.tesseract_env import configure_tesseract
            import cv2
            import numpy as np
            import pytesseract
            report['version'] = __version__
            required = ['assets/basepilot_logo.ico', 'assets/basepilot_logo.png', 'assets/vision/clash_digits.png', 'assets/vision/clash_digits.json', 'templates/events/reward_header.png', 'templates/16_9/clanmenu.png', 'templates/16_9/clanexit.png', 'templates/16_9/nextbase.png']
            for name in required:
                if get_resource_path(name).is_file():
                    continue
                raise RuntimeError(f"Missing release resource: {name}")
            image_count = 0
            for aspect in ('16_9', '16_10'):
                templates = list(get_resource_path(f"templates/{aspect}").glob('*.png'))
                if not templates:
                    raise RuntimeError(f"No bundled templates for {aspect}")
                for template in templates:
                    if cv2.imread(str(template)) is not None:
                        continue
                    raise RuntimeError(f"Cannot decode template: {template.name}")
                image_count += len(templates)
            report['checks']['resources'] = {'templates_decoded': image_count}
            from app.services.clash_digits import load_digit_bank
            _, digit_labels, _ = load_digit_bank()
            report['checks']['price_digits'] = {'digits': sorted(set(map(int, digit_labels))), 'glyph_variants': len(digit_labels)}
            modules = ('app.core.bot', 'app.core.strategies', 'app.core.upgrader', 'app.core.clan', 'app.core.loot_filter', 'app.services.window', 'app.services.input', 'app.services.vision', 'app.services.display', 'app.ui.qt.point_picker', 'app.ui.qt.template_capture')
            for name in modules:
                importlib.import_module(name)
            report['checks']['feature_imports'] = list(modules)
            from unittest.mock import Mock, patch
            from app.core.bot import Bot
            with patch('socket.socket.connect', side_effect=RuntimeError('Network attempted')):
                bot = Bot()
                bot.window = Mock()
                bot.window.find_window.return_value = True
                bot._reset_loot_session = Mock()
                bot._emit_loot_update = Mock()
                bot._ensure_correct_village = Mock()
                bot._run_loop = Mock()
                bot._finish_unread_raid = Mock()
                bot.start('valkyries', 0)
                bot._run_loop.assert_called_once_with('valkyries', 0, False, False, False)
            report['checks']['free_access'] = {'unlimited_start_without_activation': True, 'network_disabled': True}
            if not report['frozen'] and not os.environ.get('TESSERACT_CMD'):
                staged = get_resource_path('tesseract_bundle/tesseract.exe')
                if staged.is_file():
                    os.environ['TESSERACT_CMD'] = str(staged)
            configure_tesseract()
            tess = Path(pytesseract.pytesseract.tesseract_cmd)
            if not tess.is_file():
                raise RuntimeError(f"Tesseract executable missing: {tess}")
            if report['frozen'] and tess.parent != get_resource_path('tesseract'):
                raise RuntimeError('Self-test must use bundled Tesseract')
            if not ((tess.parent / 'tessdata') / 'eng.traineddata').is_file():
                raise RuntimeError('English OCR traineddata missing')
            canvas = np.full((120, 700, 3), 255, np.uint8)
            cv2.putText(canvas, '1234567890', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (0, 0, 0), 4, cv2.LINE_AA)
            text = pytesseract.image_to_string(canvas, config='--psm 7 -c tessedit_char_whitelist=0123456789').strip()
            if text != '1234567890':
                raise RuntimeError(f"Bundled OCR smoke returned {text!r}")
            report['checks']['ocr'] = {'tesseract_version': str(pytesseract.get_tesseract_version()), 'digits': text}
            from PySide6.QtCore import QSettings
            from PySide6.QtWidgets import QApplication
            from app.ui.qt.branding import apply_app_icon
            from app.ui.qt.main_window import MainWindow
            from app.ui.qt.theme import apply_theme
            QSettings.setDefaultFormat(QSettings.Format.IniFormat)
            QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, scratch)
            QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.SystemScope, scratch)
            application = QApplication([])
            application.setApplicationName('BasePilot release check')
            apply_app_icon(application)
            apply_theme(application)
            window = MainWindow()
            application.processEvents()
            if window._stack.count() != 4:
                raise RuntimeError('Expected Run, Settings, Players and Logs pages')
            report['checks']['gui'] = {'pages': window._stack.count()}
            window.close()
            application.processEvents()
            report['ok'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text((json.dumps(report, indent=2) + """
"""), encoding='utf-8')
    if report['ok']:
        return 0
    return 1

