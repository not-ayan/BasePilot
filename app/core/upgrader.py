#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in ScanResult at SETUP_ANNOTATIONS (offset 14); Python 3.13]
#   - unsupported_opcode: no handler for SETUP_ANNOTATIONS [in PassResult at SETUP_ANNOTATIONS (offset 14); Python 3.13]
#   - stack_underflow: an operand is missing from the stack [in UpgradeAdvisor.scan; Python 3.13]
"""Upgrade planning and execution for current Build cards and legacy menus.

The current UI needs one inventory scan. Subsequent choices refresh builder and
resource readings independently, reuse known prices, and find only the selected
card. A fresh card, matching target name, exact confirmation price/currency,
recognized green Confirm button, and a verified builder drop still gate every
purchase. Cached affordability never authorizes a click by itself. Legacy menus
retain their cross-scan corroboration path; dry mode never starts an upgrade.
"""

import difflib
import re
import cv2
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple
from types import SimpleNamespace
from app.utils.common import get_resource_path
from app.config import ASPECT_16_10, ASPECT_16_9
from app.core.upgrade_menu import SECTION_IN_PROGRESS, detect_build_menu, UpgradeRow, merge_rows, parse_builder_chip, parse_builder_menu
from app.core.upgrade_policy import MODE_MAXER, MODE_RUSHER, UpgradePolicy, can_pay, choose_upgrade
from app.core.village_state import read_hud_resources_stable
from app.services.clash_digits import read_price_digits
from app.core.ui_navigation import find_build_button, find_buildings_tab, find_ui_label, is_village_hud
from app.services.vision import VisionService
from app.utils.logger import setup_logger
logger = setup_logger('Upgrader')
MODE_OFF = 'off'
MODE_DRY = 'dry'
AUTO_UPGRADE_MODES = (MODE_OFF, MODE_DRY, MODE_MAXER, MODE_RUSHER)
LIVE_UPGRADE_MODES = (MODE_MAXER, MODE_RUSHER)
_BUILDER_TEMPLATES = ('builder.png', 'gbuilder.png')
_SCROLL_BASELINE = {ASPECT_16_9: (1305, 605), ASPECT_16_10: (1305, 672)}
_MAX_SCROLL_STEPS = 12
_WHEEL_CLICKS = 2
_STALE_STEPS_AT_LIST_END = 2
_RESET_TO_TOP_WHEEL_CLICKS = ((_MAX_SCROLL_STEPS * _WHEEL_CLICKS) + 4)
_EXEC_ROW_STABLE_TOL = 30
_EXEC_LABEL_FUZZY_MIN = 0.85
_EXEC_UPGRADE_WORD_POLLS = 8
_EXEC_MAX_UPGRADE_CLICKS = 2
_EXEC_RED_VETO_FRACTION = 0.05
_EXEC_COOLDOWN_SECONDS = 1800
_EXEC_VERIFY_POLLS = 5
_EXEC_UPGRADE_BAND_Y_FRAC = 0.66
@dataclass(frozen=True)
class ScanResult:
    rows: List[UpgradeRow]
    chip: Optional[Tuple[(int, int)]]
    hud: Optional[Tuple[(Optional[int], Optional[int], Optional[int])]]

@dataclass(frozen=True)
class PassResult:
    """One advisor pass, for the bot loop and the UI status panel."""
    scan: Optional[ScanResult]
    pick: Optional[UpgradeRow]
    executed: bool
    note: str
    collecting: bool = False
    retryable: bool = False
    from_cache: bool = False

def _row_summary(rows):
    parts = []
    for r in rows:
        if r.cost is not None:
            tag = f"{r.cost}{(r.resource or '?')[0]}"
        else:
            tag = 'RED' if r.affordable is False else '?'
        parts.append(f"{r.label}{(' / ' + r.detail) if r.detail else ''}[{(r.section or '-')[:4]}:{tag}]")
    return ' | '.join(parts)

def _compact(text):
    return ''.join((ch for ch in (text or '').lower() if ch.isalnum()))

def _labels_match(a, b, min_ratio=_EXEC_LABEL_FUZZY_MIN):
    return difflib.SequenceMatcher(None, _compact(a), _compact(b)).ratio() >= min_ratio

def _candidate_key(row):
    return (_compact(row.label), _compact(row.detail), row.cost, row.resource)

def _dump_debug_frame(frame, prefix):
    """Save a frame to the debug dir; returns the path or None. Never raises."""
    try:
        import cv2 as _cv2
        from app.utils.common import get_user_app_data_dir
        dbg = (get_user_app_data_dir() / 'debug')
        dbg.mkdir(parents=True, exist_ok=True)
        path = (dbg / f"{prefix}_{int(time.time())}.jpg")
        _cv2.imwrite(str(path), frame, [_cv2.IMWRITE_JPEG_QUALITY, 88])
        logger.info('Upgrade exec: frame saved to %s', path)
        return path
    except Exception:
        logger.debug('Upgrade exec: debug frame dump failed', exc_info=True)
        return None

class UpgradeAdvisor:
    """Scans the builder popup, decides which upgrade to start, and (in live modes)
    starts it. Keep ONE instance per bot session — it remembers per-row execution
    cooldowns so a row that fails to execute is not hammered every pass."""
    def __init__(self, window, input_service, vision, config, stop_event):
        self.window = window
        self.input = input_service
        self.vision = vision
        self.config = config
        self.stop_event = stop_event
        self._exec_cooldowns = {}
        self._prev_scan_rows = None
        self._last_pass_collecting = False
        self._menu_scroll_point = None
        self._using_build_menu = False
        self._cached_rows = None
        self._cached_at = 0
        self._cache_new_ui = False
        self._last_scan_new_ui = False

    def has_cached_inventory(self):
        return (self._cache_new_ui and self._cached_rows is not None and (time.monotonic() - self._cached_at) < 600)

    def _cached_snapshot(self, reserve_builders):
        frame = self._frame()
        if frame is None:
            return None
        if find_buildings_tab(frame, self.vision)[0] is not None:
            self._close_popup()
            frame = self._frame()
        if frame is not None:
            if self.vision.find_template(frame, 'mbuilder.png', region=VisionService.bottom_half_region(frame))[0] is not None:
                return None
        if not is_village_hud(frame, self.vision):
            return None
        chip = parse_builder_chip(frame)
        hud = None
        if chip is not None and chip[0] > reserve_builders:
            hud = read_hud_resources_stable(self._frame, self.stop_event.wait)
        return ScanResult(rows=list(self._cached_rows), chip=chip, hud=hud)

    def _remove_started_candidate(self, pick):
        remaining = []
        for row in (self._cached_rows or []):
            same = (bool(row.detail) == bool(pick.detail) and ((not row.detail) or _labels_match(row.detail, pick.detail, 0.8)) and _labels_match(row.label, pick.label, 0.8) and row.cost == pick.cost and row.resource == pick.resource)
            if not same:
                remaining.append(row)
            else:
                count = re.search('(?i)\\bx\\s*(\\d+)\\b', row.label)
                if not count:
                    continue
                if not int(count.group(1)) > 1:
                    continue
                label = ((row.label[:count.start(1)] + str((int(count.group(1)) - 1))) + row.label[count.end(1):])
                remaining.append(replace(row, label=label))
        self._cached_rows = remaining

    def _frame(self):
        frame = self.window.screenshot()
        if frame is None or frame.size == 0:
            return None
        self.config.set_target_size_from_frame(frame)
        return frame

    def _scroll_point(self):
        if self._menu_scroll_point is not None:
            return self._menu_scroll_point
        ref = _SCROLL_BASELINE.get(self.config.aspect_key)
        if ref is None:
            ref = _SCROLL_BASELINE[ASPECT_16_10]
        x, y = self.config.scale_point([ref[0], ref[1]])
        return (int(x), int(y))

    def _close_popup(self):
        try:
            self.input.click(*self.config.get_point('empty'), pause=0.3)
        except Exception:
            logger.debug('Upgrade scan: empty-point tap failed', exc_info=True)
        self._menu_scroll_point = None
        self._using_build_menu = False
        self.stop_event.wait(0.6)

    def _escape_ui(self, reason):
        """Back out of whatever screen the executor is on: exit X if visible, else
    empty-ground taps. Never clicks okay.png — an unknown dialog's Okay can
    confirm something."""
        logger.info('Upgrade exec: escaping UI (%s)', reason)
        for _ in range(2):
            frame = self._frame()
            if frame is None:
                return None
            ex, ey = self.vision.find_template(frame, 'exit.png')
            if ex:
                self.input.click(ex, ey, pause=0.3)
            else:
                self._close_popup()
            if not self.stop_event.wait(0.3):
                continue
            return None

    def _open_builder_popup(self):
        """Open legacy builder list or the recognized Buildings & Heroes panel."""
        self._menu_scroll_point = None
        self._using_build_menu = False
        frame = self._frame()
        if frame is None:
            return False
        tab = find_buildings_tab(frame, self.vision)
        if tab[0] is not None:
            if detect_build_menu(frame) is None:
                self.input.click(*tab, pause=0.4)
                if self.stop_event.wait(0.4):
                    return False
                frame = self._frame()
            layout = detect_build_menu(frame)
            if layout is None:
                self._close_popup()
                return False
            self._using_build_menu = True
            self._menu_scroll_point = layout.scroll_point
            self.input.scroll(*self._scroll_point(), _RESET_TO_TOP_WHEEL_CLICKS, upward=True)
            return (not self.stop_event.wait(0.6))
        if self.vision.find_template(frame, 'mbuilder.png', region=VisionService.bottom_half_region(frame))[0] is not None:
            logger.info('Upgrade advisor: Builder Base is open; Home Village upgrades skipped')
            return False
        home_roi = VisionService.bottom_half_region(frame)
        ax, _ay = self.vision.find_template(frame, 'attack.png', region=home_roi)
        if not ax and not is_village_hud(frame, self.vision):
            logger.info('Upgrade advisor: not on the home screen (Attack missing) — skipping')
            return False
        top_roi = VisionService.top_half_region(frame)
        for name in _BUILDER_TEMPLATES:
            bx, by = self.vision.find_template(frame, name, region=top_roi)
            if not bx:
                continue
            self.input.click(bx, by, pause=0.5)
            if self.stop_event.wait(0.5):
                return False
            self.input.scroll(*self._scroll_point(), _RESET_TO_TOP_WHEEL_CLICKS, upward=True)
            return (not self.stop_event.wait(1.0))
        bx, by = find_build_button(frame, self.vision)
        if bx is not None:
            self.input.click(bx, by, pause=0.5)
            for _ in range(3):
                if self.stop_event.wait(0.4):
                    return False
                panel = self._frame()
                layout = detect_build_menu(panel)
                if layout is None:
                    tab = find_buildings_tab(panel, self.vision)
                    if tab[0] is not None:
                        self.input.click(*tab, pause=0.4)
                        if self.stop_event.wait(0.4):
                            return False
                        layout = detect_build_menu(self._frame())
                if layout is None:
                    continue
                self._using_build_menu = True
                self._menu_scroll_point = layout.scroll_point
                self.input.scroll(*self._scroll_point(), _RESET_TO_TOP_WHEEL_CLICKS, upward=True)
                return (not self.stop_event.wait(0.6))
            logger.info('Upgrade advisor: Buildings & Heroes panel unrecognized; select that Build tab before retrying')
            self._close_popup()
            return False
        logger.info('Upgrade advisor: builder portrait not found — skipping')
        return False

    def scan(self, reserve_builders=0):
        """From the home screen: open the builder popup, parse every scroll position,
    close it. Returns :class:`ScanResult` or ``None`` when preconditions fail."""
        frame = self._frame()
        if frame is None:
            return None
        if find_buildings_tab(frame, self.vision)[0] is not None:
            self._close_popup()
            frame = self._frame()
            if frame is None:
                return None
        if self.vision.find_template(frame, 'mbuilder.png', region=VisionService.bottom_half_region(frame))[0] is not None:
            logger.info('Upgrade advisor: Builder Base is open; Home Village upgrades skipped')
            return None
        home_roi = VisionService.bottom_half_region(frame)
        ax, _ay = self.vision.find_template(frame, 'attack.png', region=home_roi)
        if not ax and not is_village_hud(frame, self.vision):
            logger.info('Upgrade scan: not on the home screen (Attack missing) — skipping')
            return None
        chip = parse_builder_chip(frame)
        if chip is not None and chip[0] <= reserve_builders:
            return ScanResult(rows=[], chip=chip, hud=None)
        hud = read_hud_resources_stable(self._frame, self.stop_event.wait) if chip is not None else None
        if chip is None:
            _dump_debug_frame(frame, 'chipfail')
        if hud is None or chip is None or chip[0] == 0:
            return ScanResult(rows=[], chip=chip, hud=hud)
        if not self._open_builder_popup():
            return None
        rows = []
        section = ''
        stale = 0
        previous_body = None
        for _step in range(_MAX_SCROLL_STEPS):
            if self.stop_event.is_set():
                break
            frame = self._frame()
            if frame is None:
                break
            layout = detect_build_menu(frame) if self._using_build_menu else None
            if layout is not None:
                left = min((card[0] for card in layout.cards))
                body = frame[(layout.title_bottom + 3):round((frame.shape[0] * 0.87)), (left + 3):(frame.shape[1] - 3)]
                if previous_body is not None and body.shape == previous_body.shape and float(cv2.absdiff(body, previous_body).mean()) < 0.8:
                    break
                previous_body = body.copy()
            parsed = parse_builder_menu(frame, initial_section=section, batch_labels=self._using_build_menu)
            if parsed:
                section = parsed[-1].section
            before = len(rows)
            rows = merge_rows(rows, parsed)
            stale = (stale + 1) if len(rows) == before else 0
            if stale >= _STALE_STEPS_AT_LIST_END:
                break
            self.input.scroll(*self._scroll_point(), _WHEEL_CLICKS)
            if not self.stop_event.wait(1.0):
                continue
            break
        self._last_scan_new_ui = self._using_build_menu
        self._close_popup()
        return ScanResult(rows=rows, chip=chip, hud=hud)

    def _cooldown_active(self, row):
        for (label, detail, cost, resource), when in self._exec_cooldowns.items():
            if not row.cost == cost:
                continue
            if not row.resource == resource:
                continue
            if not bool(row.detail) == bool(detail):
                continue
            if detail and not _labels_match(row.detail, detail, 0.8):
                continue
            if not _labels_match(row.label, label):
                continue
            if not (time.monotonic() - when) < _EXEC_COOLDOWN_SECONDS:
                continue
            return True
        return False

    def _rows_without_cooldowns(self, rows):
        kept = []
        for row in rows:
            if row.section != SECTION_IN_PROGRESS and self._cooldown_active(row):
                logger.info('Upgrade advisor: %r is on execution cooldown — skipping this pass', row.label)
            else:
                kept.append(row)
        return kept

    def _corroborate_rows(self, rows):
        """Cross-scan corroboration: a costed row is pickable only when the PREVIOUS
    scan also saw it at the exact same cost. One-off OCR bargains (dropped leading
    digits: 'XBow 350000' for a 9.5M row) never repeat a value, so they die here
    instead of wasting an execution attempt; and a row whose resource icon misread
    this scan adopts the previous scan's resolved resource (the dark-elixir read
    is intermittent — this is what keeps Barbarian King pickable)."""
        prev = self._prev_scan_rows
        self._prev_scan_rows = rows
        self._last_pass_collecting = prev is None
        if prev is None:
            logger.info('Upgrade advisor: first scan this session — collecting, no pick yet')
            return []
        out = []
        for row in rows:
            if row.cost is None:
                out.append(row)
            else:
                match = None
                for p in prev:
                    if not p.cost == row.cost:
                        continue
                    if not bool(p.detail) == bool(row.detail):
                        continue
                    if row.detail and not _labels_match(row.detail, p.detail, 0.8):
                        continue
                    if not _labels_match(row.label, p.label, 0.7):
                        continue
                    if p.resource and row.resource and p.resource != row.resource:
                        continue
                    match = p
                    break
                if match is None:
                    logger.debug('Upgrade advisor: %r (%s) not corroborated by the previous scan — holding', row.label, row.cost)
                else:
                    if row.resource is None and match.resource is not None:
                        row = replace(row, resource=match.resource)
                    out.append(row)
        return out

    def run_pass(self, mode, reserve_builders=1, order=None):
        """Plan from one inventory, refresh budgets, then validate the chosen dialog."""
        if mode not in AUTO_UPGRADE_MODES or mode == MODE_OFF:
            return PassResult(None, None, False, 'auto-upgrade off')
        started = time.monotonic()
        cached = self.has_cached_inventory()
        scan = self._cached_snapshot(reserve_builders) if cached else self.scan(reserve_builders=reserve_builders)
        if scan is None:
            return PassResult(None, None, False, 'waiting for a readable home screen', retryable=True, from_cache=cached)
        if not cached:
            logger.info('Upgrade inventory (%.1fs): %d rows, builders=%s, HUD=%s', (time.monotonic() - started), len(scan.rows), scan.chip, scan.hud)
            if scan.rows:
                logger.info('Upgrade inventory rows: %s', _row_summary(scan.rows))
            if scan.rows and self._last_scan_new_ui:
                self._cached_rows = list(scan.rows)
                self._cached_at = time.monotonic()
                self._cache_new_ui = True
            else:
                self._cache_new_ui = False
                self._cached_rows = None
        else:
            logger.info('Upgrade budgets (cached inventory): builders=%s, HUD=%s', scan.chip, scan.hud)
        free = scan.chip[0] if scan.chip else None
        if free is None:
            return PassResult(scan, None, False, 'builder count unreadable; retrying', retryable=True, from_cache=cached)
        if mode in LIVE_UPGRADE_MODES and free <= reserve_builders:
            return PassResult(scan, None, False, 'all available builders are busy or reserved', from_cache=cached)
        if scan.hud is None:
            return PassResult(scan, None, False, 'resource balances unreadable; retrying', retryable=True, from_cache=cached)
        kwargs = {} if order else {}
        policy = UpgradePolicy(mode=MODE_RUSHER if mode == MODE_RUSHER else MODE_MAXER, reserve_builders=max(0, int(reserve_builders)), **kwargs)
        rows = self._rows_without_cooldowns(scan.rows)
        modern = self._cache_new_ui and self._cached_rows is not None
        if modern:
            rows = [
                row for row in rows
                if row.section != SECTION_IN_PROGRESS and row.cost is not None and row.resource is not None
            ]
        else:
            rows = self._corroborate_rows(rows)
            if self._last_pass_collecting:
                return PassResult(scan, None, False, 'first legacy scan saved for price corroboration', collecting=True)
        pick = choose_upgrade(rows, scan.hud, free, policy)
        if pick is None:
            for index, name in enumerate(('gold', 'elixir', 'dark_elixir')):
                if scan.hud[index] is not None:
                    continue
                if not any(((row.resource == name and row.cost is not None and row.section != SECTION_IN_PROGRESS) for row in rows)):
                    continue
                _res0 = index
                _res0.append(name)
                continue
            if missing and mode in LIVE_UPGRADE_MODES and free > reserve_builders:
                return PassResult(scan, None, False, ('waiting to read ' + ', '.join(missing)), retryable=True, from_cache=cached)
            logger.info('Upgrade advisor: no affordable candidate remains with current balances and builder reserve')
            return PassResult(scan, None, False, 'no affordable upgrade remains', from_cache=cached)
        if mode == MODE_DRY:
            logger.info('Upgrade advisor (dry): WOULD START %r (cost %s %s)', pick.label, pick.cost, pick.resource)
            return PassResult(scan, pick, False, ('preview: would start ' + pick.label), from_cache=cached)
        executed = self._execute(pick, scan)
        if executed and modern:
            self._remove_started_candidate(pick)
        elif not executed:
            self._exec_cooldowns[_candidate_key(pick)] = time.monotonic()
        note = ('started ' + pick.label) if executed else ('could not verify ' + pick.label)
        return PassResult(scan, pick, executed, note, from_cache=cached)

    def _row_matches_pick(self, row, pick):
        if bool(row.detail) != bool(pick.detail):
            return False
        if row.detail and not _labels_match(row.detail, pick.detail, 0.8):
            return False
        if row.section == SECTION_IN_PROGRESS:
            return False
        if row.affordable is not True or row.cost is None:
            return False
        if pick.cost is not None and row.cost != pick.cost:
            return False
        if pick.resource and row.resource and pick.resource != row.resource:
            return False
        exact_numbers = pick.cost is not None and row.cost == pick.cost and pick.resource is not None and row.resource == pick.resource
        if exact_numbers:
            return _labels_match(row.label, pick.label, 0.6)
        return _labels_match(row.label, pick.label, _EXEC_LABEL_FUZZY_MIN)

    def _find_pick_row_once(self, pick, section, near_y=None):
        """Parse the current popup frame; returns (point, section) — point None on miss.
    ``section`` carries the running header across scroll positions. With ``near_y``
    (a prior sighting's y), a row with the pick's exact cost at ~the same height
    also counts: at confirmation time identity is position + cost — the label can
    respell arbitrarily between frames (live repro: 'Bombx' confirmed as 'BornbxB')."""
        frame = self._frame()
        if frame is None:
            return (None, section)
        rows = parse_builder_menu(frame, initial_section=section, batch_labels=True, target=pick)
        if rows:
            section = rows[-1].section
        for row in rows:
            if not self._row_matches_pick(row, pick):
                continue
            return (row.center, section)
        if near_y is not None:
            tol = self.config.scale_scalar(_EXEC_ROW_STABLE_TOL)
            for row in rows:
                if not row.section != SECTION_IN_PROGRESS:
                    continue
                if not row.affordable is True:
                    continue
                if row.cost is None:
                    continue
                if not row.cost == pick.cost:
                    continue
                if not row.resource == pick.resource:
                    continue
                if not bool(row.detail) == bool(pick.detail):
                    continue
                if row.detail and not _labels_match(row.detail, pick.detail, 0.8):
                    continue
                if not _labels_match(row.label, pick.label, 0.6):
                    continue
                if not abs((row.line_y - near_y)) <= tol:
                    continue
                return (row.center, section)
        return (None, section)

    def _execute(self, pick, scan):
        """Click the picked row and walk the Upgrade chain. True only when the builder
    chip verifies the start (free count dropped below the scan-time reading)."""
        free_before = scan.chip[0] if scan.chip else None
        if free_before is None or free_before <= 0:
            return False
        if not self._open_builder_popup():
            return False
        row_pt = None
        candidate = None
        confirm_misses = 0
        section = ''
        scrolls = 0
        for _ in range((_MAX_SCROLL_STEPS * 3)):
            if self.stop_event.is_set():
                return False
            found, section = self._find_pick_row_once(pick, section, near_y=candidate[1] if candidate else None)
            if found:
                if candidate and abs((found[1] - candidate[1])) <= self.config.scale_scalar(_EXEC_ROW_STABLE_TOL):
                    row_pt = found
                    break
                candidate = found
                confirm_misses = 0
                if self.stop_event.wait(0.25):
                    return False
            elif candidate is not None and confirm_misses < 2:
                confirm_misses += 1
                if self.stop_event.wait(0.3):
                    return False
            else:
                candidate = None
                confirm_misses = 0
                scrolls += 1
                if scrolls >= _MAX_SCROLL_STEPS:
                    break
                self.input.scroll(*self._scroll_point(), _WHEEL_CLICKS)
                if not self.stop_event.wait(1.0):
                    continue
                return False
        if not row_pt:
            logger.info('Upgrade exec: row %r not re-found in the popup — cooling it down and closing', pick.label)
            self._exec_cooldowns[_candidate_key(pick)] = time.monotonic()
            self._close_popup()
            return False
        if self.stop_event.wait(0.4):
            return False
        fresh, section = self._find_pick_row_once(pick, section, near_y=row_pt[1])
        if fresh is None:
            logger.info('Upgrade exec: row %r lost right before the click — closing without clicking', pick.label)
            self._close_popup()
            return False
        row_pt = fresh
        logger.info('Upgrade exec: clicking row %r at %s', pick.label, row_pt)
        self.input.click(*row_pt, pause=0.6)
        started = self._walk_upgrade_chain(pick, free_before=free_before)
        if not started:
            self._exec_cooldowns[_candidate_key(pick)] = time.monotonic()
            self._escape_ui('upgrade chain did not complete')
            return False
        verify_frame = self._frame()
        self._close_popup()
        if self.stop_event.wait(0.8):
            return False
        if self._verify_builder_drop(free_before):
            logger.info('Upgrade exec: STARTED %r (cost %s %s) — builder chip dropped below %d free', pick.label, pick.cost, pick.resource, free_before)
            self._exec_cooldowns.pop(_candidate_key(pick), None)
            return True
        logger.warning('Upgrade exec: clicked the Upgrade chain for %r but the builder chip did not drop — cooling this row down', pick.label)
        if verify_frame is not None:
            _dump_debug_frame(verify_frame, 'upgexec_verifyfail')
        self._exec_cooldowns[_candidate_key(pick)] = time.monotonic()
        self._escape_ui('no builder-chip confirmation')
        return False

    @staticmethod
    def _word_is(text, target, min_ratio=0.75):
        return difflib.SequenceMatcher(None, _compact(text), target).ratio() >= min_ratio

    def _find_upgrade_word(self, frame):
        """Bottom-band OCR hunt for the button that commits the upgrade. Hero (and
    crafting) dialogs label it "Confirm"; building flows label it "Upgrade" — hunt
    Confirm first. A bare "Upgrade" with a "time" neighbor on the same line is the
    inert "Upgrade time" caption (live repro: clicked it twice on the Barbarian
    King dialog while the real button said Confirm). Returns (x, y) or None."""
        fh, fw = frame.shape[:2]
        y0 = int((fh * _EXEC_UPGRADE_BAND_Y_FRAC))
        region = (0, y0, fw, (fh - y0))
        words = []
        for kwargs in ({'brightness_floor': 228}, {'brightness_floor': 200}, {}, {'roi_upscale': 2}):
            try:
                words += self.vision.find_words_ocr(frame, region=region, white_text=True, preprocess=True, min_confidence=25, **kwargs)
            except Exception:
                logger.debug('Upgrade exec: word OCR failed (%s)', kwargs, exc_info=True)
            continue
        if not words:
            return None
        def has_time_neighbor(w):
            for other in words:
                if other is w:
                    continue
                same_line = abs(((other.top + (other.height / 2)) - (w.top + (w.height / 2)))) <= (max(w.height, other.height) * 0.7)
                gap = (other.left - (w.left + w.width))
                if not same_line:
                    continue
                if 0 <= gap:
                    if not gap <= (2.5 * max(8, w.height)):
                        continue
                else:
                    continue
                if not self._word_is(other.text, 'time'):
                    continue
                return True
            return False

        confirms = [w for w in words if self._word_is(w.text, 'confirm')]
        upgrades = [w for w in words if self._word_is(w.text, 'upgrade') if not (has_time_neighbor(w))]
        def resource_button(w):
            x0, x1 = max(0, (w.left - 8)), min(fw, ((w.left + w.width) + 8))
            y0, y1 = max(0, (w.top - (4 * w.height))), min(fh, (w.top + w.height))
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            blue = (
                (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130)
                & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 80)
            )
            return float(blue.mean()) < 0.15

        confirms = [w for w in confirms if resource_button(w)]
        upgrades = [w for w in upgrades if resource_button(w)]
        pool = confirms or upgrades
        if not pool:
            return None
        word = max(pool, key=(lambda w: w.top))
        wx, wy = word.left + (word.width // 2), word.top + (word.height // 2)
        px0 = max(0, (word.left - word.height))
        px1 = min(fw, (word.left + word.width))
        if word.top > (fh * 0.8) and not confirms:
            py0 = max(0, (word.top - (5 * word.height)))
            py1 = max((py0 + 1), (word.top - (2 * word.height)))
        else:
            py0 = word.top + word.height
            py1 = min(fh, (word.top + (3 * word.height)))
        redness = VisionService.red_hue_fraction(frame[py0:py1, px0:px1])
        if redness >= _EXEC_RED_VETO_FRACTION:
            logger.warning('Upgrade exec: Upgrade word found but cost zone reads red (%.2f) — vetoing the click', redness)
            return None
        return (wx, wy)

    def _selection_matches_pick(self, frame, pick):
        """After the row click: does the screen actually show the picked target? The
    selection bottom-bar titles it ("Eagle Artillery (Level 5)") and upgrade
    dialogs put it in the header ("Upgrade Barbarian King to Level 80?"). A row
    click on a still-easing list lands on the NEIGHBOR row (live repro: picked
    Eagle Artillery 13M, bought Elixir Storage 4M — user had to cancel at a 50%
    refund loss), so the name is now a hard gate on every commit click. Returns
    True / False / None (no name text found — treated as NOT matching)."""
        fh, fw = frame.shape[:2]
        words = []
        for y0f, y1f in ((0.6, 0.8), (0.02, 0.12)):
            region = (0, int((fh * y0f)), fw, int((fh * (y1f - y0f))))
            for kwargs in ({'brightness_floor': 200}, {}):
                try:
                    words += self.vision.find_words_ocr(frame, region=region, white_text=True, preprocess=True, min_confidence=20, **kwargs)
                except Exception:
                    logger.debug('Upgrade exec: name-gate OCR failed', exc_info=True)
                continue
        if not words:
            return None
        target = _compact((pick.detail or pick.label))
        lines = VisionService.cluster_ocr_boxes_by_y(words, 12)
        best = 0.0
        for line in lines:
            line = sorted(line, key=(lambda w: w.left))
            for i in range(len(line)):
                joined = ''
                for j in range(i, min((i + 3), len(line))):
                    joined += _compact(line[j].text)
                    if len(joined) < 3:
                        continue
                    ratio = difflib.SequenceMatcher(None, joined, target).ratio()
                    best = max(best, ratio)
        logger.info('Upgrade exec: name gate for %r — best screen match %.2f', pick.label, best)
        return best >= 0.6

    def _find_green_confirm_blob(self, frame):
        """Positional/color fallback for the commit button: dialogs place the green
    Confirm at a stable spot (x~0.70, y~0.83 on both hero and building screens),
    but its white-on-bright-green text reads on JPEG dumps yet intermittently NOT
    on live raw frames (live repro: Eagle confirm sat unclicked while offline the
    same frame OCR'd fine). Look for a large green blob in the dialog-button band;
    x is fenced to exclude the home Attack!/Shop buttons and the crafting screen's
    component buttons sit far above the y band. Returns (x, y) or None."""
        import cv2 as _cv2
        import numpy as _np
        fh, fw = frame.shape[:2]
        y0, y1 = int((fh * 0.78)), int((fh * 0.93))
        x0, x1 = int((fw * 0.6)), int((fw * 0.78))
        band = frame[y0:y1, x0:x1]
        if band.size == 0:
            return None
        hsv = _cv2.cvtColor(band, _cv2.COLOR_BGR2HSV)
        mask = _cv2.inRange(hsv, (40, 80, 120), (80, 255, 255))
        contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        best = None
        min_area = ((fw * fh) * 0.002)
        for c in contours:
            area = _cv2.contourArea(c)
            if area < min_area:
                continue
            if best is not None and not area > best[0]:
                continue
            bx, by, bw, bh = _cv2.boundingRect(c)
            best = (area, ((x0 + bx) + (bw // 2)), ((y0 + by) + (bh // 2)))
        if best is None:
            return None
        _area, cx, cy = best
        patch = frame[max(0, (cy - 30)):(cy + 30), max(0, (cx - 90)):(cx + 90)]
        if VisionService.red_hue_fraction(patch) >= _EXEC_RED_VETO_FRACTION:
            logger.warning('Upgrade exec: green-blob confirm candidate reads red — vetoing')
            return None
        return (cx, cy)

    def _find_crafting_confirm(self, frame, pick):
        """The selected seasonal component has its own mid-screen Confirm.

    Require the crafting footer, green button, exact price and currency.
    Other component Upgrade buttons and magic-item buttons are never used.
    """
        h, w = frame.shape[:2]
        footer = self.vision.find_words_ocr(frame, region=(round((w * 0.47)), round((h * 0.81)), round((w * 0.3)), round((h * 0.06))), preprocess=False, roi_upscale=2)
        if not any((self._word_is(word.text, 'crafting', 0.85) for word in footer)):
            return None
        words = self.vision.find_words_ocr(frame, region=(round((w * 0.775)), round((h * 0.19)), round((w * 0.11)), round((h * 0.49))), white_text=True, brightness_floor=200, roi_upscale=3)
        if not any((self._word_is(word.text, 'confirm', 0.75) for word in words)):
            words = self.vision.find_words_ocr(frame, region=(round((w * 0.775)), round((h * 0.19)), round((w * 0.11)), round((h * 0.49))), preprocess=False, roi_upscale=3)
        if not any((self._word_is(word.text, 'confirm', 0.75) for word in words)):
            art = cv2.imread(str(get_resource_path('templates/events/crafting_confirm.png')), 0)
            if art is not None:
                art = cv2.resize(art, (max(1, round(((art.shape[1] * h) / 1440))), max(1, round(((art.shape[0] * h) / 1440)))))
                rx, ry = round((w * 0.775)), round((h * 0.19))
                roi = cv2.cvtColor(frame[ry:round((h * 0.68)), rx:round((w * 0.885))], cv2.COLOR_BGR2GRAY)
                if roi.shape[0] >= art.shape[0] and roi.shape[1] >= art.shape[1]:
                    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, art, cv2.TM_CCOEFF_NORMED))
                    if score >= 0.82:
                        words = [SimpleNamespace(text='confirm', left=(rx + loc[0]), top=(ry + loc[1]), width=art.shape[1], height=art.shape[0])]
        for word in words:
            if not self._word_is(word.text, 'confirm', 0.75):
                continue
            x0, x1 = round((w * 0.775)), round((w * 0.862))
            y0, y1 = word.top + word.height, min(h, (word.top + (3 * word.height)))
            price_words = self.vision.find_words_ocr(frame, region=(x0, y0, (x1 - x0), (y1 - y0)), white_text=True, brightness_floor=200, roi_upscale=3, tesseract_config='--psm 7 -c tessedit_char_whitelist=0123456789')
            digits = ''.join((c for token in price_words for c in token.text if c.isdigit()))
            if (not digits or int(digits) != pick.cost):
                continue
            icon = frame[y0:y1, round((w * 0.862)):round((w * 0.885))]
            icon_hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
            gold = float(
                ((icon_hsv[:, :, 0] >= 15) & (icon_hsv[:, :, 0] <= 32)
                 & (icon_hsv[:, :, 1] > 90) & (icon_hsv[:, :, 2] > 140)).mean()
            )
            pink = VisionService.pink_fraction(icon)
            vivid = VisionService.pink_fraction(icon, val_floor=170)
            if vivid >= 0.08:
                resource = 'elixir'
            elif gold >= 0.08:
                resource = 'gold'
            elif pink >= 0.1:
                resource = 'dark_elixir'
            else:
                resource = None
            if resource != pick.resource:
                continue
            button = frame[max(0, (word.top - (word.height // 2))):y1, x0:round((w * 0.885))]
            hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
            green = (
                (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 85)
                & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 90)
            )
            if float(green.mean()) < 0.2:
                continue
            if VisionService.red_hue_fraction(button) >= _EXEC_RED_VETO_FRACTION:
                continue
            return ((word.left + (word.width // 2)), (word.top + (word.height // 2)))
        return None

    def _find_resource_confirm(self, frame, pick):
        """Exact price/currency on the green resource button; no hammer/gem fallback."""
        if pick.cost is None or pick.resource is None:
            return None
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 70, 90), (85, 255, 255))
        contours, _ = cv2.findContours(green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        borders, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = (list(contours) + list(borders))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (fw * 0.09) <= w <= (fw * 0.3) and (fh * 0.06) <= h <= (fh * 0.18):
                if not y >= (fh * (0.2 if pick.detail else 0.65)):
                    continue
            if (cv2.countNonZero(green[y:(y + h), x:(x + w)]) / (w * h)) < 0.2:
                continue
            point = find_ui_label(frame, self.vision, 'Confirm', region=((x + round((w * 0.15))), (y + 2), round((w * 0.7)), round((h * 0.4))), min_confidence=60)
            if point[0] is None:
                continue
            ry = (y + round((h * 0.43)))
            rx, rw = x + round((w * 0.05)), round((w * 0.68))
            rh = max(1, (((y + h) - ry) - 3))
            price = frame[ry:(ry + rh), rx:(rx + rw)]
            from app.core.upgrade_menu import _card_red_fraction, _classify_card_resource
            if _card_red_fraction(price) >= _EXEC_RED_VETO_FRACTION:
                continue
            reading = read_price_digits(price)
            if reading is not None:
                if reading.value != pick.cost:
                    continue
            else:
                words = self.vision.find_words_ocr(frame, region=(rx, ry, rw, rh), white_text=True, brightness_floor=200, roi_upscale=3, tesseract_config='--psm 7 -c tessedit_char_whitelist=0123456789,')
                digits = ''.join((c for word in sorted(words, key=(lambda w: w.left)) for c in word.text if c.isdigit()))
                if (not digits or int(digits) != pick.cost):
                    continue
            icon_box = ((x + round((w * 0.76))), ry, max(1, round((w * 0.23))), rh)
            if _classify_card_resource(frame, icon_box) != pick.resource:
                continue
            return point

    def _confirm_build_card_upgrade(self, pick):
        for _ in range(_EXEC_UPGRADE_WORD_POLLS):
            if self.stop_event.is_set():
                return False
            frame = self._frame()
            if frame is None:
                return False
            if self._selection_matches_pick(frame, pick) is True:
                point = self._find_resource_confirm(frame, pick)
                if point is not None:
                    self.input.click(*point, pause=0.6, rand=False)
                    return True
            if not self.stop_event.wait(0.4):
                continue
            return False
        logger.warning('Upgrade exec: direct confirmation did not match name, cost and currency for %r', pick.label)
        _dump_debug_frame(frame, 'upgexec_confirmfail')
        return False

    def _walk_upgrade_chain(self, pick, free_before=None):
        """After the row click: hunt and click "Upgrade" up to twice (bottom bar →
    confirm dialog). True when at least one Upgrade was clicked and the chain went
    quiet (no further Upgrade word — the confirm dialog closed itself)."""
        if getattr(self, '_using_build_menu', False):
            return self._confirm_build_card_upgrade(pick)
        clicks = 0
        quiet_polls = 0
        dumped = False
        name_ok = False
        name_strikes = 0
        for _ in range((_EXEC_UPGRADE_WORD_POLLS + (_EXEC_MAX_UPGRADE_CLICKS * 4))):
            if self.stop_event.is_set():
                return False
            frame = self._frame()
            if frame is None:
                return False
            if not dumped:
                _dump_debug_frame(frame, 'upgexec_row')
                dumped = True
            if clicks == 0 and not name_ok:
                verdict = self._selection_matches_pick(frame, pick)
                if verdict is not True:
                    name_strikes += 1
                    if name_strikes >= 3:
                        logger.warning('Upgrade exec: screen does not name %r — wrong selection, aborting without buying', pick.label)
                        _dump_debug_frame(frame, 'upgexec_wrongsel')
                        return False
                    if self.stop_event.wait(0.6):
                        return False
                else:
                    name_ok = True
                    pt = self._find_crafting_confirm(frame, pick)
                    if pt is None:
                        pt = self._find_upgrade_word(frame)
                    if pt is None and clicks >= 1:
                        pt = self._find_green_confirm_blob(frame)
                        if pt is not None:
                            logger.info('Upgrade exec: confirm found via green-blob fallback at %s', pt)
                    if pt is None:
                        if clicks == 0:
                            if self.stop_event.wait(0.5):
                                return False
                        else:
                            quiet_polls += 1
                            if quiet_polls >= 3:
                                return True
                            if self.stop_event.wait(0.5):
                                return False
                            continue
                            if clicks >= _EXEC_MAX_UPGRADE_CLICKS:
                                chip = parse_builder_chip(frame)
                                if chip is not None and free_before is not None and chip[0] < free_before:
                                    logger.info('Upgrade exec: builder count dropped after the click chain; upgrade started')
                                    return True
                                logger.warning('Upgrade exec: Upgrade word still visible after %d clicks — aborting', clicks)
                                _dump_debug_frame(frame, 'upgexec_fail')
                                return False
                            logger.info('Upgrade exec: clicking Upgrade/Confirm at %s (click %d)', pt, (clicks + 1))
                            self.input.click(*pt, pause=0.7)
                            clicks += 1
                            quiet_polls = 0
                            if not self.stop_event.wait(1.2):
                                continue
                            return False
        if clicks == 0:
            frame = self._frame()
            if frame is not None:
                _dump_debug_frame(frame, 'upgexec_fail')
            logger.info('Upgrade exec: no Upgrade button appeared after the row click')
            return False
        return True

    def _verify_builder_drop(self, free_before):
        """Poll the builder chip until it reads below ``free_before``."""
        for _ in range(_EXEC_VERIFY_POLLS):
            if self.stop_event.wait(0.6):
                return False
            frame = self._frame()
            if frame is None:
                continue
            chip = parse_builder_chip(frame)
            if chip is None:
                continue
            if chip[0] < free_before:
                return True
            logger.info('Upgrade exec verify: builder chip reads %s (was %d free)', chip, free_before)
        return False

