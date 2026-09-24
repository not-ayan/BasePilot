'''Bot orchestration for the Qt UI.'''
from __future__ import annotations
import threading
from PySide6.QtCore import QObject, Signal
from app.core.bot import Bot
from app.utils.logger import setup_logger
from app.utils.profile_settings_store import load_profile_settings

logger = setup_logger('BotController')

class BotController(QObject):
    statusChanged = Signal(str, bool)
    botStarted = Signal()
    botFinished = Signal(object)
    runningChanged = Signal(bool)
    # Live village state from the bot thread: dict with any of
    # state/builders/lab/storages/note (missing keys = unchanged/unknown).
    stateChanged = Signal(dict)
    # Session loot totals: gold, elixir, dark elixir, elapsed seconds.
    lootChanged = Signal(int, int, int, float)

    def __init__(self, bot_version):
        super().__init__()
        self._bot_version = bot_version
        self._bot = Bot()
        self._bot_thread = None

    def is_running(self):
        return self._bot_thread is not None and self._bot_thread.is_alive()

    def start(self, *, method, minutes, star_bonus, ranked_fill, upgrade_walls, multi_run_players, builder_base = False, loot_prioritise = 'both', auto_upgrade = 'off', secondary_troop_template = '', secondary_troop_count = 12):
        '''`minutes <= 0` = unlimited ("run until maxed") - single Home Village runs only.'''
        if self.is_running():
            return None

        def worker():
            error_msg = None

            def on_status(msg):
                self.statusChanged.emit(msg, 'not found' in msg.lower())

            def on_state(payload):
                self.stateChanged.emit(dict(payload))

            def on_loot(gold, elixir, dark, elapsed):
                self.lootChanged.emit(int(gold), int(elixir), int(dark), float(elapsed))

            try:
                profile = load_profile_settings()
                self._bot.start(
                    method,
                    minutes,
                    star_bonus = star_bonus,
                    status_callback = on_status,
                    loot_callback = on_loot,
                    state_callback = on_state,
                    multi_run_players = multi_run_players,
                    ranked_fill = ranked_fill,
                    upgrade_walls = upgrade_walls,
                    earthquake_method = profile.earthquake_method,
                    builder_base = builder_base,
                    loot_prioritise = loot_prioritise,
                    wall_upgrade_threshold = profile.wall_upgrade_threshold_m * 1000000,
                    auto_upgrade = auto_upgrade,
                    reserve_builders = profile.reserve_builders,
                    upgrade_order = profile.upgrade_order,
                    min_gold = profile.min_gold_k * 1000,
                    min_elixir = profile.min_elixir_k * 1000,
                    min_dark_elixir = (profile.min_dark_elixir if getattr(profile, 'filter_dark_elixir', False) else 0),
                    max_next_skips = profile.max_next_skips,
                    battle_end_delay = profile.battle_end_delay_s,
                    loot_match_condition = getattr(profile, 'loot_match_condition', 'any'),
                    secondary_troop_template = secondary_troop_template,
                    secondary_troop_count = secondary_troop_count
                )
            except InterruptedError:
                logger.info('Bot thread stopped by user')
            except Exception as exc:
                error_msg = str(exc)
                logger.exception('Bot thread failed')
            self.botFinished.emit(error_msg)

        self._bot_thread = threading.Thread(target = worker, daemon = True, name = 'BotThread')
        self.runningChanged.emit(True)
        self.botStarted.emit()
        self._bot_thread.start()

    def stop(self):
        self._bot.stop()
        self.statusChanged.emit('Stopping...', False)

    def on_bot_finished(self):
        self.runningChanged.emit(False)
