import subprocess
import shutil
from typing import Callable, Optional
from utils.logger import system_logger
from commands.command_types import SystemMode

class SystemActionHandler:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.amixer_path = shutil.which("amixer")
        self.pactl_path = shutil.which("pactl")
        self.is_paused = False

    def change_volume(self, delta_percent: int) -> str:
        """Adjusts system master volume (e.g. +5% or -5%)."""
        sign = "+" if delta_percent > 0 else "-"
        abs_val = abs(delta_percent)
        desc = f"VOLUME_{'UP' if delta_percent > 0 else 'DOWN'} ({sign}{abs_val}%)"

        if self.dry_run:
            system_logger.info(f"[System Action] [SIMULATED] {desc}")
            return desc

        # Try amixer first (Linux standard ALSA)
        if self.amixer_path:
            try:
                subprocess.run(
                    [self.amixer_path, "-q", "set", "Master", f"{abs_val}%{sign}"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                system_logger.info(f"[System Action] Volume changed via amixer: {desc}")
                return desc
            except Exception as e:
                system_logger.warning(f"amixer execution failed: {e}")

        # Try pactl next (PulseAudio / PipeWire)
        if self.pactl_path:
            try:
                subprocess.run(
                    [self.pactl_path, "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{abs_val}%"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                system_logger.info(f"[System Action] Volume changed via pactl: {desc}")
                return desc
            except Exception as e:
                system_logger.warning(f"pactl execution failed: {e}")

        system_logger.info(f"[System Action] [SIMULATED] Volume altered: {desc}")
        return desc

    def toggle_pause(self) -> str:
        self.is_paused = not self.is_paused
        status = "SYSTEM_PAUSED" if self.is_paused else "SYSTEM_RESUMED"
        system_logger.info(f"[System Action] {status}")
        return status

    def cycle_mode(self, current_mode: SystemMode) -> SystemMode:
        """Cycles GESTURE -> VOICE -> HYBRID -> GESTURE."""
        if current_mode == SystemMode.GESTURE:
            return SystemMode.VOICE
        elif current_mode == SystemMode.VOICE:
            return SystemMode.HYBRID
        else:
            return SystemMode.GESTURE
