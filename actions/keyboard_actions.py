import os
import sys
from utils.logger import system_logger

# Attempt to initialize keyboard driver
_pynput_keyboard = None
_pyautogui_available = False

try:
    from pynput.keyboard import Controller as PynputController, Key as PynputKey
    _pynput_keyboard = PynputController()
except Exception:
    _pynput_keyboard = None

try:
    import pyautogui
    # Test if display is connected
    _ = pyautogui.size()
    _pyautogui_available = True
except Exception:
    _pyautogui_available = False

class KeyboardActionHandler:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.is_real_device_ready = (not dry_run) and (_pynput_keyboard is not None or _pyautogui_available)

    def press_key(self, key_name: str) -> bool:
        """
        Presses and releases a single key (e.g. 'enter', 'escape', 'left', 'right').
        Returns True if triggered (either real or simulated).
        """
        if self.dry_run or not self.is_real_device_ready:
            system_logger.info(f"[Keyboard Action] [SIMULATED] Key pressed: {key_name}")
            return True

        try:
            if _pyautogui_available:
                import pyautogui
                pyautogui.press(key_name)
                return True
            elif _pynput_keyboard is not None:
                from pynput.keyboard import Key
                key_map = {
                    "enter": Key.enter,
                    "escape": Key.esc,
                    "left": Key.left,
                    "right": Key.right,
                    "up": Key.up,
                    "down": Key.down,
                    "backspace": Key.backspace,
                    "page_down": Key.page_down,
                    "page_up": Key.page_up
                }
                k = key_map.get(key_name.lower(), key_name)
                _pynput_keyboard.press(k)
                _pynput_keyboard.release(k)
                return True
        except Exception as e:
            system_logger.warning(f"[Keyboard Action] Real key press failed ({e}), falling back to simulated.")
            return True
        return False

    def press_hotkey(self, *keys: str) -> bool:
        """Presses and releases a hotkey combination (e.g. ('alt', 'left') for back)."""
        if self.dry_run or not self.is_real_device_ready:
            system_logger.info(f"[Keyboard Action] [SIMULATED] Hotkey pressed: {' + '.join(keys)}")
            return True

        try:
            if _pyautogui_available:
                import pyautogui
                pyautogui.hotkey(*keys)
                return True
            elif _pynput_keyboard is not None:
                from pynput.keyboard import Key
                key_map = {"alt": Key.alt, "left": Key.left, "ctrl": Key.ctrl}
                mapped_keys = [key_map.get(k.lower(), k) for k in keys]
                for k in mapped_keys:
                    _pynput_keyboard.press(k)
                for k in reversed(mapped_keys):
                    _pynput_keyboard.release(k)
                return True
        except Exception as e:
            system_logger.warning(f"[Keyboard Action] Hotkey failed ({e}), falling back to simulated.")
            return True
        return False

    def confirm(self) -> str:
        self.press_key("enter")
        return "ENTER"

    def cancel(self) -> str:
        self.press_key("escape")
        return "ESCAPE"

    def next(self) -> str:
        self.press_key("right")
        return "RIGHT_ARROW"

    def previous(self) -> str:
        self.press_key("left")
        return "LEFT_ARROW"

    def back(self) -> str:
        self.press_hotkey("alt", "left")
        return "ALT+LEFT (BACK)"
