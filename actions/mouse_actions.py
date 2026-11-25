import sys
from utils.logger import system_logger

_pynput_mouse = None
_pyautogui_available = False

try:
    from pynput.mouse import Controller as PynputMouseController, Button as PynputButton
    _pynput_mouse = PynputMouseController()
except Exception:
    _pynput_mouse = None

try:
    import pyautogui
    _ = pyautogui.size()
    _pyautogui_available = True
except Exception:
    _pyautogui_available = False

class MouseActionHandler:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.is_real_device_ready = (not dry_run) and (_pynput_mouse is not None or _pyautogui_available)

    def click(self, button: str = "left") -> str:
        """Performs a mouse click (left or right)."""
        if self.dry_run or not self.is_real_device_ready:
            system_logger.info(f"[Mouse Action] [SIMULATED] Click {button}")
            return f"{button.upper()}_CLICK"

        try:
            if _pyautogui_available:
                import pyautogui
                pyautogui.click(button=button)
                return f"{button.upper()}_CLICK"
            elif _pynput_mouse is not None:
                from pynput.mouse import Button
                btn = Button.right if button.lower() == "right" else Button.left
                _pynput_mouse.click(btn, 1)
                return f"{button.upper()}_CLICK"
        except Exception as e:
            system_logger.warning(f"[Mouse Action] Real click failed ({e}), falling back to simulated.")
            return f"{button.upper()}_CLICK"
        return f"{button.upper()}_CLICK"

    def scroll(self, clicks: int) -> str:
        """Scrolls vertically. Positive = up, negative = down."""
        direction = "UP" if clicks > 0 else "DOWN"
        if self.dry_run or not self.is_real_device_ready:
            system_logger.info(f"[Mouse Action] [SIMULATED] Scroll {direction} ({abs(clicks)} steps)")
            return f"SCROLL_{direction}"

        try:
            if _pyautogui_available:
                import pyautogui
                # In pyautogui, scroll units vary by OS
                pyautogui.scroll(clicks * 5)
                return f"SCROLL_{direction}"
            elif _pynput_mouse is not None:
                _pynput_mouse.scroll(0, clicks)
                return f"SCROLL_{direction}"
        except Exception as e:
            system_logger.warning(f"[Mouse Action] Scroll failed ({e}), falling back to simulated.")
            return f"SCROLL_{direction}"
        return f"SCROLL_{direction}"
