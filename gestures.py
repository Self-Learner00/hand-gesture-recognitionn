"""Top-level re-export for gestures definition to support both package and script execution."""
import os
import sys

# Ensure Hand_Gesture is on path
_hand_gesture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hand_Gesture")
if _hand_gesture_dir not in sys.path:
    sys.path.insert(0, _hand_gesture_dir)

from Hand_Gesture.gestures import GESTURE_INFO, GESTURE_PROMPTS

__all__ = ["GESTURE_INFO", "GESTURE_PROMPTS"]
