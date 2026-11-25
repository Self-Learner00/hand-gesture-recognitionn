import os
import sys
import cv2
import numpy as np
from typing import Optional, Tuple, List, NamedTuple, Any

# Setup path to import mp_compat
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
hand_gesture_dir = os.path.join(root_dir, "Hand_Gesture")
for p in (root_dir, hand_gesture_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import mp_compat
except Exception:
    pass

import mediapipe as mp
from utils.logger import system_logger

class HandDetection(NamedTuple):
    landmarks: Optional[List[Any]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    annotated_frame: Optional[np.ndarray] = None
    hand_detected: bool = False
    hand_count: int = 0

class HandDetector:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_bgr: np.ndarray, flip_horizontal: bool = True, padding: int = 25) -> HandDetection:
        """
        Processes a BGR video frame and returns hand landmarks and bounding box.
        Gracefully handles None or invalid frames without crashing.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return HandDetection(hand_detected=False, hand_count=0)

        frame = cv2.flip(frame_bgr, 1) if flip_horizontal else frame_bgr.copy()
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.hands.process(frame_rgb)
        except Exception as e:
            system_logger.warning(f"MediaPipe detection error: {e}")
            return HandDetection(annotated_frame=frame, hand_detected=False, hand_count=0)

        if not results or not results.multi_hand_landmarks:
            return HandDetection(annotated_frame=frame, hand_detected=False, hand_count=0)

        hand_count = len(results.multi_hand_landmarks)
        # Select the primary hand (first hand)
        primary_hand = results.multi_hand_landmarks[0]

        # Draw landmarks on frame
        for hand_lm in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

        # Compute bounding box
        x_coords = [lm.x for lm in primary_hand.landmark]
        y_coords = [lm.y for lm in primary_hand.landmark]
        x1 = max(0, int(min(x_coords) * w) - padding)
        y1 = max(0, int(min(y_coords) * h) - padding)
        x2 = min(w, int(max(x_coords) * w) + padding)
        y2 = min(h, int(max(y_coords) * h) + padding)

        return HandDetection(
            landmarks=primary_hand.landmark,
            bbox=(x1, y1, x2, y2),
            annotated_frame=frame,
            hand_detected=True,
            hand_count=hand_count
        )

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
