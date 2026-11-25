import os
import sys
import math
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, NamedTuple

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
hand_gesture_dir = os.path.join(root_dir, "Hand_Gesture")
for p in (root_dir, hand_gesture_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from gestures import GESTURE_INFO
from utils.logger import system_logger

def euclidean_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class GestureClassification(NamedTuple):
    gesture_id: int
    name: str
    badge: str
    desc: str
    confidence: float
    finger_status: List[bool]

class GestureClassifier:
    """
    Classifies 27 static hand poses using geometric landmark analysis and optional trained ML model.
    Preserves full backwards compatibility with existing gesture definitions.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_supports_full_gesture_set = False

        possible_paths = [
            model_path,
            os.path.join(hand_gesture_dir, 'model.p'),
            os.path.join(root_dir, 'model.p'),
            os.path.join(root_dir, 'hand_gesture_samplae', 'model.p')
        ]
        chosen_path = next((p for p in possible_paths if p and os.path.exists(p)), None)

        if chosen_path:
            try:
                with open(chosen_path, 'rb') as f:
                    data_dict = pickle.load(f)
                    self.model = data_dict.get('model', None)
                if self.model is not None:
                    trained_ids = {int(label) for label in getattr(self.model, 'classes_', [])}
                    self.model_supports_full_gesture_set = set(GESTURE_INFO).issubset(trained_ids)
                    system_logger.info(f"Loaded ML model from {chosen_path} (classes: {len(trained_ids)})")
            except Exception as e:
                system_logger.warning(f"Notice: ML model could not be loaded ({e}), using geometric gesture analyzer.")

    def analyze_geometry(self, landmarks) -> Tuple[int, Dict[str, Any], List[bool], float]:
        """
        Classifies static hand pose using MediaPipe landmark geometry and returns
        (gesture_id, info, finger_status, confidence).
        """
        wrist = landmarks[0]
        hand_size = euclidean_dist(wrist, landmarks[9])
        if hand_size < 1e-4:
            hand_size = 1.0

        d = euclidean_dist

        # Check extension for 4 fingers (Index, Middle, Ring, Pinky)
        index_ext = d(wrist, landmarks[8]) > d(wrist, landmarks[6]) * 1.15 and d(wrist, landmarks[8]) > d(wrist, landmarks[5]) * 1.2
        middle_ext = d(wrist, landmarks[12]) > d(wrist, landmarks[10]) * 1.15 and d(wrist, landmarks[12]) > d(wrist, landmarks[9]) * 1.2
        ring_ext = d(wrist, landmarks[16]) > d(wrist, landmarks[14]) * 1.15 and d(wrist, landmarks[16]) > d(wrist, landmarks[13]) * 1.2
        pinky_ext = d(wrist, landmarks[20]) > d(wrist, landmarks[18]) * 1.15 and d(wrist, landmarks[20]) > d(wrist, landmarks[17]) * 1.2

        if wrist.y > landmarks[9].y:
            index_ext = index_ext or (landmarks[8].y < landmarks[6].y and landmarks[8].y < landmarks[5].y)
            middle_ext = middle_ext or (landmarks[12].y < landmarks[10].y and landmarks[12].y < landmarks[9].y)
            ring_ext = ring_ext or (landmarks[16].y < landmarks[14].y and landmarks[16].y < landmarks[13].y)
            pinky_ext = pinky_ext or (landmarks[20].y < landmarks[18].y and landmarks[20].y < landmarks[17].y)

        # Thumb extension
        thumb_dist_pinky = d(landmarks[4], landmarks[17])
        thumb_mcp_dist_pinky = d(landmarks[2], landmarks[17])
        thumb_dist_middle_mcp = d(landmarks[4], landmarks[9])
        thumb_up = landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y and landmarks[4].y < landmarks[5].y
        thumb_down = landmarks[4].y > landmarks[3].y and landmarks[4].y > landmarks[2].y
        thumb_ext = (thumb_dist_pinky > thumb_mcp_dist_pinky * 1.10 or thumb_up or thumb_down) and thumb_dist_middle_mcp > hand_size * 0.42

        thumb_side = (
            not thumb_up and not thumb_down and
            d(landmarks[4], landmarks[2]) > hand_size * 0.45 and
            d(landmarks[4], landmarks[5]) > hand_size * 0.60
        )

        thumb_index_dist = d(landmarks[4], landmarks[8]) / hand_size
        index_middle_dist = d(landmarks[8], landmarks[12]) / hand_size
        middle_ring_dist = d(landmarks[12], landmarks[16]) / hand_size
        ring_pinky_dist = d(landmarks[16], landmarks[20]) / hand_size

        finger_status = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        ext_count = sum((index_ext, middle_ext, ring_ext, pinky_ext))
        all_four_extended = ext_count == 4
        compact_fingers = max(index_middle_dist, middle_ring_dist, ring_pinky_dist) < 0.25
        fingers_crossed = (landmarks[5].x - landmarks[9].x) * (landmarks[8].x - landmarks[12].x) < 0

        # Special Half-Heart check: index pointing horizontally/down while thumb down
        index_horizontal = abs(landmarks[8].y - landmarks[5].y) < 0.15 * hand_size and abs(landmarks[8].x - landmarks[5].x) > 0.45 * hand_size
        if thumb_down and index_horizontal and not middle_ext and not ring_ext and not pinky_ext:
            return 25, GESTURE_INFO[25], finger_status, 0.88

        # 24: Pinched fingers / Italian "Che Vuoi"
        tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        tip_cx = sum(t.x for t in tips) / 5.0
        tip_cy = sum(t.y for t in tips) / 5.0
        max_tip_spread = max(math.hypot(t.x - tip_cx, t.y - tip_cy) for t in tips) / hand_size
        tips_pointing_up = tip_cy < landmarks[9].y
        dist_wrist_tip_centroid = math.hypot(tip_cx - wrist.x, tip_cy - wrist.y) / hand_size
        if max_tip_spread < 0.28 and tips_pointing_up and dist_wrist_tip_centroid > 0.95 and thumb_index_dist < 0.28:
            return 24, GESTURE_INFO[24], finger_status, 0.92

        # 6: OK vs 18: Pinch
        # Stricter check for OK: thumb and index tips must be very close
        if thumb_index_dist < 0.30 and middle_ext and ring_ext and pinky_ext:
            return 6, GESTURE_INFO[6], finger_status, 0.92

        # 18: Pinch: thumb and index tips close, AND thumb distance to index tip is clearly smaller than to middle
        if thumb_index_dist < 0.25 and not middle_ext and not ring_ext and not pinky_ext:
            # Verify thumb is genuinely touching/pinching index tip, not just folded in a fist
            if d(landmarks[4], landmarks[8]) < d(landmarks[4], landmarks[12]) * 0.85:
                return 18, GESTURE_INFO[18], finger_status, 0.90

        # 5: I Love You, 8: Rock On, 15: Call Me
        if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
            return 5, GESTURE_INFO[5], finger_status, 0.94
        if index_ext and pinky_ext and not thumb_ext and not middle_ext and not ring_ext:
            return 8, GESTURE_INFO[8], finger_status, 0.91
        if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return 15, GESTURE_INFO[15], finger_status, 0.93

        # Thumb gestures
        if ext_count == 0 and thumb_up:
            return 1, GESTURE_INFO[1], finger_status, 0.95
        if ext_count == 0 and thumb_down:
            return 17, GESTURE_INFO[17], finger_status, 0.95
        if ext_count == 0 and thumb_side:
            return 23, GESTURE_INFO[23], finger_status, 0.88

        # 21: Vulcan Salute
        if (all_four_extended and index_middle_dist < 0.30 and ring_pinky_dist < 0.30 and
                middle_ring_dist > 0.32 and middle_ring_dist > 1.25 * max(index_middle_dist, ring_pinky_dist)):
            return 21, GESTURE_INFO[21], finger_status, 0.90

        # 25: ASL C
        c_finger_curve = (
            0.75 < d(wrist, landmarks[8]) / hand_size < 1.30 and
            0.75 < d(wrist, landmarks[12]) / hand_size < 1.30 and
            compact_fingers and
            0.30 < thumb_index_dist < 0.85 and
            thumb_ext
        )
        if c_finger_curve:
            return 25, GESTURE_INFO[25], finger_status, 0.86

        # Open palm variants
        if all_four_extended and not thumb_ext and compact_fingers:
            return 2, GESTURE_INFO[2], finger_status, 0.88
        if all_four_extended and thumb_ext and compact_fingers:
            return 3, GESTURE_INFO[3], finger_status, 0.90
        if all_four_extended and thumb_ext:
            return 4, GESTURE_INFO[4], finger_status, 0.92
        if all_four_extended and not thumb_ext:
            return 14, GESTURE_INFO[14], finger_status, 0.87

        # Single-finger and counting
        if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            if d(landmarks[4], landmarks[12]) / hand_size < 0.28:
                return 26, GESTURE_INFO[26], finger_status, 0.87
            return 9, GESTURE_INFO[9], finger_status, 0.94
        if middle_ext and not index_ext and not ring_ext and not pinky_ext and not thumb_ext:
            return 10, GESTURE_INFO[10], finger_status, 0.93
        if pinky_ext and not index_ext and not middle_ext and not ring_ext and not thumb_ext:
            return 11, GESTURE_INFO[11], finger_status, 0.91
        if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
            return 13, GESTURE_INFO[13], finger_status, 0.89

        # Crossed fingers & 3
        if thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
            if fingers_crossed:
                return 12, GESTURE_INFO[12], finger_status, 0.88
            return 19, GESTURE_INFO[19], finger_status, 0.88
        if thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext:
            return 20, GESTURE_INFO[20], finger_status, 0.87

        # Two fingers
        if index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            if index_middle_dist < 0.22:
                return 22, GESTURE_INFO[22], finger_status, 0.88
            return 7, GESTURE_INFO[7], finger_status, 0.93
        if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return 16, GESTURE_INFO[16], finger_status, 0.90

        # Closed fist
        if ext_count == 0:
            return 0, GESTURE_INFO[0], finger_status, 0.92

        return 0, GESTURE_INFO[0], finger_status, 0.50

    def classify(self, landmarks) -> Optional[GestureClassification]:
        """Classifies landmarks, blending geometric rule evaluation with ML model if available."""
        if landmarks is None or len(landmarks) < 21:
            return None

        gid, ginfo, finger_status, conf = self.analyze_geometry(landmarks)

        # Check ML model if available
        # Note: only override if the ML model is fully trained for all classes or has very high confidence (>0.90)
        # to prevent older/partial 9-class models from overriding clear geometric poses.
        if self.model is not None and self.model_supports_full_gesture_set:
            try:
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                min_x = min(x_coords)
                min_y = min(y_coords)
                features = []
                for lm in landmarks:
                    features.append(lm.x - min_x)
                    features.append(lm.y - min_y)

                if len(features) == 42:
                    if hasattr(self.model, "predict_proba"):
                        probs = self.model.predict_proba([features])[0]
                        best_idx = np.argmax(probs)
                        ml_pred = int(self.model.classes_[best_idx])
                        ml_conf = float(probs[best_idx])
                    else:
                        ml_pred = int(self.model.predict([features])[0])
                        ml_conf = 0.85

                    if ml_pred in GESTURE_INFO and ml_conf > 0.85:
                        gid = ml_pred
                        ginfo = GESTURE_INFO[gid]
                        conf = max(conf, ml_conf)
            except Exception as e:
                system_logger.debug(f"ML classification failed: {e}")

        return GestureClassification(
            gesture_id=gid,
            name=ginfo['name'],
            badge=ginfo['badge'],
            desc=ginfo['desc'],
            confidence=conf,
            finger_status=finger_status
        )
