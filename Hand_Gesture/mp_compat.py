import os
import sys
import types

try:
    import sounddevice
except Exception:
    from unittest.mock import MagicMock
    sys.modules['sounddevice'] = MagicMock()

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')

class LandmarkWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list

class HandsResultWrapper:
    def __init__(self, hand_landmarks_list):
        if hand_landmarks_list:
            self.multi_hand_landmarks = [LandmarkWrapper(lm_list) for lm_list in hand_landmarks_list]
        else:
            self.multi_hand_landmarks = None

class HandsCompat:
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        model_path = _DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            alt_path = os.path.join(os.getcwd(), 'Hand_Gesture', 'hand_landmarker.task')
            if os.path.exists(alt_path):
                model_path = alt_path

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = self.detector.detect(mp_image)
        return HandsResultWrapper(res.hand_landmarks)

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

def draw_landmarks(image, hand_landmark, connections=None, *args, **kwargs):
    landmarks = hand_landmark.landmark if hasattr(hand_landmark, 'landmark') else hand_landmark
    vision.drawing_utils.draw_landmarks(image, landmarks, connections)

# Patch mediapipe.solutions if not present
if not hasattr(mp, 'solutions'):
    solutions = types.ModuleType('mediapipe.solutions')
    
    hands_mod = types.ModuleType('mediapipe.solutions.hands')
    hands_mod.Hands = HandsCompat
    hands_mod.HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
    
    drawing_utils_mod = types.ModuleType('mediapipe.solutions.drawing_utils')
    drawing_utils_mod.draw_landmarks = draw_landmarks
    drawing_utils_mod.DrawingSpec = vision.drawing_utils.DrawingSpec
    
    drawing_styles_mod = types.ModuleType('mediapipe.solutions.drawing_styles')
    
    solutions.hands = hands_mod
    solutions.drawing_utils = drawing_utils_mod
    solutions.drawing_styles = drawing_styles_mod
    
    mp.solutions = solutions
