import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Auto re-exec using local venv if system python is missing dependencies
venv_dir = os.path.join(root_dir, "venv")
venv_python = os.path.join(venv_dir, "bin", "python")
if os.path.exists(venv_python) and sys.prefix != venv_dir:
    try:
        import numpy
        import cv2
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import unittest
import time
import numpy as np

from commands.command_types import SystemMode, CommandType, CommandEvent
from commands.command_mapping import CommandMapping
from actions.action_executor import ActionExecutor
from gesture.classifier import GestureClassifier, GestureClassification
from gesture.smoother import GestureSmoother
from voice.command_parser import VoiceCommandParser
from voice.recognizer import VoiceRecognizer
from system_controller import UnifiedHCIController
from utils.config import ConfigManager
from gestures import GESTURE_INFO

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_mock_thumbs_up_landmarks():
    """Generates synthetic landmarks representing Thumbs Up (ID 1)."""
    # 21 points: wrist (0), thumb (1-4), index (5-8), middle (9-12), ring (13-16), pinky (17-20)
    # Wrist at (0.5, 0.8)
    # Fingers curled: tips closer to wrist than PIP
    # Thumb pointing straight up: tip.y < pip.y
    lms = [MockLandmark(0.5, 0.8) for _ in range(21)]
    # Middle MCP (9) to set hand_size
    lms[9] = MockLandmark(0.5, 0.5)  # hand_size = 0.3
    # Thumb: 1, 2, 3, 4
    lms[1] = MockLandmark(0.48, 0.7)
    lms[2] = MockLandmark(0.46, 0.6)
    lms[3] = MockLandmark(0.44, 0.45)
    lms[4] = MockLandmark(0.44, 0.30) # Thumb tip way up
    # Pinky MCP (17)
    lms[17] = MockLandmark(0.58, 0.55)
    # Curled fingers (Index, Middle, Ring, Pinky tips curled below PIP/MCP)
    for base in [5, 9, 13, 17]:
        lms[base] = MockLandmark(lms[base].x, 0.5)
        lms[base+1] = MockLandmark(lms[base].x, 0.45)
        lms[base+2] = MockLandmark(lms[base].x, 0.55)
        lms[base+3] = MockLandmark(lms[base].x, 0.60) # tip curled down
    return lms

class UnifiedSystemTests(unittest.TestCase):
    def setUp(self):
        # Create isolated config manager in dry_run mode
        self.cfg = ConfigManager()
        self.cfg.set("system", "dry_run", True)
        self.cfg.set("gesture", "min_stable_frames", 4)
        self.cfg.set("gesture", "cooldown_seconds", 0.5)
        self.cfg.set("gesture", "confidence_threshold", 0.60)
        self.cfg.set("voice", "cooldown_seconds", 0.5)

        self.mapping = CommandMapping(self.cfg)
        self.executor = ActionExecutor(self.cfg)
        self.classifier = GestureClassifier()
        self.smoother = GestureSmoother(self.mapping, self.cfg)

    # 1. Stable Gesture Detection
    def test_01_stable_gesture_detection(self):
        """Test that a gesture requires N consecutive frames before triggering."""
        classification = GestureClassification(
            gesture_id=1, name="Yes", badge="Yes [Thumbs Up]", desc="Thumbs Up",
            confidence=0.95, finger_status=[True, False, False, False, False]
        )

        # First 3 frames should not trigger (min_stable_frames = 4)
        for i in range(3):
            event = self.smoother.update(classification, hand_detected=True)
            self.assertIsNone(event, f"Triggered prematurely on frame {i+1}")

        # 4th consecutive frame should trigger
        event = self.smoother.update(classification, hand_detected=True)
        self.assertIsNotNone(event)
        self.assertEqual(event.command, CommandType.CONFIRM)

    # 2. Repeated Gesture Prevention
    def test_02_repeated_gesture_prevention(self):
        """Test that keeping the same gesture visible across multiple frames doesn't re-trigger."""
        classification = GestureClassification(
            gesture_id=1, name="Yes", badge="Yes [Thumbs Up]", desc="Thumbs Up",
            confidence=0.95, finger_status=[True, False, False, False, False]
        )

        # Trigger initial action
        for _ in range(4):
            self.smoother.update(classification, hand_detected=True)

        # Subsequent frames holding the exact same gesture must NOT trigger
        for f in range(10):
            event = self.smoother.update(classification, hand_detected=True)
            self.assertIsNone(event, f"Incorrectly re-triggered while holding gesture on frame {f+1}")

        # Releasing hand and showing again SHOULD allow trigger
        for _ in range(self.smoother.hand_loss_tolerance + 2):
            self.smoother.update(None, hand_detected=False)

        # Now re-showing thumbs up triggers after 4 frames
        event = None
        for _ in range(4):
            event = self.smoother.update(classification, hand_detected=True)
        self.assertIsNotNone(event)

    # 3. Accidental Gesture Prevention
    def test_03_accidental_gesture_prevention(self):
        """Test that brief noise or fast flickering (< min_stable_frames) does not trigger."""
        # Frame 1: Peace (ID 7)
        c1 = GestureClassification(7, "Peace", "Peace", "V Sign", 0.9, [False, True, True, False, False])
        self.assertIsNone(self.smoother.update(c1, hand_detected=True))

        # Frame 2: Point (ID 9)
        c2 = GestureClassification(9, "Point", "Point", "Index", 0.9, [False, True, False, False, False])
        self.assertIsNone(self.smoother.update(c2, hand_detected=True))

        # Frame 3: Fist (ID 0)
        c3 = GestureClassification(0, "No", "No", "Fist", 0.9, [False, False, False, False, False])
        self.assertIsNone(self.smoother.update(c3, hand_detected=True))

        # None of them reached 4 consecutive frames, no command should have fired
        self.assertEqual(self.executor.last_command, CommandType.NONE)

    # 4. Gesture -> Command -> Action
    def test_04_gesture_to_command_to_action(self):
        """Test full pipeline: 👍 Thumbs Up -> CONFIRM -> ENTER."""
        classification = GestureClassification(
            gesture_id=1, name="Yes", badge="Yes [Thumbs Up]", desc="Thumbs Up",
            confidence=0.95, finger_status=[True, False, False, False, False]
        )
        for _ in range(3):
            self.smoother.update(classification, hand_detected=True)
        event = self.smoother.update(classification, hand_detected=True)
        self.assertIsNotNone(event)

        result = self.executor.execute(event, SystemMode.HYBRID)
        self.assertTrue(result.success)
        self.assertEqual(result.command, CommandType.CONFIRM)
        self.assertEqual(result.action_name, "ENTER")
        self.assertEqual(self.executor.last_action_name, "ENTER")

    # 5. Voice -> Command -> Action
    def test_05_voice_to_command_to_action(self):
        """Test voice pipeline: 'confirm' -> CONFIRM -> ENTER."""
        parser = VoiceCommandParser(self.mapping, self.cfg)
        cmd_event = parser.parse("please confirm")
        self.assertIsNotNone(cmd_event)
        self.assertEqual(cmd_event.command, CommandType.CONFIRM)

        result = self.executor.execute(cmd_event, SystemMode.HYBRID)
        self.assertTrue(result.success)
        self.assertEqual(result.command, CommandType.CONFIRM)
        self.assertEqual(result.action_name, "ENTER")

    # 6. Gesture and Voice Triggering the Same Underlying Action
    def test_06_gesture_and_voice_same_action(self):
        """Both 👍 and voice 'confirm' reach the exact same ActionExecutor action handler."""
        # 1. Voice triggers CANCEL -> ESCAPE
        voice_parser = VoiceCommandParser(self.mapping, self.cfg)
        v_event = voice_parser.parse("cancel")
        res_v = self.executor.execute(v_event, SystemMode.HYBRID)
        self.assertEqual(res_v.action_name, "ESCAPE")

        time.sleep(0.55) # wait out cooldown

        # 2. Gesture Thumbs Down (ID 17) triggers CANCEL -> ESCAPE
        g_class = GestureClassification(17, "Thumbs Down", "Thumbs Down", "Down", 0.95, [True, False, False, False, False])
        for _ in range(4):
            g_event = self.smoother.update(g_class, hand_detected=True)
        self.assertIsNotNone(g_event)
        res_g = self.executor.execute(g_event, SystemMode.HYBRID)
        self.assertEqual(res_g.action_name, "ESCAPE")

    # 7. Switching Between Modes
    def test_07_switching_between_modes(self):
        """Test mode switching: GESTURE mode rejects voice, VOICE mode rejects gesture."""
        g_event = CommandEvent(command=CommandType.CONFIRM, source="gesture", raw_input="Thumbs Up")
        v_event = CommandEvent(command=CommandType.CONFIRM, source="voice", raw_input="confirm")

        # In GESTURE Mode
        res_g = self.executor.execute(g_event, SystemMode.GESTURE)
        self.assertTrue(res_g.success, "Gesture must execute in GESTURE mode")
        res_v = self.executor.execute(v_event, SystemMode.GESTURE)
        self.assertFalse(res_v.success, "Voice must be rejected in GESTURE mode")

        # In VOICE Mode
        time.sleep(0.55) # wait out cooldown from res_g
        res_g2 = self.executor.execute(g_event, SystemMode.VOICE)
        self.assertFalse(res_g2.success, "Gesture must be rejected in VOICE mode")
        res_v2 = self.executor.execute(v_event, SystemMode.VOICE)
        self.assertTrue(res_v2.success, "Voice must execute in VOICE mode")

    # 8. Hybrid Mode
    def test_08_hybrid_mode(self):
        """In HYBRID mode, both gesture and voice commands are accepted."""
        g_event = CommandEvent(command=CommandType.NEXT, source="gesture", raw_input="Peace")
        v_event = CommandEvent(command=CommandType.PREVIOUS, source="voice", raw_input="previous")

        res_g = self.executor.execute(g_event, SystemMode.HYBRID)
        self.assertTrue(res_g.success)
        self.assertEqual(res_g.action_name, "RIGHT_ARROW")

        time.sleep(0.55) # cooldown

        res_v = self.executor.execute(v_event, SystemMode.HYBRID)
        self.assertTrue(res_v.success)
        self.assertEqual(res_v.action_name, "LEFT_ARROW")

    # 9. Camera Disconnection Handling
    def test_09_camera_disconnection_handling(self):
        """Test that detector and smoother handle None or empty frames gracefully without crash."""
        controller = UnifiedHCIController(dry_run=True)
        # Process None frame
        annotated = controller.process_single_frame(None)
        self.assertIsNone(annotated)

        # Process black empty frame
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated2 = controller.process_single_frame(empty_frame)
        self.assertIsNotNone(annotated2)
        controller.shutdown()

    # 10. Microphone Disconnection Handling
    def test_10_microphone_disconnection_handling(self):
        """Test that voice recognizer handles hardware errors gracefully without crashing."""
        rec = VoiceRecognizer(self.mapping, self.cfg)
        # Start and stop recognizer
        rec.start()
        # Even if mic hardware is absent, mic_status reflects status cleanly
        self.assertIn(rec.mic_status, ["CONNECTED", "NO_MIC_HARDWARE", "SR_UNAVAILABLE", "INITIALIZING"] + [f"MIC_ERROR: {rec.mic_status}"])
        # Inject command manually still works
        event = rec.inject_phrase("click")
        self.assertIsNotNone(event)
        self.assertEqual(event.command, CommandType.CLICK)
        rec.stop()

    # 11. Low-Confidence Recognition
    def test_11_low_confidence_recognition(self):
        """Test that low-confidence gestures (< threshold) are filtered out."""
        low_conf_gesture = GestureClassification(
            gesture_id=1, name="Yes", badge="Yes", desc="Thumbs Up",
            confidence=0.40, finger_status=[True, False, False, False, False]
        )
        # Feed 10 consecutive frames
        for _ in range(10):
            event = self.smoother.update(low_conf_gesture, hand_detected=True)
            self.assertIsNone(event, "Low-confidence gesture should be discarded")

    # 12. Application Startup / Shutdown
    def test_12_application_startup_shutdown(self):
        """Test clean initialization and shutdown of UnifiedHCIController."""
        controller = UnifiedHCIController(dry_run=True)
        self.assertIsNotNone(controller.detector)
        self.assertIsNotNone(controller.classifier)
        self.assertIsNotNone(controller.action_executor)
        self.assertIsNotNone(controller.voice_recognizer)

        controller.start_voice()
        controller.shutdown()
        self.assertFalse(controller.voice_recognizer.running)

    # 13. Existing Gesture Functionality Integrity
    def test_13_existing_gesture_functionality(self):
        """Verify all 27 gestures remain intact and classified properly."""
        self.assertEqual(len(GESTURE_INFO), 27)
        for gid, info in GESTURE_INFO.items():
            self.assertIn("name", info)
            self.assertIn("badge", info)
            self.assertIn("desc", info)

        # Test geometric analysis on synthetic thumbs-up landmarks
        synthetic_thumbs_up = create_mock_thumbs_up_landmarks()
        gid, ginfo, finger_status, conf = self.classifier.analyze_geometry(synthetic_thumbs_up)
        self.assertEqual(gid, 1, f"Expected Thumbs Up (ID 1), got ID {gid} ({ginfo['name']})")
        self.assertTrue(finger_status[0], "Thumb should be extended")

    # 14. Half Heart / ILY Pose Recognition
    def test_14_half_heart_ily_gesture(self):
        """Verify that index horizontal + thumb down (half-heart pose) is classified as 25 (I Love You) instead of 17 (Thumbs Down)."""
        lms = [MockLandmark(0.5, 0.8) for _ in range(21)]
        lms[9] = MockLandmark(0.5, 0.5)  # hand_size = 0.3
        # Thumb pointing down
        lms[1] = MockLandmark(0.48, 0.55)
        lms[2] = MockLandmark(0.46, 0.60)
        lms[3] = MockLandmark(0.44, 0.68)
        lms[4] = MockLandmark(0.42, 0.75) # tip down
        # Index extended horizontally to form top half-heart arch
        lms[5] = MockLandmark(0.52, 0.50)
        lms[6] = MockLandmark(0.44, 0.49)
        lms[7] = MockLandmark(0.36, 0.49)
        lms[8] = MockLandmark(0.28, 0.50) # tip horizontal away from MCP 5
        # Middle, ring, pinky curled into palm
        for base in [9, 13, 17]:
            lms[base] = MockLandmark(lms[base].x, 0.50)
            lms[base+1] = MockLandmark(lms[base].x, 0.48)
            lms[base+2] = MockLandmark(lms[base].x, 0.54)
            lms[base+3] = MockLandmark(lms[base].x, 0.58)

        gid, ginfo, finger_status, conf = self.classifier.analyze_geometry(lms)
        self.assertEqual(gid, 25, f"Expected I Love You (Half Heart) (ID 25), got ID {gid} ({ginfo['name']})")
        self.assertEqual(ginfo['name'], "I Love You")

if __name__ == '__main__':
    unittest.main()
