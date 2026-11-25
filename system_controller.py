import os
import sys
import time
import cv2
import numpy as np
from typing import Optional

# Setup import path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from commands.command_types import SystemMode, CommandType, CommandEvent
from commands.command_mapping import CommandMapping
from actions.action_executor import ActionExecutor
from gesture.detector import HandDetector
from gesture.classifier import GestureClassifier
from gesture.smoother import GestureSmoother
from voice.recognizer import VoiceRecognizer
from ui.dashboard import DashboardHUD
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

class UnifiedHCIController:
    """
    Unified controller coordinating Camera Gesture Recognition, Speech Recognition,
    Command Mapping, Action Execution, and Dashboard HUD rendering.
    """
    def __init__(self, config_path: Optional[str] = None, mode: Optional[SystemMode] = None, dry_run: bool = False):
        self.cfg_mgr = ConfigManager(config_path) if config_path else config_manager
        if dry_run:
            self.cfg_mgr.set("system", "dry_run", True)

        # Mode setup
        cfg_mode_str = self.cfg_mgr.get("system", "mode", default="HYBRID")
        self.current_mode = mode or SystemMode.from_string(cfg_mode_str)

        # 1. Command and Action Layer (Shared)
        self.command_mapping = CommandMapping(self.cfg_mgr)
        self.action_executor = ActionExecutor(self.cfg_mgr, on_mode_change=self.set_mode)

        # 2. Gesture Subsystem
        self.detector = HandDetector()
        self.classifier = GestureClassifier()
        self.smoother = GestureSmoother(self.command_mapping, self.cfg_mgr)

        # 3. Voice Subsystem
        self.voice_recognizer = VoiceRecognizer(self.command_mapping, self.cfg_mgr)

        # 4. User Interface HUD
        self.hud = DashboardHUD()

        self.running: bool = False
        self.cap: Optional[cv2.VideoCapture] = None

    def set_mode(self, new_mode: SystemMode):
        self.current_mode = new_mode
        self.cfg_mgr.set("system", "mode", new_mode.value)
        system_logger.info(f"[MODE CHANGED] New mode: {new_mode.value}")

    def cycle_mode(self):
        new_mode = self.action_executor.system.cycle_mode(self.current_mode)
        self.set_mode(new_mode)

    def toggle_pause(self):
        cmd_event = CommandEvent(
            command=CommandType.PAUSE,
            source="shortcut",
            raw_input="Key: P"
        )
        self.action_executor.execute(cmd_event, self.current_mode)

    def start_voice(self):
        self.voice_recognizer.start()

    def stop_voice(self):
        self.voice_recognizer.stop()

    def process_single_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Processes one video frame and returns the annotated frame with HUD.
        Also pops and processes any pending voice commands.
        Used for both live camera feed and unit/integration testing.
        """
        # 1. Process Voice Queue first
        while True:
            voice_cmd = self.voice_recognizer.get_pending_command()
            if not voice_cmd:
                break
            exec_res = self.action_executor.execute(voice_cmd, self.current_mode)
            if exec_res.success:
                self.voice_recognizer.last_status_message = f"EXECUTED: {exec_res.action_name}"
            else:
                self.voice_recognizer.last_status_message = f"BLOCKED: {exec_res.message}"

        # If frame is None, return None immediately
        if frame_bgr is None:
            return None

        # 2. Process Hand Detection
        detection = self.detector.process(frame_bgr)
        classification = None
        annotated_frame = detection.annotated_frame if detection.annotated_frame is not None else frame_bgr.copy()

        if detection.hand_detected and detection.landmarks:
            classification = self.classifier.classify(detection.landmarks)

            # Draw gesture badge on hand bounding box
            if classification and detection.bbox:
                x1, y1, x2, y2 = detection.bbox
                box_color = (0, 240, 100) if classification.gesture_id != 0 else (60, 60, 240)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                badge_text = classification.badge
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                badge_y = max(th + 10, y1 - 10)
                cv2.rectangle(annotated_frame, (x1, badge_y - th - 6), (x1 + tw + 12, badge_y + 4), box_color, -1)
                cv2.putText(annotated_frame, badge_text, (x1 + 6, badge_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # 3. Process Gesture Smoothing & Debouncing
        cmd_event = self.smoother.update(classification, detection.hand_detected)
        if cmd_event is not None:
            self.action_executor.execute(cmd_event, self.current_mode)

        return annotated_frame

    def run(self, camera_index: Optional[int] = None, window_name: str = "Unified Hand + Voice HCI", fullscreen: bool = False):
        """
        Main application loop.
        Captures camera video, processes audio in background, renders HUD, and handles key inputs.
        """
        if camera_index is None:
            camera_index = self.cfg_mgr.get("hardware", "camera_index", default=0)

        system_logger.info(f"Starting Unified HCI System in {self.current_mode.value} mode...")
        self.start_voice()

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            system_logger.error(f"Error: Could not open camera {camera_index}. Running in headless/simulation loop.")
            self._run_fallback_loop()
            return

        # Request high-definition resolution for maximum hand visibility
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        system_logger.info(f"Camera opened with resolution: {actual_w}x{actual_h}")

        # Create clean, borderless/uncluttered window without Qt helper toolbar
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
        # Enlarge default window to 1600x900 or maximize
        cv2.resizeWindow(window_name, 1600, 900)
        is_fullscreen = fullscreen or self.cfg_mgr.get("hardware", "fullscreen", default=False)
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        self.running = True
        prev_time = time.time()
        fps = 30.0

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    system_logger.warning("Camera frame read failed. Reconnecting camera...")
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(camera_index)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Process frame and inputs
                frame_hud = self.process_single_frame(frame)

                # Get latest classification for HUD display
                classification = None
                if self.smoother.current_stable_gid in self.classifier.GESTURE_INFO if hasattr(self.classifier, 'GESTURE_INFO') else True:
                    from gestures import GESTURE_INFO
                    if self.smoother.current_stable_gid in GESTURE_INFO:
                        ginfo = GESTURE_INFO[self.smoother.current_stable_gid]
                        from gesture.classifier import GestureClassification
                        classification = GestureClassification(
                            gesture_id=self.smoother.current_stable_gid,
                            name=ginfo['name'],
                            badge=ginfo['badge'],
                            desc=ginfo['desc'],
                            confidence=0.90,
                            finger_status=self.smoother.current_finger_status
                        )

                # Draw complete HUD overlay
                frame_hud = self.hud.draw(
                    frame=frame_hud,
                    mode=self.current_mode,
                    is_paused=self.action_executor.is_paused,
                    classification=classification,
                    stability_progress=self.smoother.stability_progress,
                    is_waiting_for_release=self.smoother.is_waiting_for_release,
                    last_command=self.action_executor.last_command,
                    last_action=self.action_executor.last_action_name,
                    last_source=self.action_executor.last_source,
                    action_status=self.action_executor.last_status,
                    cooldown_remaining=self.action_executor.cooldown_remaining(),
                    max_cooldown=self.action_executor.active_cooldown,
                    mic_status=self.voice_recognizer.mic_status,
                    is_voice_listening=self.voice_recognizer.is_listening,
                    last_voice_transcript=self.voice_recognizer.last_transcript,
                    fps=fps,
                    voice_status=self.voice_recognizer.last_status_message
                )

                cv2.imshow(window_name, frame_hud)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('h'), ord('q'), 27):  # 'h', 'q', ESC to exit
                    break
                elif key == ord('f'):
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        system_logger.info("Fullscreen enabled.")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        system_logger.info("Fullscreen disabled (windowed mode).")
                elif key == ord('m'):
                    self.cycle_mode()
                elif key == ord('p'):
                    self.toggle_pause()
                elif key == ord('t'):
                    # Interactive test voice phrase injection
                    test_phrases = ["confirm", "cancel", "next", "previous", "click", "scroll down", "volume up"]
                    import random
                    choice = random.choice(test_phrases)
                    system_logger.info(f"[Test Trigger] Injecting test voice phrase: '{choice}'")
                    self.voice_recognizer.inject_phrase(choice)

        except KeyboardInterrupt:
            system_logger.info("Application interrupted by user.")
        finally:
            self.shutdown()

    def _run_fallback_loop(self):
        """Fallback loop for systems without camera."""
        self.running = True
        system_logger.info("Running in fallback loop (press Ctrl+C to exit).")
        try:
            while self.running:
                while True:
                    voice_cmd = self.voice_recognizer.get_pending_command()
                    if not voice_cmd:
                        break
                    self.action_executor.execute(voice_cmd, self.current_mode)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Cleans up all resources safely."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        self.stop_voice()
        cv2.destroyAllWindows()
        system_logger.info("Unified HCI System shutdown cleanly.")
