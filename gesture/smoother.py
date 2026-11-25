from collections import deque
from typing import Optional, Dict, Any, List
import time

from gesture.classifier import GestureClassification
from gestures import GESTURE_INFO
from commands.command_types import CommandType, CommandEvent
from commands.command_mapping import CommandMapping
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

class GestureSmoother:
    """
    Temporal smoothing, debouncing, and trigger state-machine for hand gestures.
    Enforces frame stability, accidental trigger rejection, cooldowns, and mandatory hand-release
    to prevent unintentional repeated clicks/actions.
    """
    def __init__(self, command_mapping: CommandMapping, cfg_mgr: Optional[ConfigManager] = None):
        self.command_mapping = command_mapping
        self.cfg_mgr = cfg_mgr or config_manager

        self.min_stable_frames = self.cfg_mgr.get("gesture", "min_stable_frames", default=6)
        self.confidence_threshold = self.cfg_mgr.get("gesture", "confidence_threshold", default=0.70)
        self.require_release = self.cfg_mgr.get("gesture", "require_release_to_retrigger", default=True)
        self.hand_loss_tolerance = self.cfg_mgr.get("gesture", "hand_loss_tolerance_frames", default=3)

        self.history = deque(maxlen=9)
        self.candidate_gid: int = -1
        self.consecutive_frames: int = 0
        self.has_triggered_current_gesture: bool = False
        self.loss_frames: int = 0
        self.last_triggered_time: float = 0.0

        self.current_stable_gid: int = 0
        self.current_finger_status: List[bool] = [False] * 5

    @property
    def stability_progress(self) -> float:
        """Returns 0.0 to 1.0 indicating how close the current gesture is to firing."""
        if self.min_stable_frames <= 0:
            return 1.0
        return min(1.0, self.consecutive_frames / float(self.min_stable_frames))

    @property
    def is_waiting_for_release(self) -> bool:
        """Returns True if the gesture is held and already fired, requiring change/release."""
        return self.has_triggered_current_gesture and self.consecutive_frames >= self.min_stable_frames

    def reset(self):
        """Full reset of smoother state."""
        self.history.clear()
        self.candidate_gid = -1
        self.consecutive_frames = 0
        self.has_triggered_current_gesture = False
        self.loss_frames = 0
        self.current_stable_gid = 0

    def update(self, classification: Optional[GestureClassification], hand_detected: bool) -> Optional[CommandEvent]:
        """
        Processes one video frame's classification result with sliding-window consensus.
        Returns a CommandEvent only when a stable, valid gesture trigger occurs.
        """
        now = time.time()

        # 1. Handle Hand Absence / Loss
        if not hand_detected or classification is None:
            self.loss_frames += 1
            if self.loss_frames > self.hand_loss_tolerance:
                # Hand genuinely left camera view -> reset candidate and release lock
                self.candidate_gid = -1
                self.consecutive_frames = 0
                self.has_triggered_current_gesture = False
                self.current_stable_gid = 0
                self.history.clear()
            return None

        # Hand detected -> reset loss frame counter
        self.loss_frames = 0
        self.current_finger_status = classification.finger_status

        # 2. Reject Low-Confidence Predictions
        if classification.confidence < self.confidence_threshold:
            system_logger.debug(
                f"Ignored low-confidence gesture '{classification.name}' ({classification.confidence:.2f} < {self.confidence_threshold:.2f})"
            )
            return None

        # Push into history buffer
        self.history.append(classification.gesture_id)

        # 3. Majority Consensus Smoothing over sliding window
        # Require candidate to be the clear consensus (>60% of recent frames)
        gid_counts = {}
        for g in self.history:
            gid_counts[g] = gid_counts.get(g, 0) + 1
        consensus_gid = max(gid_counts, key=gid_counts.get)
        consensus_ratio = gid_counts[consensus_gid] / len(self.history)

        if consensus_ratio < 0.60:
            # Jittery transition between two poses -> do not count stability
            return None

        # 4. Check Gesture Stability across Consecutive Frames
        if consensus_gid == self.candidate_gid:
            self.consecutive_frames += 1
        else:
            # Gesture changed! This clears the release lock for previous gesture
            self.candidate_gid = consensus_gid
            self.consecutive_frames = 1
            self.has_triggered_current_gesture = False

        self.current_stable_gid = self.candidate_gid

        # 5. Require Configurable Stable Frame Threshold
        if self.consecutive_frames < self.min_stable_frames:
            return None

        # 6. Look up mapped command
        cmd = self.command_mapping.get_command_for_gesture(self.candidate_gid)
        if cmd == CommandType.NONE:
            return None

        # 7. Check Retrigger Prevention (Debouncing / Cooldown)
        is_repeatable = self.command_mapping.is_repeatable_gesture_command(cmd)

        if self.has_triggered_current_gesture:
            if not is_repeatable:
                # Discrete command (e.g. CONFIRM, CLICK, BACK, NEXT)
                # Requires user to change gesture or remove hand before firing again
                return None
            else:
                # Repeatable command (e.g. SCROLL_UP, VOLUME_UP)
                repeat_cooldown = self.cfg_mgr.get("gesture", "repeat_cooldown_seconds", default=0.25)
                if now - self.last_triggered_time < repeat_cooldown:
                    return None

        # 8. Fire Command Event!
        self.has_triggered_current_gesture = True
        self.last_triggered_time = now

        stable_info = GESTURE_INFO.get(self.candidate_gid, {"name": classification.name, "badge": classification.badge})

        system_logger.info(
            f"[GESTURE TRIGGER] GID: {self.candidate_gid} ({stable_info['name']}) -> "
            f"COMMAND: {cmd.value} (Stable {self.consecutive_frames} frames, Conf {classification.confidence:.2f})"
        )

        return CommandEvent(
            command=cmd,
            source="gesture",
            raw_input=f"{stable_info['badge']} ({stable_info['name']})",
            confidence=classification.confidence,
            timestamp=now
        )
