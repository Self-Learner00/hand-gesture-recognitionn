import cv2
import numpy as np
import time
from typing import Optional, List, Tuple
from commands.command_types import SystemMode, CommandType
from gesture.classifier import GestureClassification

class DashboardHUD:
    """
    Renders an uncluttered, modern HUD designed for maximum camera visibility.
    Information is streamlined and tucked along edges/corners so the hand and interaction area
    remain unobstructed.
    """
    def __init__(self):
        # Color palette (BGR format)
        self.COLOR_BG = (18, 18, 22)
        self.COLOR_PANEL_BG = (28, 28, 34)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_SUBTEXT = (180, 185, 190)
        self.COLOR_ACCENT_GREEN = (60, 225, 110)
        self.COLOR_ACCENT_BLUE = (255, 175, 45)
        self.COLOR_ACCENT_PURPLE = (225, 90, 210)
        self.COLOR_ACCENT_YELLOW = (40, 220, 255)
        self.COLOR_ACCENT_RED = (60, 60, 240)

    def draw(
        self,
        frame: np.ndarray,
        mode: SystemMode,
        is_paused: bool,
        classification: Optional[GestureClassification],
        stability_progress: float,
        is_waiting_for_release: bool,
        last_command: CommandType,
        last_action: str,
        last_source: str,
        action_status: str,
        cooldown_remaining: float,
        max_cooldown: float,
        mic_status: str,
        is_voice_listening: bool,
        last_voice_transcript: str,
        fps: float,
        voice_status: str = ""
    ) -> np.ndarray:
        h, w, _ = frame.shape
        overlay = frame.copy()
        alpha = 0.70

        # ==============================================================
        # 1. TOP SLIM STATUS BAR (Height: 38px)
        # Sidelined across top with minimal height to leave screen clear.
        # ==============================================================
        top_h = 38
        cv2.rectangle(overlay, (0, 0), (w, top_h), self.COLOR_BG, -1)

        # Mode Badge Pill
        mode_text = f"{mode.value}"
        if mode == SystemMode.GESTURE:
            mode_color = self.COLOR_ACCENT_BLUE
        elif mode == SystemMode.VOICE:
            mode_color = self.COLOR_ACCENT_PURPLE
        else:
            mode_color = self.COLOR_ACCENT_GREEN

        cv2.rectangle(overlay, (8, 6), (110, 32), mode_color, -1)

        # Pause Indicator Pill
        pause_text = "PAUSED" if is_paused else "ACTIVE"
        pause_color = self.COLOR_ACCENT_RED if is_paused else (45, 160, 50)
        cv2.rectangle(overlay, (116, 6), (195, 32), pause_color, -1)

        # Blend top bar overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw top text labels (crisp on top of blended bar)
        cv2.putText(frame, mode_text, (18, 25), cv2.FONT_HERSHEY_DUPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)
        cv2.putText(frame, pause_text, (126, 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Mic & Voice status (compact text)
        mic_color = (60, 220, 100) if "CONNECTED" in mic_status else (80, 140, 240)
        mic_pill = f"MIC: {mic_status[:12]}"
        cv2.putText(frame, mic_pill, (208, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mic_color, 1, cv2.LINE_AA)

        v_state = "LISTENING" if is_voice_listening else "IDLE"
        v_color = self.COLOR_ACCENT_PURPLE if is_voice_listening else self.COLOR_SUBTEXT
        cv2.putText(frame, f"VOICE: {v_state}", (340, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, v_color, 1, cv2.LINE_AA)

        # FPS indicator in top-right
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 78, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 240, 180), 1, cv2.LINE_AA)

        # Shortcuts reminder in center-right of top bar
        hints = "[M] Mode  [P] Pause  [F] Fullscreen  [Q] Quit"
        cv2.putText(frame, hints, (w - 380, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_SUBTEXT, 1, cv2.LINE_AA)

        # ==============================================================
        # 2. COMPACT, SIDELINED ACTION & GESTURE CARD (Top-Left)
        # Width: 260px, Height: 110px. Sidelined so center is 100% free!
        # ==============================================================
        card_w = 260
        card_h = 106
        cx1, cy1 = 8, 44
        cx2, cy2 = cx1 + card_w, cy1 + card_h

        card_overlay = frame.copy()
        cv2.rectangle(card_overlay, (cx1, cy1), (cx2, cy2), self.COLOR_PANEL_BG, -1)
        cv2.addWeighted(card_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (60, 60, 70), 1)

        # Line 1: Gesture name & mini stability gauge
        if classification:
            g_badge = classification.badge
            if len(g_badge) > 22:
                g_badge = g_badge[:20] + ".."
            cv2.putText(frame, g_badge, (cx1 + 8, cy1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.48, self.COLOR_TEXT, 1, cv2.LINE_AA)

            # Mini Stability bar right aligned in card
            sb_x = cx1 + card_w - 75
            sb_y = cy1 + 10
            sb_w = 65
            sb_h = 7
            cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (50, 50, 50), -1)
            fill_w = int(sb_w * stability_progress)
            col_bar = self.COLOR_ACCENT_GREEN if stability_progress >= 0.99 else self.COLOR_ACCENT_BLUE
            if fill_w > 0:
                cv2.rectangle(frame, (sb_x, sb_y), (sb_x + fill_w, sb_y + sb_h), col_bar, -1)
        else:
            cv2.putText(frame, "No hand in frame", (cx1 + 8, cy1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, self.COLOR_SUBTEXT, 1, cv2.LINE_AA)

        # Line 2: Unified Command
        cmd_str = last_command.value if last_command != CommandType.NONE else "NONE"
        cv2.putText(frame, f"CMD: {cmd_str}", (cx1 + 8, cy1 + 44), cv2.FONT_HERSHEY_DUPLEX, 0.58, self.COLOR_ACCENT_YELLOW, 1, cv2.LINE_AA)
        if last_source != "NONE":
            cv2.putText(frame, f"({last_source})", (cx1 + card_w - 65, cy1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_SUBTEXT, 1, cv2.LINE_AA)

        # Line 3: Action Executed
        cv2.putText(frame, f"ACT: {last_action}", (cx1 + 8, cy1 + 68), cv2.FONT_HERSHEY_DUPLEX, 0.52, self.COLOR_ACCENT_GREEN, 1, cv2.LINE_AA)

        # Line 4: Status / Cooldown
        status_disp = action_status
        status_color = self.COLOR_ACCENT_GREEN
        if is_waiting_for_release:
            status_disp = "HELD - Release hand to retrigger"
            status_color = self.COLOR_ACCENT_YELLOW
        elif cooldown_remaining > 0:
            status_disp = f"Cooldown {cooldown_remaining:.1f}s"
            status_color = self.COLOR_ACCENT_BLUE

        if len(status_disp) > 34:
            status_disp = status_disp[:32] + ".."
        cv2.putText(frame, status_disp, (cx1 + 8, cy1 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

        # Mini Cooldown Progress Line at bottom of card
        if max_cooldown > 0:
            cd_ratio = max(0.0, min(1.0, cooldown_remaining / max_cooldown))
            cd_w = int((card_w - 16) * cd_ratio)
            cv2.rectangle(frame, (cx1 + 8, cy2 - 6), (cx2 - 8, cy2 - 3), (45, 45, 55), -1)
            if cd_w > 0:
                cv2.rectangle(frame, (cx1 + 8, cy2 - 6), (cx1 + 8 + cd_w, cy2 - 3), self.COLOR_ACCENT_BLUE, -1)

        # ==============================================================
        # 3. BOTTOM SLIM VOICE & FINGER STATUS TICKER (Height: 34px)
        # Sidelined at the very bottom, keeping the center view vast and open.
        # ==============================================================
        bot_h = 34
        by1 = max(0, h - bot_h)
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, by1), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(bot_overlay, alpha, frame, 1 - alpha, 0, frame)

        # Voice Transcript & Action Ticker (Left-aligned)
        if last_voice_transcript:
            v_text = f"Heard: \"{last_voice_transcript}\""
            if voice_status:
                v_text += f" -> {voice_status}"
        else:
            v_text = "Voice: Ready (say 'confirm', 'cancel', 'click', 'scroll up', etc.)"

        if len(v_text) > 85:
            v_text = v_text[:82] + "..."
        cv2.putText(frame, v_text, (10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # Compact Finger Status Pills (Right-aligned in bottom bar)
        if classification and classification.finger_status:
            labels = ['T', 'I', 'M', 'R', 'P']
            pill_start_x = w - 130
            for i, (lbl, is_ext) in enumerate(zip(labels, classification.finger_status)):
                fx = pill_start_x + (i * 24)
                col = (40, 220, 60) if is_ext else (65, 65, 75)
                cv2.rectangle(frame, (fx, by1 + 6), (fx + 20, by1 + 28), col, -1)
                cv2.rectangle(frame, (fx, by1 + 6), (fx + 20, by1 + 28), (140, 140, 150), 1)
                cv2.putText(frame, lbl, (fx + 5, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
