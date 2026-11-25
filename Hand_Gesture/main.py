import sys
import os
import math
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
for p in (script_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# Auto re-exec using local venv if system python is missing dependencies
venv_dir = os.path.join(root_dir, "venv")
venv_python = os.path.join(venv_dir, "bin", "python")
if os.path.exists(venv_python) and sys.prefix != venv_dir:
    try:
        import cv2
        import mediapipe
        import sklearn
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

try:
    import mp_compat
except Exception:
    from . import mp_compat

try:
    from gestures import GESTURE_INFO
except Exception:
    from .gestures import GESTURE_INFO

import numpy as np
import cv2
import mediapipe as mp
import pickle

# Robust model path resolution
possible_paths = [
    os.path.join(script_dir, 'model.p'),
    os.path.join(script_dir, '..', 'model.p'),
    os.path.join(script_dir, '..', 'hand_gesture_samplae', 'model.p'),
    os.path.join(script_dir, 'hand_gesture_samplae', 'model.p'),
    '../hand_gesture_samplae/model.p',
    'model.p'
]
model_path = next((p for p in possible_paths if os.path.exists(p)), None)
model = None
model_supports_full_gesture_set = False
if model_path:
    try:
        with open(model_path, 'rb') as f:
            mode_dict = pickle.load(f)
            model = mode_dict.get('model', None)
        if model is not None:
            trained_ids = {int(label) for label in getattr(model, 'classes_', [])}
            model_supports_full_gesture_set = set(GESTURE_INFO).issubset(trained_ids)
    except Exception as e:
        print(f"Notice: Model could not be loaded ({e}), using geometric gesture analyzer.")

def euclidean_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def analyze_hand_gesture(landmarks):
    """
    Classifies a static hand pose using MediaPipe landmark geometry.
    Returns:
        gesture_id (int): 0-26
        info (dict): GESTURE_INFO entry
        finger_status (list of bool): [Thumb, Index, Middle, Ring, Pinky]
    """
    wrist = landmarks[0]
    hand_size = euclidean_dist(wrist, landmarks[9])
    if hand_size < 1e-4:
        hand_size = 1.0

    d = euclidean_dist

    # Check extension for 4 fingers (Index, Middle, Ring, Pinky)
    # A finger is extended if tip is significantly farther from wrist than PIP joint
    index_ext = d(wrist, landmarks[8]) > d(wrist, landmarks[6]) * 1.15 and d(wrist, landmarks[8]) > d(wrist, landmarks[5]) * 1.2
    middle_ext = d(wrist, landmarks[12]) > d(wrist, landmarks[10]) * 1.15 and d(wrist, landmarks[12]) > d(wrist, landmarks[9]) * 1.2
    ring_ext = d(wrist, landmarks[16]) > d(wrist, landmarks[14]) * 1.15 and d(wrist, landmarks[16]) > d(wrist, landmarks[13]) * 1.2
    pinky_ext = d(wrist, landmarks[20]) > d(wrist, landmarks[18]) * 1.15 and d(wrist, landmarks[20]) > d(wrist, landmarks[17]) * 1.2

    # Upright palm check: if wrist is below palm base, tips should also be higher than PIPs
    if wrist.y > landmarks[9].y:
        index_ext = index_ext or (landmarks[8].y < landmarks[6].y and landmarks[8].y < landmarks[5].y)
        middle_ext = middle_ext or (landmarks[12].y < landmarks[10].y and landmarks[12].y < landmarks[9].y)
        ring_ext = ring_ext or (landmarks[16].y < landmarks[14].y and landmarks[16].y < landmarks[13].y)
        pinky_ext = pinky_ext or (landmarks[20].y < landmarks[18].y and landmarks[20].y < landmarks[17].y)

    # Thumb extension:
    # Check distance of thumb tip (4) relative to pinky MCP (17) and middle MCP (9)
    thumb_dist_pinky = d(landmarks[4], landmarks[17])
    thumb_mcp_dist_pinky = d(landmarks[2], landmarks[17])
    thumb_dist_middle_mcp = d(landmarks[4], landmarks[9])
    thumb_up = landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y and landmarks[4].y < landmarks[5].y
    thumb_down = landmarks[4].y > landmarks[3].y and landmarks[4].y > landmarks[2].y
    thumb_ext = (thumb_dist_pinky > thumb_mcp_dist_pinky * 1.10 or thumb_up or thumb_down) and thumb_dist_middle_mcp > hand_size * 0.42

    # Thumb pointing direction:
    thumb_side = (
        not thumb_up and not thumb_down and
        d(landmarks[4], landmarks[2]) > hand_size * 0.45 and
        d(landmarks[4], landmarks[5]) > hand_size * 0.60
    )

    # Relative distances between finger tips
    thumb_index_dist = d(landmarks[4], landmarks[8]) / hand_size
    index_middle_dist = d(landmarks[8], landmarks[12]) / hand_size
    middle_ring_dist = d(landmarks[12], landmarks[16]) / hand_size
    ring_pinky_dist = d(landmarks[16], landmarks[20]) / hand_size

    finger_status = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
    ext_count = sum((index_ext, middle_ext, ring_ext, pinky_ext))
    all_four_extended = ext_count == 4
    compact_fingers = max(index_middle_dist, middle_ring_dist, ring_pinky_dist) < 0.25
    fingers_crossed = (
        (landmarks[5].x - landmarks[9].x) * (landmarks[8].x - landmarks[12].x) < 0
    )

    # Pinched fingers / Italian "Che Vuoi" (24): all five fingertips gathered together pointing upward
    tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    tip_cx = sum(t.x for t in tips) / 5.0
    tip_cy = sum(t.y for t in tips) / 5.0
    max_tip_spread = max(math.hypot(t.x - tip_cx, t.y - tip_cy) for t in tips) / hand_size
    tips_pointing_up = tip_cy < landmarks[9].y
    dist_wrist_tip_centroid = math.hypot(tip_cx - wrist.x, tip_cy - wrist.y) / hand_size
    if max_tip_spread < 0.28 and tips_pointing_up and dist_wrist_tip_centroid > 0.95 and thumb_index_dist < 0.28:
        return 24, GESTURE_INFO[24], finger_status

    # Pinch shapes take precedence over the finger-extension pattern.
    if thumb_index_dist < 0.34 and middle_ext and ring_ext and pinky_ext:
        return 6, GESTURE_INFO[6], finger_status
    if thumb_index_dist < 0.34 and not middle_ext and not ring_ext and not pinky_ext:
        return 18, GESTURE_INFO[18], finger_status

    # I Love You (ILY), Rock On, and Call Me / Shaka.
    if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
        return 5, GESTURE_INFO[5], finger_status
    if index_ext and pinky_ext and not thumb_ext and not middle_ext and not ring_ext:
        return 8, GESTURE_INFO[8], finger_status
    if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return 15, GESTURE_INFO[15], finger_status

    # Half Heart / ILY: Index extended horizontally or curved, thumb extended downward/outward, other 3 fingers curled
    index_pointing_out = (
        d(landmarks[8], landmarks[5]) > hand_size * 0.38 or
        d(landmarks[8], landmarks[9]) > hand_size * 0.40 or
        abs(landmarks[8].x - landmarks[5].x) > hand_size * 0.35
    )
    if (index_pointing_out and (thumb_down or thumb_ext or landmarks[4].y > landmarks[2].y) and
            landmarks[8].y < landmarks[4].y and
            not middle_ext and not ring_ext and not pinky_ext and
            thumb_index_dist > 0.25):
        return 25, GESTURE_INFO[25], finger_status

    # Thumb direction distinguishes yes, thumbs down, and hitchhiker when other fingers are curled.
    if ext_count == 0 and thumb_up:
        return 1, GESTURE_INFO[1], finger_status
    if ext_count == 0 and thumb_down:
        return 17, GESTURE_INFO[17], finger_status
    if ext_count == 0 and thumb_side:
        return 23, GESTURE_INFO[23], finger_status

    # Vulcan Salute (Live Long and Prosper - 21) before general 4-finger poses
    if (all_four_extended and index_middle_dist < 0.30 and ring_pinky_dist < 0.30 and
            middle_ring_dist > 0.32 and middle_ring_dist > 1.25 * max(index_middle_dist, ring_pinky_dist)):
        return 21, GESTURE_INFO[21], finger_status

    # ASL C (25): fingers curved together in an arc opposite curved thumb
    c_finger_curve = (
        0.75 < d(wrist, landmarks[8]) / hand_size < 1.30 and
        0.75 < d(wrist, landmarks[12]) / hand_size < 1.30 and
        compact_fingers and
        0.30 < thumb_index_dist < 0.85 and
        thumb_ext
    )
    if c_finger_curve:
        return 25, GESTURE_INFO[25], finger_status

    # The 4-finger open-palm classes are distinguished by thumb and finger spacing.
    if all_four_extended and not thumb_ext and compact_fingers:
        return 2, GESTURE_INFO[2], finger_status
    if all_four_extended and thumb_ext and compact_fingers:
        return 3, GESTURE_INFO[3], finger_status
    if all_four_extended and thumb_ext:
        return 4, GESTURE_INFO[4], finger_status
    if all_four_extended and not thumb_ext:
        return 14, GESTURE_INFO[14], finger_status

    # Familiar single-finger and counting poses.
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        # ASL D (26): index pointing up, thumb touching curled middle fingertip
        if d(landmarks[4], landmarks[12]) / hand_size < 0.28:
            return 26, GESTURE_INFO[26], finger_status
        return 9, GESTURE_INFO[9], finger_status
    if middle_ext and not index_ext and not ring_ext and not pinky_ext and not thumb_ext:
        return 10, GESTURE_INFO[10], finger_status
    if pinky_ext and not index_ext and not middle_ext and not ring_ext and not thumb_ext:
        return 11, GESTURE_INFO[11], finger_status
    if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
        return 13, GESTURE_INFO[13], finger_status

    # Crossed fingers (good luck) and an uncrossed count of 3 share an extension pattern.
    if thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
        if fingers_crossed:
            return 12, GESTURE_INFO[12], finger_status
        return 19, GESTURE_INFO[19], finger_status
    if thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext:
        return 20, GESTURE_INFO[20], finger_status

    # Two fingers extended: Peace (7) vs Together / ASL U (22)
    if index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if index_middle_dist < 0.22:
            return 22, GESTURE_INFO[22], finger_status
        return 7, GESTURE_INFO[7], finger_status
    if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return 16, GESTURE_INFO[16], finger_status

    # A closed fist with thumb curled
    if ext_count == 0:
        return 0, GESTURE_INFO[0], finger_status

    return 0, GESTURE_INFO[0], finger_status


def draw_hud(frame, current_gesture, finger_status):
    """Draws an elegant, readable status overlay on the video frame."""
    h, w, _ = frame.shape
    num_classes = len(GESTURE_INFO)
    num_rows = (num_classes + 1) // 2
    row_h = 16
    guide_bottom = min(h - 10, 105 + num_rows * row_h + 10)

    # 1. Top HUD Header background (semi-transparent dark bar)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    # Two-column gesture guide lists all classes.
    guide_left = max(5, w - 360)
    cv2.rectangle(overlay, (guide_left, 70), (w - 8, guide_bottom), (30, 30, 30), -1)
    alpha = 0.72
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Detected Gesture Banner
    if current_gesture is not None:
        badge = current_gesture['badge']
        desc = current_gesture['desc']
        
        cv2.putText(frame, f"GESTURE: {badge}", (15, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (50, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pose: {desc}", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Show a hand gesture to the camera", (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

    # 3. Finger Status Pills (T, I, M, R, P)
    if finger_status is not None:
        labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        start_x = 420 if w > 700 else w - 280
        for i, (f_name, is_on) in enumerate(zip(labels, finger_status)):
            fx = start_x + (i * 52)
            if fx + 48 < w:
                color = (40, 220, 40) if is_on else (80, 80, 80)
                text_col = (255, 255, 255)
                cv2.rectangle(frame, (fx, 15), (fx + 46, 45), color, -1)
                cv2.rectangle(frame, (fx, 15), (fx + 46, 45), (200, 200, 200), 1)
                cv2.putText(frame, f_name[0], (fx + 16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2, cv2.LINE_AA)

    # 4. Side Reference Guide
    cv2.putText(frame, f"GESTURE GUIDE: {num_classes}", (guide_left + 8, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 255, 255), 2, cv2.LINE_AA)
    for gesture_id, info in GESTURE_INFO.items():
        col = gesture_id % 2
        row = gesture_id // 2
        short_name = info['name']
        x = guide_left + 8 + col * 176
        y = 107 + row * row_h
        if y < guide_bottom - 2:
            cv2.putText(frame, f"{gesture_id}: {short_name}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)


def run_legacy_demo():
    """Legacy visual-only classifier demo loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    padding = 30
    gesture_history = deque(maxlen=4)

    window_name = 'Sign Language'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    print("Sign Language Gesture Recognition is running. Press 'f' for fullscreen, 'h' or 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        current_gesture = None
        last_finger_status = None

        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
                gid, ginfo, finger_status = analyze_hand_gesture(hand_landmark.landmark)
                last_finger_status = finger_status

                x_coords = [lm.x for lm in hand_landmark.landmark]
                y_coords = [lm.y for lm in hand_landmark.landmark]
                min_x = min(x_coords)
                min_y = min(y_coords)
                temp_data = []
                for lm in hand_landmark.landmark:
                    temp_data.append(lm.x - min_x)
                    temp_data.append(lm.y - min_y)

                if model is not None and len(temp_data) == 42:
                    try:
                        ml_pred = int(model.predict([np.asarray(temp_data)])[0])
                        if ml_pred in GESTURE_INFO and (
                            model_supports_full_gesture_set or (gid == 0 and ml_pred != 0)
                        ):
                            gid = ml_pred
                            ginfo = GESTURE_INFO[gid]
                    except Exception:
                        pass

                gesture_history.append(gid)
                stable_gid = max(set(gesture_history), key=gesture_history.count)
                current_gesture = GESTURE_INFO[stable_gid]

                x1 = max(0, int(min(x_coords) * w) - padding)
                y1 = max(0, int(min(y_coords) * h) - padding)
                x2 = min(w, int(max(x_coords) * w) + padding)
                y2 = min(h, int(max(y_coords) * h) + padding)

                box_color = (0, 240, 100) if stable_gid != 0 else (60, 60, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                badge_text = current_gesture['badge']
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                badge_y = max(th + 10, y1 - 10)
                cv2.rectangle(frame, (x1, badge_y - th - 6), (x1 + tw + 12, badge_y + 4), box_color, -1)
                cv2.putText(frame, badge_text, (x1 + 6, badge_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        draw_hud(frame, current_gesture, last_finger_status)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xff
        if key in (ord('h'), ord('q'), 27):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def main():
    """Launches the upgraded unified HAND + VOICE computer control system."""
    import argparse
    parser = argparse.ArgumentParser(description="Unified Hand + Voice HCI")
    parser.add_argument("--classic", action="store_true", help="Run classic visual-only classifier demo")
    parser.add_argument("--mode", type=str, default="HYBRID", help="GESTURE, VOICE, or HYBRID")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending real OS events")
    parser.add_argument("--fullscreen", action="store_true", help="Start application in fullscreen mode")
    args, _ = parser.parse_known_args()

    if args.classic:
        run_legacy_demo()
    else:
        try:
            from system_controller import UnifiedHCIController
            from commands.command_types import SystemMode
            mode = SystemMode.from_string(args.mode)
            controller = UnifiedHCIController(mode=mode, dry_run=args.dry_run)
            controller.run(fullscreen=args.fullscreen)
        except Exception as e:
            print(f"Notice: Launching fallback legacy demo ({e})")
            run_legacy_demo()

if __name__ == '__main__':
    main()

