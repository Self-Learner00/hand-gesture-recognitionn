import os
import sys

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
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import cv2

try:
    from gestures import GESTURE_INFO, GESTURE_PROMPTS
except Exception:
    from .gestures import GESTURE_INFO, GESTURE_PROMPTS

DATA_DIR = os.path.join(script_dir, 'data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

no_of_gesture_class = len(GESTURE_INFO)
data_size = 300

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera device 0.")
    sys.exit(1)

for i in range(no_of_gesture_class):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    gesture = GESTURE_INFO[i]
    prompt = GESTURE_PROMPTS[i]
    print(f"Collecting class {i}: {gesture['name']} — {prompt}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preview = frame.copy()
        cv2.putText(preview, f"Class {i}/{no_of_gesture_class - 1}: {gesture['name']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, prompt[:64], (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(preview, "Press 'h' to start collecting", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Gesture Frame', preview)
        if cv2.waitKey(30) & 0xff == ord('h'):
            break

    count = 0
    while count < data_size:
        ret, frame = cap.read()
        if not ret:
            break
        preview = frame.copy()
        cv2.putText(preview, f"{gesture['name']}: {count + 1}/{data_size}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Gesture Frame', preview)
        cv2.waitKey(30)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(count)), frame)
        count +=1

    if count < data_size:
        print(f"Camera stopped while collecting class {i}; saved {count} samples.")
        break

cap.release()
cv2.destroyAllWindows()
