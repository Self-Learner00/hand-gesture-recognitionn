import sys
import os

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
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

try:
    import mp_compat
except Exception:
    from . import mp_compat

import mediapipe as mp
import pickle
import cv2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

possible_data_dirs = [
    os.path.join(script_dir, 'data'),
    os.path.join(script_dir, '..', 'data'),
    os.path.join(script_dir, '..', 'hand_gesture_samplae', 'data'),
    os.path.join(script_dir, 'hand_gesture_samplae', 'data'),
]
DATA_DIR = next((d for d in possible_data_dirs if os.path.exists(d)), os.path.join(script_dir, 'data'))

data = []
labels = []

if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print(f"Data directory '{DATA_DIR}' does not exist or is empty. Run Collect_Data.py first.")
    sys.exit(0)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmark = result.multi_hand_landmarks[0]
            x_ = [lm.x for lm in hand_landmark.landmark]
            y_ = [lm.y for lm in hand_landmark.landmark]
            min_x = min(x_)
            min_y = min(y_)

            temp_data = []
            for lm in hand_landmark.landmark:
                temp_data.append(lm.x - min_x)
                temp_data.append(lm.y - min_y)

            if len(temp_data) == 42:
                data.append(temp_data)
                labels.append(dir_)

if len(data) == 0:
    print(f"No valid hand gesture samples found in '{DATA_DIR}'.")
    print("Please run Collect_Data.py first to capture gesture images from your webcam.")
    sys.exit(0)

output_path = os.path.join(script_dir, 'data.pickle')
with open(output_path, 'wb') as file:
    pickle.dump({'data': data, 'labels': labels}, file)
print(f"Dataset successfully created ({len(data)} samples) and saved to {output_path}")
