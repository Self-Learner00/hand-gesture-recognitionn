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
        import sklearn
        import numpy
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

possible_data_files = [
    os.path.join(script_dir, 'data.pickle'),
    os.path.join(script_dir, '..', 'data.pickle'),
    os.path.join(script_dir, '..', 'hand_gesture_samplae', 'data.pickle'),
    os.path.join(script_dir, 'hand_gesture_samplae', 'data.pickle'),
    '../hand_gesture_samplae/data.pickle',
    'data.pickle'
]
data_path = next((p for p in possible_data_files if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError("data.pickle not found! Please run Create_dataset.py first.")

with open(data_path, 'rb') as f:
    data_dir = pickle.load(f)

data = np.asarray(data_dir['data'])
labels = np.asarray(data_dir['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"{score * 100:.2f}% of samples were classified correctly!")

model_save_path = os.path.join(script_dir, 'model.p')
with open(model_save_path, 'wb') as f:
    pickle.dump({'model': model}, f)
print(f"Model saved successfully to {model_save_path}")
