# Hand Gesture Recognition - Sign Language Detection

A real-time **Hand Gesture Recognition System** for detecting and classifying **Sign Language gestures** using computer vision (OpenCV), **MediaPipe** hand landmarks, and a **Machine Learning model (Random Forest)**.

This project allows you to collect gesture images, process and extract hand landmark data, train a model, and run real-time sign language detection using your webcam.

---
# Demo video -  https://lnkd.in/p/gkkskVBD
---

---
## 📚 Features
- Real-time gesture recognition using webcam
- Single and dual hand detection
- 27 static gesture classes, including counting poses, common signals, and ASL handshapes
- MediaPipe Hands for robust landmark detection
- RandomForestClassifier trained from collected hand landmark data
- Modular code structure (easy to modify or extend)

---

## 🗂️ Project Structure
```bash
Hand_Gesture/
│
├── Collect_Data.py        # Collect gesture images using webcam
├── Create_dataset.py      # Extract hand landmarks from collected images
├── ML_Model.py            # Train the Random Forest model
├── main.py                # Real-time hand gesture recognition script
├── gestures.py            # Shared gesture labels and pose prompts
├── data/                  # Contains gesture images organized by class
├── data.pickle            # Processed data (features + labels)
├── model.p                # Trained Random Forest model
```

---

## ⚡ Installation
```bash
pip install opencv-python mediapipe scikit-learn numpy
```

---

## 🔧 How to Use

### 1. Data Collection:
Run the following command and show each gesture when prompted:
```bash
python Hand_Gesture/Collect_Data.py    # Collects 300 samples for each of the 27 gesture classes
```

### 2. Extract Landmark Data:
```bash
python Hand_Gesture/Create_dataset.py      # Creates Hand_Gesture/data.pickle
```

### 3. Train ML Model:
```bash
python Hand_Gesture/ML_Model.py      # Trains all collected classes and creates Hand_Gesture/model.p
```

### 4. Real-Time Prediction:
```bash
python Hand_Gesture/main.py
```
Press **'h'** to close the window.

---

## 🔎 The 27 Gesture Classes

Collect gestures in the order shown. The collector displays the class name and pose prompt before capturing each set.

| ID | Gesture | Pose |
|----|---------|------|
| 0 | No / Fist | Closed fist |
| 1 | Yes / Thumbs Up | Thumb up, other fingers curled |
| 2 | Thank you | Four fingers straight and close, thumb tucked |
| 3 | Greetings | Open palm with fingers together |
| 4 | Hello | Open palm with fingers spread |
| 5 | I Love You (ILY) | Thumb, index, and pinky extended |
| 6 | OK | Thumb and index touch; other fingers extended |
| 7 | Peace | Index and middle fingers extended |
| 8 | Rock On | Index and pinky extended |
| 9 | Point | Index finger extended |
| 10 | Middle Finger | Middle finger extended |
| 11 | ASL I | Pinky extended |
| 12 | Crossed Fingers / Good Luck | Index and middle crossed, with thumb extended |
| 13 | Count Three | Index, middle, and ring fingers extended |
| 14 | Count Four | Four fingers spread, thumb tucked |
| 15 | Call Me / Shaka | Thumb and pinky extended |
| 16 | L Shape | Thumb and index extended at a right angle |
| 17 | Thumbs Down | Thumb down, other fingers curled |
| 18 | Pinch | Thumb and index touch; other fingers curled |
| 19 | Count Three (Thumb) | Thumb, index, and middle extended |
| 20 | Count Four (Thumb) | Thumb, index, middle, and ring extended |
| 21 | Vulcan Salute | Index & middle paired, ring & pinky paired, separated in a V |
| 22 | Together / ASL U | Index and middle extended straight and touching together |
| 23 | Hitchhiker | Closed fist with thumb extended horizontally sideways |
| 24 | Pinched Fingers | All five fingertips touching together pointing upwards (🤌 Che Vuoi) |
| 25 | I Love You (Half Heart / C) | Fingers and thumb curved into an open C / half-heart shape (ILY) |
| 26 | ASL D | Index pointing straight up; thumb touching curled middle fingertip |

The live recognizer classifies static hand poses; it does not analyze motion between frames. Retrain the model after collecting all 27 classes to use the learned classifier for the full set.

---

## 💪 Future Improvements
- Improve model accuracy using deep learning (e.g., CNN)
- Use LSTM for sequence-based gesture recognition
- Export as web or mobile app

---

## 👍 Acknowledgements
- [MediaPipe by Google](https://mediapipe.dev/)
- OpenCV
- scikit-learn

---

## 📅 License
This project is open-source and free to use for learning, research, and personal projects. No formal license is attached—just give credit if you find it helpful!
