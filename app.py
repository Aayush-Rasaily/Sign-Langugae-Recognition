import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import time
import os

# ----------------------------
# LOAD MODEL AND LABELS
# ----------------------------
model = load_model("my_sign_model.h5")

with open("labels.json", "r") as f:
    labels = json.load(f)

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ----------------------------
# CAPTURE SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
prev_time = 0

print("✅ Starting Sign Language Detection...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for selfie-view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, c = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Crop hand region
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess for model
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict
            preds = model.predict(hand_img)
            label_idx = np.argmax(preds)
            confidence = np.max(preds)

            if confidence > 0.7:
                label = labels[label_idx]
                text = f"{label.upper()} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  
print("✅ Sign Language Detection Stopped.")