import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dataset directory
DATA_DIR = 'MP_DATA'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["HELLO", "LOVE", "ILOVEYOU", "RIGHT", "NO", "OKAY"]

# Create folders for reference
for label in labels:
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

def collect_data(label, num_samples=100):
    cap = cv2.VideoCapture(0)
    data_list = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        count = 0
        while cap.isOpened() and count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                # Get bounding box of hand
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                # Add padding
                pad = 20
                x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
                x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

                roi = frame[y_min:y_max, x_min:x_max]
                roi = cv2.resize(roi, (64, 64))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # keep RGB
                roi = roi.astype("float32") / 255.0  # normalize
                data_list.append(roi)

                count += 1
                cv2.putText(frame, f"Collecting {label}: {count}/{num_samples}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save as .npy
    save_path = os.path.join(DATA_DIR, f"{label}.npy")
    np.save(save_path, np.array(data_list))
    print(f"Saved {len(data_list)} frames for label '{label}' to {save_path}")

if __name__ == "__main__":
    print("Available labels:", labels)
    label = input("Enter the label you want to collect: ").strip().upper()
    if label in labels:
        collect_data(label, num_samples=200)
    else:
        print("Invalid label! Please choose from:", labels)