import cv2
import os
import numpy as np
import mediapipe as mp

# Path to store images
DATA_DIR = "Images/"

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def capture_images(label, num_samples=100):
    """
    Captures images from webcam and saves them in .npy format.
    """
    save_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(save_path):
        print(f"❌ Folder '{save_path}' not found. Run your folder creation script first.")
        return

    cap = cv2.VideoCapture(0)
    print(f"📸 Starting capture for '{label}' ... Press 'q' to quit early")

    data_list = []
    count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                # Get bounding box
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0
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
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = roi.astype("float32") / 255.0  # normalize
                data_list.append(roi)

                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                count += 1
                cv2.putText(frame, f"{label}: {count}/{num_samples}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Capture - " + label, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Save as .npy
    np.save(os.path.join(save_path, f"{label}.npy"), np.array(data_list))
    print(f"✅ Saved {len(data_list)} frames for '{label}'")

    cap.release()
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    capture_images("HELLO", num_samples=50)
    capture_images("LOVE", num_samples=50)
    capture_images("ILOVEYOU", num_samples=50)
    capture_images("RIGHT", num_samples=50)
    capture_images("NO", num_samples=50)
    capture_images("OKAY", num_samples=50)
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        capture_images(char, num_samples=50)
    print("All captures complete.")
    
    