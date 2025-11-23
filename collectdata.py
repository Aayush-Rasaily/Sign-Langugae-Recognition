import os
import cv2
import numpy as np

# ----------------------------
# SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
DATA_DIR = 'Images'
IMG_SIZE = 64

# Define labels and their keys
labels = {
    **{chr(i + 97): chr(i + 65) for i in range(26)},  # 'a'-'z' -> 'A'-'Z'
    '1': 'HELLO',
    '2': 'LOVE',
    '3': 'ILOVEYOU',
    '4': 'RIGHT',
    '5': 'NO',
    '6': 'OKAY'
}

# Ensure all label folders exist
for label in labels.values():
    path = os.path.join(DATA_DIR, label)
    os.makedirs(path, exist_ok=True)

# Initialize image counters for each label
count = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in labels.values()}

print("✅ Data collection started")
print("Press keys a-z or 1-6 to capture samples")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Define ROI box
    x1, y1, x2, y2 = 100, 100, 364, 364
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    roi = frame[y1:y2, x1:x2]

    # Show ROI preview
    cv2.imshow("ROI", roi)

    # Display instructions on main window
    cv2.putText(frame, "Press key (a-z or 1-6) to capture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    # If key pressed corresponds to a label
    char = chr(key)
    if char in labels:
        label_name = labels[char]
        label_path = os.path.join(DATA_DIR, label_name)

        # Resize and save image
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img_path = os.path.join(label_path, f"{count[label_name]}.png")
        cv2.imwrite(img_path, resized)
        count[label_name] += 1

        print(f"📸 Saved: {img_path}")

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection stopped")
