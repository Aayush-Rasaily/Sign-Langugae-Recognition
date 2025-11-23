import os
import cv2
import numpy as np

directory = 'Images/'

# Labels
alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
special_words = ["HELLO", "LOVE", "ILOVEYOU", "RIGHT", "NO", "OKAY"]
all_labels = alphabets + special_words

# Create folders if they don't exist
for folder in all_labels:
    path = os.path.join(directory, folder)
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize count for each label
count = {label: len(os.listdir(os.path.join(directory, label))) for label in all_labels}

# Key mapping for letters (a-z) and special words (1-6)
labels_keys = {chr(i + 97): alphabets[i] for i in range(26)}
labels_keys.update({
    '1': "HELLO",
    '2': "LOVE",
    '3': "ILOVEYOU",
    '4': "RIGHT",
    '5': "NO",
    '6': "OKAY"
})

cap = cv2.VideoCapture(0)
x1, y1, x2, y2 = 0, 40, 300, 400  # ROI coordinates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imshow("Data Collection", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    char = chr(key)
    if char in labels_keys:
        label_name = labels_keys[char]
        img_path = os.path.join(directory, label_name, f"{count[label_name]}.png")
        cv2.imwrite(img_path, roi)
        count[label_name] += 1
        print(f"Saved {img_path}")

cap.release()
cv2.destroyAllWindows()
