import os
import numpy as np
import cv2

# Path to your image dataset
DATA_DIR = "Images"   # or "MP_DATA" if that’s your folder

# Output directory for npy files
OUTPUT_DIR = "MP_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Labels (same as before)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["HELLO", "LOVE", "ILOVEYOU", "RIGHT", "NO", "OKAY"]

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found for {label}, skipping...")
        continue

    data_list = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize and normalize
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        data_list.append(img)

    npy_path = os.path.join(OUTPUT_DIR, f"{label}.npy")
    np.save(npy_path, np.array(data_list))
    print(f"✅ Saved {len(data_list)} images to {npy_path}")

print("\n🎉 Conversion complete! All images saved as .npy files.")
