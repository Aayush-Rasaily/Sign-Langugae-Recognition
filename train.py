import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

# ----------------------------
# SETTINGS
# ----------------------------
DATA_DIR = "MP_DATA"   # <<--- YOUR ORIGINAL DATASET
IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32

# ----------------------------
# LOAD DATA
# ----------------------------
data_list = []
label_list = []
classes = sorted([f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.npy')])

for idx, class_name in enumerate(classes):
    file_path = os.path.join(DATA_DIR, f"{class_name}.npy")
    imgs = np.load(file_path)

    # 🔥 FIX: Skip broken or incomplete files
    if len(imgs.shape) != 4:
        print(f"⚠ Skipping file: {file_path} | Bad shape: {imgs.shape}")
        continue

    data_list.append(imgs)
    label_list.append(np.full((len(imgs),), idx))

data = np.concatenate(data_list, axis=0).astype("float32")
labels = np.concatenate(label_list, axis=0)
labels = to_categorical(labels, num_classes=len(classes))

print("✅ Data Loaded")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True
)

# ----------------------------
# MODEL
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ----------------------------
# EVALUATE MODEL
# ----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Final Test Loss: {test_loss:.4f}")

# ----------------------------
# SAVE MODEL + LABELS
# ----------------------------
model.save("my_sign_model.h5")
with open("labels.json", "w") as f:
    json.dump(classes, f)

print("\n✅ Training done. Model saved as 'my_sign_model.h5' and 'labels.json'")

# ----------------------------
# PLOT ACCURACY & LOSS
# ----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
