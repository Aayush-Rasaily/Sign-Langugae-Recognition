This project aims to recognize hand gestures of sign language using deep learning techniques. It captures hand gestures via webcam and classifies them into predefined sign language categories such as letters, numbers, or words.

The project uses Computer Vision + CNN/Conv2D Model to train a model and real-time gesture detection to make predictions.

📁 Folder Structure
SignLanguageRecognition/
│── MP_DATA/              # Dataset of sign language images (generated using webcam)
│── model/                # Trained model (h5 file)
│── README.md             # Project documentation (this file)
│── collect_data.py       # To capture and save hand gestures
│── train_model.py        # Train CNN model
│── real_time_detection.py# Real-time sign prediction using webcam
│── requirements.txt      # Dependencies list

🧠 Tech Stack
Component	Technology
Language	Python
Deep Learning	TensorFlow / Keras
Data Handling	NumPy, Pandas
Image Processing	OpenCV
Model Type	CNN
IDE	VS Code / Jupyter Notebook
🚀 How to Run the Project
1️⃣ Create & Activate Virtual Environment
conda create -n signlang python=3.9
conda activate signlang

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Collect Sign Language Data

Run the following script and show gestures in front of webcam:

python collect_data.py

4️⃣ Train the Model
python train_model.py

5️⃣ Real-time Gesture Detection
python real_time_detection.py

📸 Dataset Collection Process

Webcam is used to capture hand gestures.

Background is removed using MediaPipe / CV Zone.

Images are stored automatically in folders like:
MP_DATA/A/1.jpg, MP_DATA/A/2.jpg, etc.

Each folder represents one class/category.

📊 Model Accuracy & Evaluation

Train/Test Split used: 80% / 20%

Metrics used:

Accuracy

Loss

Validation Accuracy

You can visualize training results using:

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

📦 requirements.txt

Add this file:

opencv-python
numpy
tensorflow
mediapipe
cvzone
matplotlib

📈 Future Improvements

✔ Add more sign language classes (full ASL/ISL vocabulary)
✔ Use LSTM for gesture sequence recognition
✔ Deploy using Streamlit / Flask as a Web App
✔ Convert model to TensorFlow Lite for Mobile

📜 License

This project is licensed under the MIT License – you are free to use and modify it.
