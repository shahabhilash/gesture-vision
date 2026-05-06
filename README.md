<div align="center">
  <h1>🤖 Gesture Vision: Real-Time Sign Language Recognition</h1>
  <p><i>An advanced, sequence-based machine learning pipeline using MediaPipe, OpenCV, and Keras LSTMs.</i></p>
</div>

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technical Architecture](#-technical-architecture)
3. [Installation & Setup](#-installation--setup)
4. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
   - [Phase 1: Data Collection](#phase-1-data-collection)
   - [Phase 2: Data Preprocessing](#phase-2-data-preprocessing)
   - [Phase 3: Model Training](#phase-3-model-training)
   - [Phase 4: Real-Time Inference](#phase-4-real-time-inference)
5. [Understanding the Data Structure](#-understanding-the-data-structure)
6. [Adding Custom Gestures](#-adding-custom-gestures)
7. [Directory Map](#-directory-map)

---

## 🔍 Project Overview
**Gesture Vision** is a computer vision project designed to translate dynamic hand gestures into readable text in real-time. Unlike static image classifiers, this system analyzes a continuous **timeseries of frames**, allowing it to understand complex, motion-based sign language.

By leveraging Google's **MediaPipe Holistic** model, we extract 126 structural hand landmarks per frame. We then feed a rolling window of 30 frames into a custom **Long Short-Term Memory (LSTM)** neural network built with TensorFlow and Keras to predict the gesture sequence.

---

## ⚙️ Technical Architecture
- **Computer Vision:** `OpenCV` (Video capture, rendering) & `MediaPipe` (Landmark extraction)
- **Deep Learning:** `TensorFlow` & `Keras` (LSTM sequential modeling)
- **Data Processing:** `NumPy` & `scikit-learn` (Train/test splitting, array structuring)
- **Data Visualization:** `Matplotlib` & `Seaborn` (Confusion matrices, training history)

---

## 💻 Installation & Setup

Before running the codebase, ensure you have Python 3.8+ installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/shahabhilash/gesture-vision.git
cd gesture-vision
```

### 2. Create a Virtual Environment (Recommended)
**Windows:**
```bash
py -m venv venv
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Local Dependencies
To run the data collection locally on your machine, install the GUI-enabled packages:
```bash
py -m pip install --no-cache-dir opencv-python mediapipe==0.10.9 tensorflow numpy scikit-learn matplotlib seaborn
```

> **Note for Windows Users:** You must use the `py` command instead of `python` throughout this guide to prevent Windows from incorrectly opening the Microsoft Store.

---

## 🚀 Step-by-Step Execution Guide

This pipeline must be executed sequentially to generate the required datasets and models.

### Phase 1: Data Collection
The AI needs raw visual data to learn from. This script captures your webcam feed and extracts the hand keypoints. 
```bash
py collect.py
```
- **What it does:** Iterates through predefined gestures (`hello`, `thanks`, `yes`, `no`, etc.). For each gesture, it records 30 distinct sequences. Each sequence consists of 30 continuous frames.
- **Output:** Raw `.npy` files containing the keypoint arrays are saved into `data/raw/`.

### Phase 2: Data Preprocessing
Raw keypoint files must be structured into robust multi-dimensional arrays for the neural network.
```bash
py extract.py
```
- **What it does:** Scans the `data/raw/` folder, validates the shape of all arrays, and stacks them into one massive Features tensor (`X.npy`) and a Labels tensor (`y.npy`). It also dynamically maps your string labels to integers.
- **Output:** Saves `X.npy`, `y.npy`, and `label_map.npy` inside `data/processed/`.

### Phase 3: Model Training
We feed the processed arrays into a 3-layer LSTM neural network.
```bash
py src/train.py
```
- **What it does:** Splits the data (70% Train, 15% Validation, 15% Test). Trains the LSTM network for up to 150 epochs using Early Stopping to prevent overfitting.
- **Output:** 
  1. The compiled model weights are saved to `models/gesture_model.keras`.
  2. A Confusion Matrix and Accuracy/Loss graphs are exported to `plots/`.

### Phase 4: Real-Time Inference (Streamlit)
Deploy the trained model into a web application that works both locally and on the Cloud.
```bash
streamlit run src/app.py
```
* **What it does:** Starts a web server and opens a browser window. It utilizes `streamlit-webrtc` to securely pipe your webcam feed to the backend model for processing.


---

## 🧠 Understanding the Data Structure

For every single frame captured by the webcam:
1. MediaPipe detects the **Left Hand** and maps 21 3D landmarks (x, y, z). `21 * 3 = 63 values`
2. MediaPipe detects the **Right Hand** and maps 21 3D landmarks. `21 * 3 = 63 values`
3. We flatten these coordinates into a single array of **126 features per frame**.
4. The LSTM model is designed to accept an input shape of `(30, 126)`, meaning it looks at **30 consecutive frames** of these 126 features to make one prediction.

---

## 🔧 Adding Custom Gestures

Want to add your own signs? It takes less than 5 minutes!
1. Open `collect.py` and locate the `ACTIONS` array on Line 8.
2. Add your new gesture to the list: `ACTIONS = ['hello', 'thanks', 'your_new_gesture']`.
3. Re-run the entire pipeline from Phase 1 to Phase 4. The model will automatically adjust its output layers to accommodate the new classes!

---

## 🗺️ Directory Map
```text
gesture-vision/
├── data/                    
│   ├── raw/                 # Ignored by git; stores raw data collection
│   └── processed/           # Ignored by git; stores extracted training tensors
├── models/                  # Ignored by git; stores the trained .keras weights
├── plots/                   # Autogenerated training visualization graphs
├── src/
│   ├── app.py               # The main real-time inference loop and UI
│   ├── inference.py         # Rolling-buffer and prediction logic
│   ├── model.py             # Keras Sequential LSTM architecture
│   └── train.py             # Model training, callbacks, and evaluation logic
├── collect.py               # Data Collection engine
├── extract.py               # Feature processing engine
└── README.md                # This file
```
