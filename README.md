# Gesture Vision

A sequence-based Sign Language Recognition application built with MediaPipe, OpenCV, TensorFlow/Keras, and Flask.

## Features
- Collects continuous streams of webcam data and extracts 126 keypoints per frame (63 per hand).
- Trains a Sequence-Based LSTM Model on the collected keypoint timeseries data.
- Deploys as a local Flask Web Application for real-time inference in the browser.

---

## 🚀 How to Run the Project (End-to-End)

Follow these steps in order to train and deploy the model:

### 1. Collect Data
First, run the data collection script. It will open your webcam and guide you through collecting 30 sequences of 30 frames each for various gestures.
```bash
python collect.py
```

### 2. Extract and Process
Once the raw `.npy` files are collected, run the extraction script. This maps string labels to integers and builds the final 3D arrays (`X.npy` and `y.npy`) for training.
```bash
python extract.py
```

### 3. Train the Model
Train the LSTM model on the processed dataset. This will split your data into Train/Val/Test sets, evaluate the model, generate accuracy/loss plots, and save the final `gesture_model.keras` weights.
```bash
python src/train.py
```

### 4. Deploy the Application
Finally, start the Flask web server to see the live gesture predictions in your web browser.
```bash
python src/app.py
```
After running the command, open **http://localhost:5000** in your web browser.

---

## Directory Structure
- `data/` - Holds raw sequences and the processed datasets.
- `models/` - Contains the saved keras models after training.
- `plots/` - Contains the confusion matrix and loss/accuracy plots.
- `src/` - The core application source files (`app.py`, `inference.py`, `model.py`, `train.py`).
- `templates/` - The HTML files for the web interface.
