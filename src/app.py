import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import os
import numpy as np
import sys

# Ensure imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import GestureRecognizer

# Setup robust paths assuming the app is run from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'gesture_model.keras')
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'label_map.npy')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Recognizer
try:
    recognizer = GestureRecognizer(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH
    )
except Exception as e:
    print(f"Warning: Recognizer not initialized. {e}")
    recognizer = None

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def gen_frames():
    cap = cv2.VideoCapture(0)
    current_action = ""
    current_confidence = 0.0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            
            if recognizer is not None:
                # MediaPipe Processing
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # Draw Landmarks
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Prediction
                keypoints = extract_keypoints(results)
                action, confidence = recognizer.predict(keypoints, threshold=0.7)
                
                if action:
                    current_action = action
                    current_confidence = confidence
                    
                # Display Text Overlay
                display_text = f"Action: {current_action} ({current_confidence:.2f})" if current_action else "Collecting sequence..."
                cv2.rectangle(frame, (0, 0), (640, 40), (40, 40, 40), -1)
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 255), -1)
                cv2.putText(frame, "Model not loaded. Train model first.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app
    print("\nStarting Web Application...")
    print("Open http://localhost:5000 in your browser to view the application.")
    app.run(host='0.0.0.0', port=5000, debug=False)
