import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import os
import sys

# Ensure imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import GestureRecognizer

# Setup robust paths relative to the root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'gesture_model.keras')
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'label_map.npy')

# Initialize MediaPipe globally
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# WebRTC STUN Server Configuration for Cloud Deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Gesture Vision", page_icon="🤖", layout="wide")
st.title("🤖 Gesture Vision: Real-Time Recognition")
st.markdown("Enable your webcam below to start predicting gestures using the trained LSTM model.")

class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # We initialize Holistic and Recognizer inside the processor 
        # so that each new web connection gets its own clean sequence buffer!
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.current_action = ""
        self.current_confidence = 0.0
        
        try:
            self.recognizer = GestureRecognizer(
                model_path=MODEL_PATH,
                label_map_path=LABEL_MAP_PATH
            )
        except Exception as e:
            self.recognizer = None
            print(f"Failed to load recognizer: {e}")

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        return np.concatenate([lh, rh])

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.recognizer is None:
            cv2.putText(img, "Model missing! Train first.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # MediaPipe Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        
        # Draw Landmarks
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Prediction
        keypoints = self.extract_keypoints(results)
        action, confidence = self.recognizer.predict(keypoints, threshold=0.7)
        
        if action:
            self.current_action = action
            self.current_confidence = confidence
            
        # Display Text Overlay
        display_text = f"Action: {self.current_action} ({self.current_confidence:.2f})" if self.current_action else "Collecting sequence..."
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Embed the WebRTC streamer in the Streamlit app
webrtc_streamer(
    key="gesture-recognition",
    video_processor_factory=GestureVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown("---")
st.markdown("Make sure you have collected data and trained the model before deploying.")
