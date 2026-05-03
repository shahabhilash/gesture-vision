import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import deque

class GestureRecognizer:
    def __init__(self, model_path='models/gesture_model.keras', label_map_path='data/processed/label_map.npy', sequence_length=30):
        self.sequence_length = sequence_length
        self.sequence = deque(maxlen=sequence_length)
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found at {label_map_path}. Please run extract.py first.")
            
        self.model = load_model(model_path)
        
        # Load label map and reverse it for integer -> string prediction
        label_map = np.load(label_map_path, allow_pickle=True).item()
        self.actions = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
        
    def predict(self, keypoints, threshold=0.7):
        """
        Adds new keypoints to the sequence and predicts the gesture if the sequence is full.
        Returns: (predicted_action, confidence_score) or (None, 0.0)
        """
        self.sequence.append(keypoints)
        
        if len(self.sequence) == self.sequence_length:
            # Prepare input array for model: shape (1, 30, 126)
            input_data = np.expand_dims(self.sequence, axis=0)
            
            res = self.model.predict(input_data, verbose=0)[0]
            action_idx = np.argmax(res)
            confidence = res[action_idx]
            
            if confidence > threshold:
                return self.actions[action_idx], confidence
                
        return None, 0.0
