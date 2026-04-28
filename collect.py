import cv2
import numpy as np
import os
import mediapipe as mp

# Configuration
DATA_PATH = os.path.join('data', 'raw')
ACTIONS = ['hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'help']
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

def extract_keypoints(results):
    """Extracts left and right hand keypoints (63 each = 126 total)."""
    # 21 landmarks * 3 coordinates (x,y,z) = 63 values per hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def main():
    # Ensure raw directories exist
    for action in ACTIONS:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Auto-detect default webcam
    cap = cv2.VideoCapture(0)

    # Start MediaPipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                # 2-second countdown before each sequence starts
                for i in range(2, 0, -1):
                    ret, frame = cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1) # Mirror display
                    
                    # Show countdown on screen
                    cv2.putText(frame, f'Get ready for "{action}" ({sequence + 1}/{NO_SEQUENCES}) in {i}...', 
                                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Webcam Data Collection', frame)
                    cv2.waitKey(1000) # Wait 1 second (1000 ms) per countdown tick
                
                sequence_data = []

                # Collect the 30 frames for the current sequence
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: break

                    frame = cv2.flip(frame, 1)
                    
                    # Convert to RGB for mediapipe processing
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    # Draw landmarks on the screen for visual feedback
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    # Show current target label and sequence progress
                    cv2.putText(frame, f'Recording "{action}" | Seq {sequence + 1}/{NO_SEQUENCES}', 
                                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Webcam Data Collection', frame)

                    # Extract the 126 keypoints using our utility function
                    keypoints = extract_keypoints(results)
                    sequence_data.append(keypoints)

                    # Allow quitting safely if 'q' is pressed
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\nCollection aborted.")
                        return
                        
                # Save the keypoints for the current sequence
                # File path: data/raw/LABEL/seq_number.npy
                npy_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
                np.save(npy_path, sequence_data)
                
    print("\nData collection finished successfully.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
