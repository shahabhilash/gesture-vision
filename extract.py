import numpy as np
import os

# Configuration Paths
RAW_PATH = os.path.join('data', 'raw')
PROCESSED_PATH = os.path.join('data', 'processed')
SEQUENCE_LENGTH = 30
FEATURES = 126

def main():
    # Ensure processed directory is ready
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    if not os.path.exists(RAW_PATH):
        print(f"Error: {RAW_PATH} not found. Ensure you collect data first.")
        return

    # Find valid action folders within the raw directory
    actions = [d for d in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, d))]
    
    # Generate label-to-integer mapping
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    
    print(f"Discovered actions: {actions}")
    print(f"Generated Label Map: {label_map}")
    
    # Process npy files and structure them for ML models
    for action in actions:
        action_path = os.path.join(RAW_PATH, action)
        # Sort files simply to ensure consistent reading (optional but good practice)
        files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        
        for file in files:
            # Load the individual sequence array (shape should be 30, 126)
            res = np.load(os.path.join(action_path, file))
            
            # Formally check shape to prevent corrupt/short sequences making it in
            if res.shape == (SEQUENCE_LENGTH, FEATURES):
                sequences.append(res)
                labels.append(label_map[action])
            else:
                print(f"Warning: Skipping {file} in {action} (bad shape {res.shape})")

    if not sequences:
        print("No valid sequences were processed.")
        return

    # Stack 30 frames per sequence into the final 3D numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    # Save processed resources
    np.save(os.path.join(PROCESSED_PATH, 'X.npy'), X)
    np.save(os.path.join(PROCESSED_PATH, 'y.npy'), y)
    np.save(os.path.join(PROCESSED_PATH, 'label_map.npy'), label_map)
    
    print(f"\nExtraction complete.")
    print(f"X shape: {X.shape} (samples, frames, keypoints)")
    print(f"y shape: {y.shape} (samples,)")
    print(f"Files saved in: {PROCESSED_PATH}")

if __name__ == '__main__':
    main()
