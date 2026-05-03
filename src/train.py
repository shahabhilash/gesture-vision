import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model

PROCESSED_PATH = os.path.join('data', 'processed')
MODEL_PATH = os.path.join('models')
PLOTS_PATH = os.path.join('plots')

def plot_metrics(history):
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'loss.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, label_map):
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Sort labels to match indices
    labels = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PLOTS_PATH, 'confusion_matrix.png'))
    plt.close()

def main():
    # Load processed data
    print("Loading data...")
    try:
        X = np.load(os.path.join(PROCESSED_PATH, 'X.npy'))
        y = np.load(os.path.join(PROCESSED_PATH, 'y.npy'))
        label_map = np.load(os.path.join(PROCESSED_PATH, 'label_map.npy'), allow_pickle=True).item()
    except FileNotFoundError:
        print("Error: Processed data not found. Run extract.py first.")
        return
        
    num_classes = len(label_map)
    sequence_length = X.shape[1]
    features = X.shape[2]
    
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Classes: {num_classes}")
    
    # Train / Val / Test split (e.g., 70% train, 15% val, 15% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)
    
    print(f"Train split: X={X_train.shape}, y={y_train.shape}")
    print(f"Val split:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test split:  X={X_test.shape}, y={y_test.shape}")
    
    # Build model
    model = build_model(sequence_length=sequence_length, features=features, num_classes=num_classes)
    model.summary()
    
    # Setup Callbacks
    os.makedirs(MODEL_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_PATH, 'gesture_model.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        callbacks=callbacks,
        batch_size=32
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Predictions and Confusion Matrix
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("Generating plots...")
    plot_metrics(history)
    plot_confusion_matrix(y_test, y_pred, label_map)
    
    print("Training complete. Best model saved at:", checkpoint_path)

if __name__ == '__main__':
    main()
