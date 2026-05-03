import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(sequence_length=30, features=126, num_classes=7):
    """
    Builds a sequence-based LSTM model for gesture recognition.
    """
    model = Sequential()
    
    # LSTM Layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, features)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    
    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
