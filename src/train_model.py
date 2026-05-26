import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATA_PATH = "data"
actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for seq in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, seq, f"{frame}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
    LSTM(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))
model.save("model.keras", save_format="keras")
