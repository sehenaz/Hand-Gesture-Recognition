import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data"
actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print("=" * 60)
print("LOADING YOUR HAND GESTURE DATA...")
print("=" * 60)

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"⚠️  WARNING: No data found for letter {action}")
        continue
    
    seq_count = 0
    for seq in os.listdir(action_path):
        try:
            window = []
            for frame in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, seq, f"{frame}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
            seq_count += 1
        except Exception as e:
            print(f"⚠️  Error loading {action}/{seq}: {e}")
    
    print(f"✓ Loaded {seq_count} sequences for letter '{action}'")

print(f"\n📊 Total sequences loaded: {len(sequences)}")
print(f"📊 Total letters: {len(set(labels))}")

X = np.array(sequences)
y = to_categorical(labels)

# Split data: 95% training, 5% validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(f"\n🎯 Training samples: {len(X_train)}")
print(f"🎯 Validation samples: {len(X_test)}")

# Build improved model with dropout for better generalization
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 126)),
    Dropout(0.2),
    LSTM(256, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(128, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n" + "=" * 60)
print("TRAINING MODEL ON YOUR DATA...")
print("=" * 60)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train with more epochs
history = model.fit(X_train, y_train, 
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=1)

# Save the model in the modern .keras format
model.save("action_v2.keras")

print("\n" + "=" * 60)
print("✅ MODEL TRAINED SUCCESSFULLY!")
print("=" * 60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print("Model saved as: action_v2.keras")
print("=" * 60)
