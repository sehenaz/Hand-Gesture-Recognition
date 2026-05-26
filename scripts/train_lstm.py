"""
LSTM Model Training - Fixed for nested folder structure
data/LETTER/sequence_num/frame.npy
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data")
SAVE_PATH  = os.path.join(BASE_DIR, "src", "action_v2.keras")
SEQUENCE_LENGTH = 30
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("=" * 50)
print("   ASL LSTM TRAINING — Fixed Version")
print("=" * 50)

# ── Step 1: Load Data ──────────────────────────────────────────
X, y = [], []

for label in LABELS:
    label_folder = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_folder):
        print(f"  ⚠️  {label} nahi mila, skip")
        continue

    # Sirf folders lo (sequence numbers)
    seq_folders = sorted([
        f for f in os.listdir(label_folder)
        if os.path.isdir(os.path.join(label_folder, f))
    ], key=lambda x: int(x) if x.isdigit() else 0)

    loaded = 0
    for seq_num in seq_folders:
        seq_path = os.path.join(label_folder, seq_num)

        # Frames load karo (0.npy, 1.npy, ...)
        frame_files = sorted([
            f for f in os.listdir(seq_path)
            if f.endswith('.npy')
        ], key=lambda x: int(x.replace('.npy',''))
                              if x.replace('.npy','').isdigit() else 0)

        if len(frame_files) < SEQUENCE_LENGTH:
            continue  # Incomplete sequence skip

        frames = []
        ok = True
        for ff in frame_files[:SEQUENCE_LENGTH]:
            try:
                arr = np.load(os.path.join(seq_path, ff))
                frames.append(arr)
            except Exception as e:
                ok = False
                break

        if ok and len(frames) == SEQUENCE_LENGTH:
            X.append(frames)
            y.append(label)
            loaded += 1

    print(f"  ✅ {label}: {loaded} sequences loaded")

print(f"\n✅ Total : {len(X)} sequences")

if len(X) == 0:
    print("❌ Koi data nahi! Structure check karo.")
    exit()

X = np.array(X)
y = np.array(y)
print(f"✅ Shape : {X.shape}")   # (N, 30, features)

# ── Step 2: Encode & Split ─────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Step 3: Model ──────────────────────────────────────────────
n_features = X.shape[2]
n_classes  = len(le.classes_)
print(f"✅ Features per frame : {n_features}")
print(f"✅ Classes            : {n_classes}")

model = Sequential([
    LSTM(64,  return_sequences=True,
         input_shape=(SEQUENCE_LENGTH, n_features)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── Step 4: Train ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_PATH, save_best_only=True, verbose=1)
]

print("\n🚀 Training shuru...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ── Step 5: Evaluate ───────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"✅ Test Accuracy : {acc*100:.2f}%")
print(f"{'='*50}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(
    le.inverse_transform(y_true),
    le.inverse_transform(y_pred),
    target_names=le.classes_
))

# ── Step 6: Graphs ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss')
axes[1].legend()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[2],
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
graph_path = os.path.join(BASE_DIR, 'training_results.png')
plt.savefig(graph_path)
plt.show()

print(f"\n✅ Graph : {graph_path}")
print(f"✅ Model : {SAVE_PATH}")
print("🎉 Training complete!")