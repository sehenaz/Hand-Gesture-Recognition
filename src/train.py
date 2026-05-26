import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = "dataset"
img_size = 64

data = []
labels = []

# FIXED label mapping
label_map = {label: idx for idx, label in enumerate(os.listdir(data_dir))}

for label in os.listdir(data_dir):
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(img)
        labels.append(label_map[label])

data = np.array(data) / 255.0
data = data.reshape(-1, img_size, img_size, 1)

labels = to_categorical(labels, num_classes=len(label_map))

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ✅ Improved Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# ✅ Save model
model.save("model.keras")

print("✅ Model Trained & Saved Successfully!")