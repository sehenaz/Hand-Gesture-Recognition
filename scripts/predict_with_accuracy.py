import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from utils import mediapipe_detection, extract_keypoints
import os
from datetime import datetime

# Create output folder for saved images
OUTPUT_FOLDER = "predictions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
sentence = []
sequence = []
threshold = 0.8

model = tf.keras.models.load_model("action_v2.keras")

cap = cv2.VideoCapture(0)

print("=" * 50)
print("HAND GESTURE RECOGNITION - LIVE PREDICTION")
print("=" * 50)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save current frame with prediction")
print("=" * 50)

with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        predicted_letter = ""
        confidence = 0.0

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            letter = actions[np.argmax(res)]
            confidence = res[np.argmax(res)] * 100  # Convert to percentage

            predicted_letter = letter

            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or letter != sentence[-1]:
                    sentence.append(letter)

        # Display sentence at top
        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, ''.join(sentence[-20:]),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # Display current prediction with accuracy at bottom
        if predicted_letter:
            # Background for prediction display
            cv2.rectangle(image, (0, 440), (640, 480), (0, 0, 0), -1)
            
            # Show predicted letter
            prediction_text = f"Prediction: {predicted_letter}"
            cv2.putText(image, prediction_text,
                        (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            
            # Show accuracy/confidence
            accuracy_text = f"Accuracy: {confidence:.1f}%"
            cv2.putText(image, accuracy_text,
                        (300, 465),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        # Show instructions
        cv2.putText(image, "Press 'S' to Save | 'Q' to Quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)

        cv2.imshow("Hand Gesture Recognition", image)

        key = cv2.waitKey(10) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 's' to save current frame
        if key == ord('s') and predicted_letter:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_FOLDER}/{predicted_letter}_{confidence:.1f}_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            print(f"✓ Saved: {filename} | Letter: {predicted_letter} | Accuracy: {confidence:.1f}%")

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 50)
print("Session ended!")
print(f"Captured sentence: {''.join(sentence)}")
print("=" * 50)
