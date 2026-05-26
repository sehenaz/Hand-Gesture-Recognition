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
threshold = 0.5  # Lowered threshold for better detection

model = tf.keras.models.load_model("action_v2.keras")

cap = cv2.VideoCapture(0)

print("=" * 60)
print("HAND GESTURE RECOGNITION - DEBUG MODE")
print("=" * 60)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save current frame with prediction")
print("=" * 60)

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
        top_predictions = []

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(res)[-5:][::-1]
            top_predictions = [(actions[i], res[i] * 100) for i in top_indices]
            
            letter = actions[np.argmax(res)]
            confidence = res[np.argmax(res)] * 100
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

        # Display TOP 5 predictions on the right side
        if top_predictions:
            y_offset = 80
            cv2.rectangle(image, (440, 60), (635, 300), (0, 0, 0), -1)
            cv2.putText(image, "TOP 5 PREDICTIONS:",
                        (445, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
            
            for idx, (letter, conf) in enumerate(top_predictions):
                color = (0, 255, 0) if idx == 0 else (200, 200, 200)
                text = f"{idx+1}. {letter}: {conf:.1f}%"
                cv2.putText(image, text,
                            (450, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2 if idx == 0 else 1)
                y_offset += 40

        # Display main prediction at bottom
        if predicted_letter:
            cv2.rectangle(image, (0, 440), (640, 480), (0, 0, 0), -1)
            
            # Show predicted letter (larger)
            prediction_text = f"-> {predicted_letter}"
            cv2.putText(image, prediction_text,
                        (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)
            
            # Show accuracy
            accuracy_text = f"{confidence:.1f}%"
            cv2.putText(image, accuracy_text,
                        (150, 470),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

        # Show instructions
        cv2.putText(image, "Press 'S' to Save | 'Q' to Quit",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)

        cv2.imshow("Hand Gesture Recognition - Debug Mode", image)

        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
        
        if key == ord('s') and predicted_letter:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_FOLDER}/{predicted_letter}_{confidence:.1f}_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            print(f"✓ Saved: {filename}")
            print(f"  Predicted: {predicted_letter} ({confidence:.1f}%)")
            print(f"  Top 5: {', '.join([f'{l}({c:.1f}%)' for l, c in top_predictions])}")

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 60)
print("Session ended!")
print(f"Captured sentence: {''.join(sentence)}")
print("=" * 60)
