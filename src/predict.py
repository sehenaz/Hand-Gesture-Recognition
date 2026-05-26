import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from utils import mediapipe_detection, extract_keypoints

actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
sentence = []
sequence = []
threshold = 0.8

model = tf.keras.models.load_model("action_v2.h5")

cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            letter = actions[np.argmax(res)]

            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or letter != sentence[-1]:
                    sentence.append(letter)

        cv2.rectangle(image, (0,0), (640,40), (0,0,0), -1)
        cv2.putText(image, ''.join(sentence[-20:]),
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

        cv2.imshow("Hand Gesture Sentence", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
