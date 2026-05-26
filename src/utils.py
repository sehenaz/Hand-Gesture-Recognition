# import cv2
# import numpy as np
# import mediapipe as mp

# mp_holistic = mp.solutions.holistic

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] 
#                      for res in results.pose_landmarks.landmark]).flatten() \
#                      if results.pose_landmarks else np.zeros(33*4)

#     face = np.array([[res.x, res.y, res.z] 
#                      for res in results.face_landmarks.landmark]).flatten() \
#                      if results.face_landmarks else np.zeros(468*3)

#     lh = np.array([[res.x, res.y, res.z] 
#                    for res in results.left_hand_landmarks.landmark]).flatten() \
#                    if results.left_hand_landmarks else np.zeros(21*3)

#     rh = np.array([[res.x, res.y, res.z] 
#                    for res in results.right_hand_landmarks.landmark]).flatten() \
#                    if results.right_hand_landmarks else np.zeros(21*3)

#     return np.concatenate([pose, face, lh, rh])



import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])
