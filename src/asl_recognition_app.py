"""
ASL Hand Gesture Recognition
FIXED VERSION
"""

import sys
import os
import time
import cv2
import numpy as np
from collections import deque

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
])

HAND_LANDMARKER_MODEL = "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

class _Landmark:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

class _LandmarkList:
    def __init__(self, landmarks):
        self.landmark = [_Landmark(lm.x, lm.y, lm.z) for lm in landmarks]

class _HolisticResults:
    def __init__(self):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None

class HolisticCompat:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._det_conf = min_detection_confidence
        self._track_conf = min_tracking_confidence
        self._landmarker = None

    def __enter__(self):
        self._ensure_model()
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=HAND_LANDMARKER_MODEL
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self._det_conf,
            min_hand_presence_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, *args):
        if self._landmarker:
            self._landmarker.close()

    def process(self, rgb_image):
        import mediapipe as mp
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image)
        )
        detection = self._landmarker.detect(mp_img)
        results = _HolisticResults()
        for i, handedness_list in enumerate(detection.handedness):
            label = handedness_list[0].category_name
            lm_list = _LandmarkList(detection.hand_landmarks[i])
            if label == "Left":
                results.left_hand_landmarks = lm_list
            else:
                results.right_hand_landmarks = lm_list
        return results

    @staticmethod
    def _ensure_model():
        if not os.path.exists(HAND_LANDMARKER_MODEL):
            print("📥 Downloading MediaPipe hand model...")
            import urllib.request
            try:
                urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_MODEL)
                print(f"✅ Model downloaded: {HAND_LANDMARKER_MODEL}")
            except Exception as e:
                raise RuntimeError(f"❌ Download failed: {e}")


def draw_landmarks_compat(frame, landmark_list, connections=None):
    if landmark_list is None:
        return
    h, w = frame.shape[:2]
    pts = []
    for lm in landmark_list.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, (0, 255, 128), -1)
    if connections:
        for start_idx, end_idx in connections:
            cv2.line(frame, pts[start_idx], pts[end_idx], (0, 200, 100), 2)


try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    MEDIAPIPE_ERROR = ""
except Exception as e:
    HAS_MEDIAPIPE = False
    MEDIAPIPE_ERROR = str(e)


class Config:
    MODEL_PATH = "action_v2.keras"
    OUTPUT_FOLDER = "captured_gestures"
    SEQUENCE_LENGTH = 30
    CONFIDENCE_THRESHOLD = 0.3
    STABILIZATION_FRAMES = 5
    COOLDOWN_SECONDS = 2


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    return np.concatenate([lh, rh])


class ASLRecognitionApp:
    def __init__(self):
        print("=" * 60)
        print("   ASL HAND GESTURE RECOGNITION")
        print("=" * 60)

        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

        print(f"🔍 Looking for model: {Config.MODEL_PATH}")
        abs_path = os.path.abspath(Config.MODEL_PATH)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"\n❌ Model file nahi mila!\n"
                f"   Expected path: {abs_path}\n"
            )
        print(f"✅ Model file found: {abs_path}")

        print("📦 TensorFlow import ho raha hai...")
        import tensorflow as tf
        self.tf = tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        print("📦 Model load ho raha hai...")
        self.model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

        print("📦 MediaPipe load ho raha hai...")
        if not HAS_MEDIAPIPE:
            raise ImportError(f"❌ MediaPipe error: {MEDIAPIPE_ERROR}")
        print(f"✅ MediaPipe loaded (Version: {mp.__version__})")

        HolisticCompat._ensure_model()

        self.tts_engine = None
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            print("✅ Text-to-Speech ready")
        except Exception as e:
            print(f"⚠️  TTS skip: {e}")

        self.actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.sequence = []
        self.sentence = []
        self.prediction_history = deque(maxlen=Config.STABILIZATION_FRAMES)
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.last_spoken_time = 0

        print("=" * 60)
        print("🎯 Camera start hoga...")
        print("   [Q] Quit  [S] Speak  [C] Clear")
        print("=" * 60)

    def speak(self, text):
        if self.tts_engine and text.strip():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠️  TTS error: {e}")

    def get_stable_prediction(self):
        if len(self.prediction_history) < Config.STABILIZATION_FRAMES:
            return None, 0.0
        from collections import Counter
        counts = Counter(self.prediction_history)
        best, freq = counts.most_common(1)[0]
        return best, freq / Config.STABILIZATION_FRAMES

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
        pred_text = f"{self.current_prediction}  {self.current_confidence:.1f}%"
        cv2.putText(frame, pred_text, (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 2)
        sentence_str = " ".join(self.sentence)
        cv2.putText(frame, f"Sentence: {sentence_str}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Q=Quit  S=Speak  C=Clear", (w - 320, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return frame

    def run(self):
        # ── Camera open with CAP_DSHOW (Windows fix) ──
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(2)

        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("❌ Camera nahi mila!")

        print("✅ Camera started")

        with HolisticCompat(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Frame nahi aaya, retry...")
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                self.sequence.append(keypoints)
                self.sequence = self.sequence[-Config.SEQUENCE_LENGTH:]

                if len(self.sequence) == Config.SEQUENCE_LENGTH:
                    input_data = np.expand_dims(self.sequence, axis=0)
                    res = self.model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(res)
                    confidence = float(res[idx])

                    if confidence >= Config.CONFIDENCE_THRESHOLD:
                        self.current_prediction = self.actions[idx]
                        self.current_confidence = confidence * 100
                        self.prediction_history.append(self.actions[idx])
                    else:
                        self.current_prediction = "?"
                        self.current_confidence = confidence * 100

                    stable_pred, stable_conf = self.get_stable_prediction()
                    now = time.time()
                    if (stable_pred and stable_conf >= 0.8
                            and (now - self.last_spoken_time) >= Config.COOLDOWN_SECONDS):
                        if not self.sentence or self.sentence[-1] != stable_pred:
                            self.sentence.append(stable_pred)
                            self.last_spoken_time = now
                            print(f"  ➕ Added: {stable_pred} | Sentence: {' '.join(self.sentence)}")

                frame = self.draw_ui(frame)
                draw_landmarks_compat(frame, results.right_hand_landmarks, HAND_CONNECTIONS)
                draw_landmarks_compat(frame, results.left_hand_landmarks, HAND_CONNECTIONS)
                cv2.imshow("ASL Recognition", frame)
                cv2.moveWindow("ASL Recognition", 0, 0)  # Window screen pe lao

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("👋 Exiting...")
                    break
                elif key == ord('s'):
                    sentence_str = " ".join(self.sentence)
                    print(f"🔊 Speaking: {sentence_str}")
                    self.speak(sentence_str)
                elif key == ord('c'):
                    self.sentence = []
                    print("🗑️  Sentence cleared")

        cap.release()
        cv2.destroyAllWindows()
        print("✅ App closed cleanly")


if __name__ == "__main__":
    print("🚀 Starting ASL Recognition App...")
    print(f"   Python: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")

    try:
        app = ASLRecognitionApp()
        app.run()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⛔ Bye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)