"""
ASL Hand Gesture Recognition with Sentence Formation and Speech Output
FIXED VERSION - Uses MediaPipe Tasks API (compatible with mediapipe 0.10.x on Windows)
"""

import sys
import os
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime

# Force stdout flush so prints always show in terminal
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)


# ==================== MEDIAPIPE COMPATIBILITY LAYER ====================
# Replaces mediapipe.solutions.holistic (removed from Windows wheels in 0.10.x)
# Uses the new MediaPipe Tasks API with a drop-in compatible wrapper.

HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
])

HAND_LANDMARKER_MODEL = "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

class _Landmark:
    """Mimics mediapipe NormalizedLandmark."""
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

class _LandmarkList:
    """Mimics mediapipe NormalizedLandmarkList."""
    def __init__(self, landmarks):
        self.landmark = [_Landmark(lm.x, lm.y, lm.z) for lm in landmarks]

class _HolisticResults:
    """Mimics mediapipe Holistic results object (hands only)."""
    def __init__(self):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None

class HolisticCompat:
    """
    Drop-in replacement for mp.solutions.holistic.Holistic.
    Uses MediaPipe Tasks HandLandmarker internally.
    Supports: context manager, process(rgb_image) -> results
    """
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
        self._mp = mp
        return self

    def __exit__(self, *args):
        if self._landmarker:
            self._landmarker.close()

    def process(self, rgb_image):
        """rgb_image: numpy array in RGB format (writeable=False is fine)."""
        import mediapipe as mp
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image)
        )
        detection = self._landmarker.detect(mp_img)

        results = _HolisticResults()
        for i, handedness_list in enumerate(detection.handedness):
            label = handedness_list[0].category_name  # "Left" or "Right"
            lm_list = _LandmarkList(detection.hand_landmarks[i])
            if label == "Left":
                results.left_hand_landmarks = lm_list
            else:
                results.right_hand_landmarks = lm_list

        return results

    @staticmethod
    def _ensure_model():
        """Downloads hand_landmarker.task if not already present."""
        if not os.path.exists(HAND_LANDMARKER_MODEL):
            print(f"📥 Downloading MediaPipe hand model (~10 MB)...")
            import urllib.request
            try:
                urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_MODEL)
                print(f"✅ Model downloaded: {HAND_LANDMARKER_MODEL}")
            except Exception as e:
                raise RuntimeError(
                    f"❌ Could not download hand landmarker model!\n"
                    f"   Error: {e}\n"
                    f"   Manually download from:\n   {HAND_LANDMARKER_URL}\n"
                    f"   and place '{HAND_LANDMARKER_MODEL}' next to this script."
                )


def draw_landmarks_compat(frame, landmark_list, connections=None):
    """
    Drop-in replacement for mp_drawing.draw_landmarks().
    Draws hand landmarks and connections using OpenCV.
    """
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


# ── Try importing mediapipe (just to verify it is installed) ──────────────
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    MEDIAPIPE_ERROR = ""
except Exception as e:
    HAS_MEDIAPIPE = False
    MEDIAPIPE_ERROR = str(e)


# ==================== CONFIG ====================
class Config:
    MODEL_PATH = "action_v2.keras"
    OUTPUT_FOLDER = "captured_gestures"
    SEQUENCE_LENGTH = 30
    CONFIDENCE_THRESHOLD = 0.3
    STABILIZATION_FRAMES = 5
    COOLDOWN_SECONDS = 2


# ==================== UTIL ====================
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


# ==================== MAIN APP ====================
class ASLRecognitionApp:
    def __init__(self):
        print("=" * 60)
        print("   ASL HAND GESTURE RECOGNITION")
        print("=" * 60)

        # ── Create output folder ──────────────────────────────────
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

        # ── Check model file exists BEFORE loading ────────────────
        print(f"🔍 Looking for model: {Config.MODEL_PATH}")
        abs_path = os.path.abspath(Config.MODEL_PATH)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"\n❌ Model file nahi mila!\n"
                f"   Expected path: {abs_path}\n"
                f"   'action_v2.keras' ko script ke same folder mein rakho."
            )
        print(f"✅ Model file found: {abs_path}")

        # ── Load TensorFlow ────────────────────────────────────────
        print("📦 TensorFlow import ho raha hai...")
        import tensorflow as tf
        self.tf = tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        # ── Load model ────────────────────────────────────────────
        print("📦 Model load ho raha hai...")
        self.model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

        # ── Load MediaPipe ────────────────────────────────────────
        print("📦 MediaPipe load ho raha hai...")
        if not HAS_MEDIAPIPE:
            raise ImportError(
                f"❌ MediaPipe load nahi ho raha!\n"
                f"Error: {MEDIAPIPE_ERROR}\n"
                f"Run: pip install mediapipe==0.10.9"
            )
        print(f"✅ MediaPipe loaded (Version: {mp.__version__})")

        # ── Ensure hand landmarker model is ready ─────────────────
        HolisticCompat._ensure_model()

        # ── TTS (optional) ────────────────────────────────────────
        self.tts_engine = None
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            print("✅ Text-to-Speech ready")
        except Exception as e:
            print(f"⚠️  TTS skip kiya gaya (optional hai): {e}")

        # ── State variables ───────────────────────────────────────
        self.actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.sequence = []
        self.sentence = []
        self.prediction_history = deque(maxlen=Config.STABILIZATION_FRAMES)
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.last_spoken_time = 0

        print("=" * 60)
        print("🎯 Sab kuch ready hai! Camera start hoga...")
        print("   [Q] press karo exit ke liye")
        print("   [S] press karo sentence speak karne ke liye")
        print("   [C] press karo sentence clear karne ke liye")
        print("=" * 60)

    # ── Speak helper ──────────────────────────────────────────────
    def speak(self, text):
        if self.tts_engine and text.strip():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠️  TTS error: {e}")

    # ── Get stable prediction ─────────────────────────────────────
    def get_stable_prediction(self):
        if len(self.prediction_history) < Config.STABILIZATION_FRAMES:
            return None, 0.0
        from collections import Counter
        counts = Counter(self.prediction_history)
        best, freq = counts.most_common(1)[0]
        confidence = freq / Config.STABILIZATION_FRAMES
        return best, confidence

    # ── Draw UI on frame ──────────────────────────────────────────
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

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                raise RuntimeError(
                    "❌ Camera nahi mila!\n"
                    "   - Webcam connected hai?\n"
                    "   - Koi aur app use to nahi kar raha camera?"
                )

        print("✅ Camera started")

        # ── Uses HolisticCompat instead of mp_holistic.Holistic ──
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

                # MediaPipe detection (unchanged call signature)
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                # Build sequence
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-Config.SEQUENCE_LENGTH:]

                # Predict only when sequence is full
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
                    if (
                        stable_pred
                        and stable_conf >= 0.8
                        and (now - self.last_spoken_time) >= Config.COOLDOWN_SECONDS
                    ):
                        if not self.sentence or self.sentence[-1] != stable_pred:
                            self.sentence.append(stable_pred)
                            self.last_spoken_time = now
                            print(f"  ➕ Added: {stable_pred} | Sentence: {' '.join(self.sentence)}")

                # Draw UI
                frame = self.draw_ui(frame)

                # ── Draw landmarks using compat drawing function ──
                draw_landmarks_compat(frame, results.right_hand_landmarks, HAND_CONNECTIONS)
                draw_landmarks_compat(frame, results.left_hand_landmarks, HAND_CONNECTIONS)

                cv2.imshow("ASL Recognition", frame)

                key = cv2.waitKey(10) & 0xFF
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


# ==================== ENTRY POINT ====================
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
        print("\n⛔ User ne Ctrl+C press kiya. Bye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)