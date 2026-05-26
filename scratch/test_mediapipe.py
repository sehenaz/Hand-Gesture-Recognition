import sys
print(f"Python version: {sys.version}")
try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
    import mediapipe.solutions.holistic as mp_holistic
    print("Successfully imported mediapipe.solutions.holistic")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
