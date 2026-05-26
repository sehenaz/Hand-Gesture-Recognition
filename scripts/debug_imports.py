import os
import sys
import numpy as np

print("--- DEBUG INFORMATION ---")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
print("\n--- CHECKING LIBRARIES ---")

try:
    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")
    print(f"   Path: {numpy.__file__}")
except ImportError:
    print("❌ NumPy NOT FOUND")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"   Path: {tf.__file__}")
except Exception as e:
    print(f"❌ TensorFlow ERROR: {e}")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe version: {mp.__version__}")
    print(f"   Path: {mp.__file__}")
    
    print("\n--- TESTING SOLUTIONS ---")
    try:
        from mediapipe.solutions import holistic
        print("✅ Success: from mediapipe.solutions import holistic")
    except Exception as e:
        print(f"❌ Error importing solutions: {e}")
        
    print("\n--- DIRECTORY CONTENT (mediapipe) ---")
    mp_path = os.path.dirname(mp.__file__)
    print(f"Searching in: {mp_path}")
    if os.path.exists(mp_path):
        print(f"Contents: {os.listdir(mp_path)}")
        sol_path = os.path.join(mp_path, 'solutions')
        if os.path.exists(sol_path):
            print(f"Solutions directory exists! Contents: {os.listdir(sol_path)}")
        else:
            print("❌ Solutions directory DOES NOT EXIST inside mediapipe folder")
except Exception as e:
    print(f"❌ MediaPipe ERROR: {e}")
