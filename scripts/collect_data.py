import cv2
import os

label = input("Enter letter (A-Z): ").upper()
save_dir = f"dataset/{label}"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press C to capture image, Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"Letter: {label}  Count: {count}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        print("Captured", count)
        count += 1

    if key == ord('q') or count == 30:
        break

cap.release()
cv2.destroyAllWindows()
