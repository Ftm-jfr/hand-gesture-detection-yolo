from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

# Video source
# Option 1: phone camera over Wi-Fi (e.g. IP Webcam app); set phone's address
cap = cv2.VideoCapture("http://<phone-ip>:8080/video")

# Option 2: laptop webcam;
# cap = cv2.VideoCapture(0)

# Option 3: video file;
# cap = cv2.VideoCapture("path/to/video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0])
        label = results.names[cls]

        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Hand Gesture Detection - Live", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()