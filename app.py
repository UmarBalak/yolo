from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')  # Automatically downloads if missing

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    results = model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()  # Draw bounding boxes
    
    cv2.imshow("", annotated_frame)
    
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
