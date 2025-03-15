from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('yolo11n.pt')  # Automatically downloads if missing

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define trapezoidal restricted area (4 points: top-left, top-right, bottom-right, bottom-left)
trapezoid_pts = np.array([[250, 150], [400, 150], [450, 300], [200, 300]], np.int32)

def is_inside_trapezoid(box, trapezoid_pts):
    """Check if the center of a detected object is inside the trapezoidal area."""
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Calculate center of detected object
    
    # Use point-in-polygon check
    return cv2.pointPolygonTest(trapezoid_pts, (cx, cy), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()  # Draw bounding boxes

    # Draw trapezoidal restricted area
    cv2.polylines(annotated_frame, [trapezoid_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    isAlert = {'alert': [False, ""], 'personCount': 0}
    classInIntrusion = ['person', 'bicycle', 'car', 'motorcycle']

    # Loop through detected objects
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        class_id = int(cls.item())  # Convert to integer
        if class_id == 0:
            isAlert['personCount'] += 1

        if class_id in [0, 1, 2, 3]:  # Class ID 0 corresponds to "person" in COCO dataset
            if is_inside_trapezoid(box.tolist(), trapezoid_pts):
                isAlert['alert'] = [True, classInIntrusion[class_id]]
    
    cv2.imshow("Object Detection", annotated_frame)
    print(isAlert)
    
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
