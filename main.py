from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
import json
import os

class AccidentDetectionSystem:
    def __init__(self, model_path="yolov8n.pt", camera_id=0, iou_threshold=0.2, 
                 persistence_frames=5, save_path="accidents"):
        """
        Initialize the accident detection system
        
        Args:
            model_path: Path to the YOLO model
            camera_id: Camera device ID
            iou_threshold: Initial IoU threshold for accident detection
            persistence_frames: Number of consecutive frames needed to confirm an accident
            save_path: Directory to save accident footage
        """
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Detection parameters
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.persistence_frames = persistence_frames
        self.save_path = save_path
        
        # Create directory for saving accident footage if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Detection state variables
        self.accident_frames_count = 0
        self.accident_in_progress = False
        self.accident_timestamp = None
        self.recording = False
        self.video_writer = None
        
        # Dynamic threshold parameters
        self.min_threshold = 0.1
        self.max_threshold = 0.5
        self.threshold_adjust_rate = 0.01
        self.false_positive_count = 0
        self.false_negative_count = 0
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # For accident verification after detection
        self.buffer_frames = []
        self.max_buffer_size = 30  # About 1 second at 30 FPS
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_accidents': 0,
            'confirmed_accidents': 0,
            'false_alarms': 0
        }
        
        # Traffic density metrics for dynamic threshold
        self.traffic_density = 0  # 0-1 scale
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    
    def adjust_threshold_dynamically(self):
        """
        Adjust the IoU threshold based on traffic density and detection history
        """
        # Base adjustment on traffic density
        if self.traffic_density > 0.7:  # High traffic
            target_threshold = max(self.min_threshold, 
                                  self.iou_threshold - self.threshold_adjust_rate)
        elif self.traffic_density < 0.3:  # Low traffic
            target_threshold = min(self.max_threshold, 
                                  self.iou_threshold + self.threshold_adjust_rate)
        else:  # Medium traffic, make smaller adjustments
            if self.false_positive_count > self.false_negative_count:
                target_threshold = min(self.max_threshold, 
                                      self.iou_threshold + (self.threshold_adjust_rate/2))
            else:
                target_threshold = max(self.min_threshold, 
                                      self.iou_threshold - (self.threshold_adjust_rate/2))
        
        # Gradually move toward target
        if abs(target_threshold - self.iou_threshold) > self.threshold_adjust_rate:
            if target_threshold > self.iou_threshold:
                self.iou_threshold += self.threshold_adjust_rate
            else:
                self.iou_threshold -= self.threshold_adjust_rate
        else:
            self.iou_threshold = target_threshold
            
        # Reset counters periodically
        if self.stats['total_frames'] % 300 == 0:  # Reset every ~10 seconds at 30 FPS
            self.false_positive_count = 0
            self.false_negative_count = 0
    
    def update_traffic_density(self, vehicle_count, frame_area):
        """
        Update traffic density metric based on number of vehicles and frame size
        """
        # Simple metric: normalize by a max expected vehicle count
        max_expected_vehicles = 15  # Adjust based on your specific scenario
        raw_density = min(1.0, vehicle_count / max_expected_vehicles)
        
        # Smooth changes in density with exponential moving average
        alpha = 0.1  # Smoothing factor
        self.traffic_density = alpha * raw_density + (1 - alpha) * self.traffic_density
    
    def detect_accident(self, frame):
        """
        Detect potential accidents in a frame
        """
        # Increment frame counter
        self.stats['total_frames'] += 1
        
        # Add frame to buffer
        self.buffer_frames.append(frame.copy())
        if len(self.buffer_frames) > self.max_buffer_size:
            self.buffer_frames.pop(0)
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Extract vehicle detections
        vehicles = []
        for r in results:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls_id) in self.vehicle_classes and float(conf) > 0.4:
                    vehicles.append(box.cpu().numpy())
        
        # Update traffic density
        self.update_traffic_density(len(vehicles), frame_area)
        
        # Check for collisions (overlapping bounding boxes)
        accident_detected = False
        overlapping_boxes = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                iou = self.calculate_iou(vehicles[i], vehicles[j])
                
                if iou > self.iou_threshold:
                    accident_detected = True
                    overlapping_boxes.extend([vehicles[i], vehicles[j]])
        
        # Process accident detection
        if accident_detected:
            self.accident_frames_count += 1
            
            # Confirm accident after persistence threshold
            if self.accident_frames_count >= self.persistence_frames and not self.accident_in_progress:
                self.stats['detected_accidents'] += 1
                self.accident_in_progress = True
                self.accident_timestamp = time.time()
                print(f"⚠️ ACCIDENT DETECTED at {time.strftime('%H:%M:%S')}")
                print(f"Current IoU threshold: {self.iou_threshold:.2f}")
                
                # Start recording
                self.start_recording(frame)
                
            # Draw all involved vehicles with red boxes
            for box in set(tuple(map(tuple, box)) for box in overlapping_boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            # Add accident warning text
            cv2.putText(frame, "ACCIDENT DETECTED", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Reset accident counter if no detection
            if self.accident_frames_count > 0:
                self.accident_frames_count -= 1
            
            # End accident state if enough frames without detection
            if self.accident_in_progress and self.accident_frames_count == 0:
                duration = time.time() - self.accident_timestamp
                print(f"Accident event ended. Duration: {duration:.1f} seconds")
                self.accident_in_progress = False
                
                # Stop recording
                if self.recording:
                    self.stop_recording()
        
        # Draw all vehicle detections
        for box in vehicles:
            x1, y1, x2, y2 = map(int, box)
            # Use green for regular detection
            color = (0, 255, 0) if tuple(box) not in map(tuple, overlapping_boxes) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Adjust threshold dynamically every 30 frames (~1 second at 30fps)
        if self.stats['total_frames'] % 30 == 0:
            self.adjust_threshold_dynamically()
        
        # Display system info on frame
        cv2.putText(frame, f"IoU Threshold: {self.iou_threshold:.2f}", (10, height - 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Traffic Density: {self.traffic_density:.2f}", (10, height - 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Accidents: {self.stats['confirmed_accidents']}", (10, height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, accident_detected
    
    def start_recording(self, frame):
        """Start recording video when accident is detected"""
        if not self.recording:
            height, width = frame.shape[:2]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"{self.save_path}/accident_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            # First, write the buffer frames to capture pre-accident footage
            for buffer_frame in self.buffer_frames:
                self.video_writer.write(buffer_frame)
            
            self.recording = True
            print(f"Recording started: {output_path}")
    
    def stop_recording(self):
        """Stop recording video"""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.recording = False
            self.video_writer = None
            print("Recording stopped")
    
    def record_frame(self, frame):
        """Record frame if recording is active"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def run(self):
        """Run the accident detection system"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("🚗 Car Accident Detection System Started")
        print(f"Initial IoU threshold: {self.iou_threshold}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame for accident detection
                processed_frame, accident = self.detect_accident(frame)
                
                # Record if in recording mode
                self.record_frame(processed_frame)
                
                # Display the resulting frame
                cv2.imshow("Car Accident Detection", processed_frame)
                
                # Break the loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            if self.recording:
                self.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            print("System stopped")
            
            # Print final statistics
            print("\nSystem Statistics:")
            print(f"Total frames processed: {self.stats['total_frames']}")
            print(f"Detected accidents: {self.stats['detected_accidents']}")
            print(f"Confirmed accidents: {self.stats['confirmed_accidents']}")
            print(f"False alarms: {self.stats['false_alarms']}")
            print(f"Final IoU threshold: {self.iou_threshold:.2f}")

# Example usage
if __name__ == "__main__":
    # Initialize the accident detection system with parameters
    detector = AccidentDetectionSystem(
        model_path="yolov8n.pt",  # Use small model for speed
        camera_id=0,              # Default camera
        iou_threshold=0.2,        # Initial threshold
        persistence_frames=5,     # Need 5 consecutive frames to confirm accident
        save_path="accident_videos"
    )
    
    # Start the detection system
    detector.run()