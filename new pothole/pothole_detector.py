import cv2
import numpy as np
from ultralytics import YOLO

# Hardcoded camera location (example: New York City)
CAMERA_LOCATION = {
    'latitude': 40.7128,
    'longitude': -74.0060
}

# Load YOLO model (using a pre-trained model for road damage/potholes)
# You can replace 'yolov8n.pt' with a custom or more specific model if available
MODEL_PATH = 'yolov8n.pt'  # Default YOLOv8 nano model

class PotholeDetector:
    def __init__(self, model_path=MODEL_PATH, camera_location=CAMERA_LOCATION):
        self.model = YOLO(model_path)
        self.camera_location = camera_location

    def detect_potholes(self, frame):
        # Run detection
        results = self.model(frame)
        return results

    def draw_detections(self, frame, results):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls] if hasattr(self.model, 'names') else str(cls)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label and confidence
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw location
                cv2.putText(frame, f"Lat: {self.camera_location['latitude']}, Lon: {self.camera_location['longitude']}",
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

def main():
    detector = PotholeDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect_potholes(frame)
        frame = detector.draw_detections(frame, results)
        cv2.imshow('Pothole Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 