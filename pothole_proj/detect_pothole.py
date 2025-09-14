from ultralytics import YOLO

# Load your trained YOLOv8 model (replace with your custom weights if available)
model = YOLO('yolov8n.pt')  # Or use 'runs/detect/train/weights/best.pt' if you have custom weights

# Run detection on your image
results = model('test_pothole.jpg')

# Print detection results (class, confidence, bounding box)
results.print()

# Show the image with detections
results.show()

# Save the result image with detections
results.save('detected_pothole.jpg')