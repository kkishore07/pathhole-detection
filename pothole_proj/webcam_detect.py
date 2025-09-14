from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run detection on the webcam (source=0), and display the results in a window
results = model.predict(source=0, show=True, conf=0.1)  # Lower confidence threshold