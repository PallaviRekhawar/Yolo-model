from ultralytics import YOLO

# Load a pretrained classification model (will download if needed)
model = YOLO('yolov8n-cls.pt')  # This will download the correct classification model

# Train the model
model.train(data='classification/', epochs=3, imgsz=64, batch=8, task='classify')