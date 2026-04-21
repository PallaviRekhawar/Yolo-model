from ultralytics import YOLO
import cv2
import torch

print("=" * 50)
print("OBJECT DETECTION DEPLOYMENT - LOCALHOST")
print("=" * 50)

# Load your trained model
print("\nLoading detection model...")
model = YOLO('runs/detect/train/weights/best.pt')  # Use your trained model
print("✓ Model loaded successfully!")

# Open webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    print("Trying alternative camera index...")
    cap = cv2.VideoCapture(1)  # Try second camera

if not cap.isOpened():
    print("❌ No webcam found! Using test image instead.")
    # Use a test image from dataset
    import os
    test_img = "datasets/COCO128/valid/images/" + os.listdir("datasets/COCO128/valid/images")[0]
    frame = cv2.imread(test_img)
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection - Test Image", annotated_frame)
    cv2.waitKey(0)
    exit()

print("✓ Webcam opened successfully!")
print("\nControls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save screenshot")
print("  - Press 'd' to toggle detection info")
print("\nStarting detection...")

show_details = True
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Get detection results
    annotated_frame = results[0].plot()
    
    # Add info text
    if show_details:
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Objects detected: {detections}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Model: YOLO Detection", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show FPS
    frame_count += 1
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display
    cv2.imshow("YOLO Object Detection - Localhost Deployment", annotated_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        cv2.imwrite(f"screenshot_{frame_count}.jpg", annotated_frame)
        print(f"✓ Screenshot saved: screenshot_{frame_count}.jpg")
    elif key == ord('d'):
        show_details = not show_details
        print(f"Details display: {'ON' if show_details else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✓ Deployment stopped")