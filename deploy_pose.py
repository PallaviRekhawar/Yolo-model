from ultralytics import YOLO
import cv2

print("=" * 50)
print("POSE ESTIMATION DEPLOYMENT - LOCALHOST")
print("=" * 50)

# Load pose estimation model
print("\nLoading pose model...")
model = YOLO('yolo11n-pose.pt')  # Pretrained pose model
print("✓ Model loaded successfully!")

# Open webcam
print("\nOpening webcam for pose detection...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("\n✓ Webcam ready! Showing pose estimation...")
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save screenshot")
print("\nDetecting poses...")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Run pose estimation
    results = model(frame, verbose=False)
    
    # Plot pose keypoints
    annotated_frame = results[0].plot()
    
    # Add info
    num_poses = len(results[0].keypoints) if results[0].keypoints is not None else 0
    cv2.putText(annotated_frame, f"Poses detected: {num_poses}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "YOLO Pose Estimation", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show keypoint count
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data
        if len(keypoints) > 0:
            num_keypoints = (keypoints[0] > 0).sum().item()
            cv2.putText(annotated_frame, f"Keypoints: {num_keypoints}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("YOLO Pose Estimation - Localhost Deployment", annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        cv2.imwrite(f"pose_screenshot_{frame_count}.jpg", annotated_frame)
        print(f"✓ Screenshot saved: pose_screenshot_{frame_count}.jpg")
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("\n✓ Pose deployment complete!")