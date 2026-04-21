from ultralytics import YOLO
import cv2
import numpy as np
import os

print("=" * 60)
print("MULTI-TASK VISION SYSTEM - DEPLOYMENT (Using Pretrained Models)")
print("=" * 60)

# Check if we can use webcam
use_webcam = False
cap = cv2.VideoCapture(0)
if cap.isOpened():
    use_webcam = True
    print("\n✓ Webcam detected!")
    cap.release()
else:
    print("\n⚠ No webcam detected. Using sample images instead.")

# Load all three models
print("\n[1/3] Loading Object Detection Model...")
detect_model = YOLO('yolo11n.pt')
print("✓ Detection model ready")

print("\n[2/3] Loading Classification Model...")
cls_model = YOLO('yolo11n-cls.pt')
print("✓ Classification model ready")

print("\n[3/3] Loading Pose Estimation Model...")
pose_model = YOLO('yolo11n-pose.pt')
print("✓ Pose model ready")

print("\n" + "=" * 60)
print("DEPLOYMENT MENU")
print("=" * 60)
print("1. Object Detection (Bounding boxes)")
print("2. Image Classification (Labels)")
print("3. Pose Estimation (Skeleton keypoints)")
print("4. ALL THREE sequentially")
print("5. Test on sample images (No webcam needed)")

choice = input("\nEnter your choice (1-5): ")

def run_webcam_detection(model, mode):
    """Run webcam detection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print(f"\n✓ Starting {mode} - Press 'q' to quit, 's' to save screenshot")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = annotated_frame.copy()  # Make writable copy
        
        # Add mode text
        cv2.putText(annotated_frame, f"Mode: {mode}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(f"YOLO {mode} - Localhost Deployment", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"{mode}_screenshot.jpg", annotated_frame)
            print(f"✓ Screenshot saved")
    
    cap.release()
    cv2.destroyAllWindows()

def test_on_samples():
    """Test on sample images if available"""
    print("\n📸 Testing on sample images...")
    
    # Create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "YOLO Deployment Test - Lab Assignment", (100, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    temp_path = "test_image.jpg"
    cv2.imwrite(temp_path, test_img)
    image_paths = [temp_path]
    
    # Also try to find any existing images
    possible_paths = [
        "datasets/COCO128/valid/images",
        "datasets/COCO128/train/images",
        "datasets/detection/valid/images",
        "."
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    image_paths.extend([os.path.join(path, img) for img in images[:2]])
                    break
            except:
                pass
    
    for img_path in image_paths[:3]:  # Max 3 images
        print(f"\n📷 Processing: {os.path.basename(img_path)}")
        
        # Detection
        print("  🔍 Object Detection...")
        det_results = detect_model(img_path)
        det_img = det_results[0].plot()
        det_img = det_img.copy()  # Make writable copy
        cv2.putText(det_img, "DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detection Result", det_img)
        cv2.waitKey(1500)
        
        # Classification
        print("  🏷️ Classification...")
        cls_results = cls_model(img_path)
        if cls_results[0].probs is not None:
            top1 = cls_results[0].names[cls_results[0].probs.top1]
            conf = cls_results[0].probs.top1conf.item()
            print(f"     → {top1} ({conf:.2%})")
            
            # Show classification image
            img = cv2.imread(img_path)
            if img is not None:
                img = img.copy()  # Make writable copy
                cv2.putText(img, f"{top1}: {conf:.2%}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Classification Result", img)
                cv2.waitKey(1500)
        
        # Pose
        print("  🧍 Pose Estimation...")
        pose_results = pose_model(img_path)
        pose_img = pose_results[0].plot()
        pose_img = pose_img.copy()  # Make writable copy
        cv2.putText(pose_img, "POSE ESTIMATION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Pose Result", pose_img)
        cv2.waitKey(1500)
    
    cv2.destroyAllWindows()
    print("\n✓ Testing complete!")
    
    # Cleanup
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")

if choice == '1':
    if use_webcam:
        run_webcam_detection(detect_model, "OBJECT DETECTION")
    else:
        test_on_samples()

elif choice == '2':
    if use_webcam:
        run_webcam_detection(cls_model, "CLASSIFICATION")
    else:
        test_on_samples()

elif choice == '3':
    if use_webcam:
        run_webcam_detection(pose_model, "POSE ESTIMATION")
    else:
        test_on_samples()

elif choice == '4':
    print("\n🔄 Running all three modes...")
    if use_webcam:
        for mode_name, model in [("DETECTION", detect_model), 
                                  ("CLASSIFICATION", cls_model), 
                                  ("POSE", pose_model)]:
            print(f"\n▶ Starting {mode_name} mode")
            run_webcam_detection(model, mode_name)
    else:
        test_on_samples()

elif choice == '5':
    test_on_samples()

else:
    print("Invalid choice")

print("\n" + "=" * 60)
print("✓ DEPLOYMENT COMPLETE")
print("=" * 60)