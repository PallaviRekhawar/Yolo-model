from ultralytics import YOLO
import cv2
import numpy as np

print("=" * 50)
print("IMAGE CLASSIFICATION DEPLOYMENT - LOCALHOST")
print("=" * 50)

# Load classification model (using pretrained since you may not have trained one yet)
print("\nLoading classification model...")
model = YOLO('yolo11n-cls.pt')  # Pretrained classification model
print("✓ Model loaded successfully!")

# Open webcam or use test images
print("\nOptions:")
print("1. Use webcam (real-time classification)")
print("2. Test on sample images")

choice = input("\nEnter choice (1 or 2): ")

if choice == '1':
    # Webcam mode
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        exit()
    
    print("\n✓ Webcam ready! Showing classifications...")
    print("Press 'q' to quit\n")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Resize for classification (224x224 is standard)
        resized = cv2.resize(frame, (224, 224))
        
        # Run classification
        results = model(resized)
        
        # Get top prediction
        if results[0].probs is not None:
            top1_idx = results[0].probs.top1
            top1_conf = results[0].probs.top1conf.item()
            top1_class = results[0].names[top1_idx]
            
            # Display on frame
            cv2.putText(frame, f"Class: {top1_class}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {top1_conf:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("YOLO Classification - Localhost", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

else:
    # Test on sample images from dataset
    import os
    test_dir = "datasets/COCO128/valid/images"
    if os.path.exists(test_dir):
        images = os.listdir(test_dir)[:5]  # First 5 images
        print(f"\nTesting on {len(images)} images...\n")
        
        for img_file in images:
            img_path = os.path.join(test_dir, img_file)
            results = model(img_path)
            
            if results[0].probs is not None:
                top1_idx = results[0].probs.top1
                top1_conf = results[0].probs.top1conf.item()
                top1_class = results[0].names[top1_idx]
                
                print(f"Image: {img_file}")
                print(f"  → Predicted: {top1_class} ({top1_conf:.2%})")
                
                # Display image
                img = cv2.imread(img_path)
                cv2.putText(img, f"{top1_class}: {top1_conf:.2%}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Classification Result", img)
                cv2.waitKey(1500)  # Show for 1.5 seconds
        cv2.destroyAllWindows()
    else:
        print("❌ No test images found")

print("\n✓ Classification deployment complete!")