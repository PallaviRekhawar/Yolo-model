# Yolo-model
 Lab Assignment 3: Multi-Task Vision System using YOLO

 ##### Group Members : Sakshi Shende , Pranita Joshi , Pallavi Rekhawar , Riya Chougule

## 📋 Overview
This project implements a comprehensive computer vision system using YOLO (You Only Look Once) architecture to perform four major tasks:
- **Object Detection** - Identifying and localizing objects in images
- **Image Classification** - Categorizing images into predefined classes
- **Pose Estimation** - Detecting human body keypoints
- **Oriented Bounding Boxes (OBB)** - Detecting rotated objects

---

## 📊 Dataset Information

### Object Detection Dataset
- **Source**: Roboflow Universe
- **Dataset Link**: [Fruit Detection Dataset](https://universe.roboflow.com/ratul/fruit-detection-8ohhe)
- **Classes**: Apple-G1, Apple-G2
- **Format**: YOLO format

### Classification Dataset
- **Source**: Roboflow Universe
- **Dataset Link**: [Flower Dataset](https://universe.roboflow.com/ambos-vgbi4/flower-dataset-lnsyp/dataset/1)
- **Classes**: Daisy, Dandelion, Rose, Sunflower, Tulip
- **Format**: YOLO Classification format

### Pose Estimation Dataset
- **Source**: Roboflow Universe
- **Dataset Link**: [Pose Estimation Dataset](https://universe.roboflow.com/viswanadh/pose-estimation-i4f6o/dataset/1)
- **Format**: YOLOv8 Pose format (COCO keypoints)

---

## 🎥 Screen Recordings

| Task | Screen Recording Link |
|------|----------------------|
| Object Detection | [Watch Recording](https://drive.google.com/file/d/1HS_GuApNTbVYi8xiZUYyGz9ySlldZ1Fv/view?usp=sharing) |
| Classification | [Watch Recording](https://drive.google.com/file/d/1QU7nPT8-XmdIKV72TQMTgJxXWwEh4P5-/view?usp=sharing) |
| Pose Estimation | [Watch Recording](https://drive.google.com/file/d/1rWuPHJ8_Lp7osob_xPRUxabg0tY4FTvO/view?usp=sharing) |

---

## 🛠️ Tools & Technologies Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Programming Language |
| Ultralytics YOLO | 8.4.40 | Deep Learning Framework |
| PyTorch | 2.11.0+cpu | Neural Network Backend |
| OpenCV | 4.13.0 | Image Processing |
| Roboflow | Latest | Dataset Management |
| NumPy | Latest | Numerical Operations |

---

## 📁 Project Structure
YOLO_Lab_Assignment/
├── datasets/
│ ├── detection/ # Object detection dataset
│ │ ├── train/
│ │ ├── valid/
│ │ └── data.yaml
│ ├── classification/ # Classification dataset
│ │ ├── train/
│ │ ├── valid/
│ │ └── data.yaml
│ └── pose/ # Pose estimation dataset
│ ├── train/
│ ├── valid/
│ └── data.yaml
├── models/ # Trained model weights (.pt files)
├── runs/ # Training outputs and results
│ ├── detect/
│ ├── classify/
│ └── pose/
├── scripts/ # Training and deployment scripts
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## DEPLOYMENT COMMANDS
python deploy_detection.py

python deploy_classification.py

python deploy_pose.py

python deploy_pretrained_fixed.py

## DEPLOYMENT QUICK REFERENCE CARD
Task	Command	Output

Detection: Webcam	-yolo detect predict model=best.pt source=0 show=True	(Bounding boxes)

Classification: Image	-yolo classify predict model=best.pt source=image.jpg	(Class labels)

Pose: Webcam	-yolo pose predict model=best.pt source=0 show=True	(Skeleton keypoints)

## Final Summary
This lab assignment successfully demonstrated the implementation of a complete multi-task vision system using YOLO architecture. All four required tasks (Detection, Classification, Pose Estimation) were trained, validated, and deployed on a local machine without any cloud services. The project showcased YOLO's versatility as a unified framework for diverse computer vision applications.

The screen recordings, dataset links, trained models, and deployment proofs have been submitted as per the assignment guidelines. The GitHub repository with comprehensive README documentation serves as a complete portfolio piece demonstrating proficiency in modern computer vision techniques using YOLO.

