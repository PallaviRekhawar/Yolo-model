import torch
import os

# Load the .pt file with weights_only=False (safe since you created this file)
model_path = 'yolov8n.pt'  # Change to your file name

print("=" * 50)
print(f"File: {model_path}")
print("=" * 50)

# Load the file (updated for PyTorch 2.6+)
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Print basic info
print(f"\nKeys in file: {list(checkpoint.keys())}")

# Check training info
if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")

if 'best_fitness' in checkpoint:
    print(f"Best fitness: {checkpoint['best_fitness']}")

if 'train_args' in checkpoint:
    print(f"\nTraining arguments:")
    for k, v in checkpoint['train_args'].items():
        print(f"  {k}: {v}")

# File size
size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\nFile size: {size:.2f} MB")

print("\n✅ Model file loaded successfully!")