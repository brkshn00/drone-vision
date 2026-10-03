from ultralytics import YOLO
import os

# Define the path to your dataset configuration file
# This file is typically downloaded from Roboflow or created manually
# It should look like this (data.yaml):
# train: ../train/images
# val: ../valid/images
# nc: 1
# names: ['your_object_name']

# Check if dataset.yaml exists, otherwise warn user
dataset_yaml = "datasets/custom_data/data.yaml"

if not os.path.exists(dataset_yaml):
    print(f"WARNING: Dataset configuration file not found at {dataset_yaml}")
    print("Please make sure you have exported your dataset to 'datasets/custom_data'")
    print("and that it contains a 'data.yaml' file.")
    # We don't exit here to allow the script to be imported/checked, 
    # but actual training will fail if called.

def train_model():
    # Load a model
    model = YOLO("yolov10n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # epochs=100: Train for 100 iterations (adjust based on accuracy needs)
    # imgsz=640: Image size
    # device='mps': Use Apple Silicon GPU if available (otherwise use 'cpu')
    try:
        results = model.train(data=dataset_yaml, epochs=100, imgsz=640, device='mps')
        print("Training completed successfully!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("re-trying with cpu...")
        try:
             results = model.train(data=dataset_yaml, epochs=100, imgsz=640, device='cpu')
             print("Training completed successfully!")
        except Exception as e2:
             print(f"Failed with CPU as well: {e2}")

if __name__ == '__main__':
    train_model()
