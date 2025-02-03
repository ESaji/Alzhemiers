from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Run inference on an image
results = model("C:\\Users\\hp\\Downloads\\ww\\model_initial_1\\alzheimers_normal_dataset\\test\\MildDemented\\e9a4b3a2-4d90-4cdd-92e9-44ee6a839046.jpg")

# Define custom confidence thresholds for each class (example)
custom_conf_thresholds = {
    0: 0.5,  # Class ID 0 with 0.5 confidence
    1: 0.3,  # Class ID 1 with 0.3 confidence
    # Add more classes and thresholds as needed
}

# Filter results based on custom confidence thresholds
filtered_boxes = []
for box in results[0].boxes:
    class_id = int(box.cls)
    if box.conf >= custom_conf_thresholds.get(class_id, 0.25):  # Default threshold if class ID not in dict
        filtered_boxes.append(box)


