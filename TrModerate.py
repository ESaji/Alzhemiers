from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

# Load a model

model = YOLO('yolov8n-cls.pt') 

# Train the model
results = model.train(data='ModerateDemented', epochs=50)