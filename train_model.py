from roboflow import Roboflow
from ultralytics import YOLO

# Download the dataset
# rf = Roboflow(api_key="cZHhMkvFfxkBZZSMPv5l")
# project = rf.workspace("hoo-gnun0").project("table_tennis_paddle")
# version = project.version(2)
# dataset = version.download("yolov11")

# Train the model
model = YOLO("yolo11n.yaml")
dataset = "datasets\\table_tennis_paddle-2\\data.yaml" # path to the dataset
model.train(data=dataset, epochs=10, imgsz=640)