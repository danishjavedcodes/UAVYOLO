from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
yolov11_model = YOLO("yolo11n.pt")
yolov10_model = YOLO("yolov10n.pt")

results = yolov11_model.train(data="/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/YOLO/data.yaml", epochs=15, imgsz=128)
result = yolov10_model.train(data="/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/YOLO/data.yaml", epochs=15, imgsz=128)