from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model.track(source="falldown_44.mp4", show=True, save=True, conf=0.45)