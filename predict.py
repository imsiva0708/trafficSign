from ultralytics import YOLO 
import cv2 

model_path = 'runs\\detect\\train2\\weights\\best.pt'

model = YOLO(model_path)

class_names = [
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]

video_path = 'video.mp4'
model.predict(video_path,show=True,save=True)