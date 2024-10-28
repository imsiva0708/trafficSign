import cv2
from ultralytics import YOLO

model = YOLO('runs\\detect\\train2\\weights\\best.pt')  

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

def run_inference(image):
    
    results = model.predict(image, conf=0.50)  # Adjust the confidence threshold as necessary
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            
            confidence = detection.conf[0]
            class_id = int(detection.cls[0])
            label = f"{class_names[class_id]}: {confidence:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Test", image)
    cv2.waitKey(5)

cap = cv2.VideoCapture(0)

ret = True
while True:
    ret, frame = cap.read()
    run_inference(frame)
    if (cv2.waitKey(1) & 0xFF== ord('q')):
        break
img = cv2.imread('speedLimitClassificationModel\\dataSet\\train\\SPEED_LIMIT_15\\63001.jpg')
run_inference(img)