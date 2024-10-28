import cv2
from ultralytics import YOLO 

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
    
    # cv2.imshow("Test", image)
    # cv2.waitKey(5)
    return image

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
frames =[]
ret = True
while True:
    ret, frame = cap.read()
    frame = run_inference(frame)
    frames.append(frame)
    if (cv2.waitKey(1) & 0xFF== ord('q')):
        break

def create_video_from_frames(frames, output_video, fps=30):
    """
    Creates a video from a list of frames.

    Parameters:
    frames (list): List of frames (images) in the form of NumPy arrays.
    output_video (str): Name of the output video file (e.g., 'output.mp4').
    fps (int): Frames per second for the video.

    Returns:
    None
    """
    if not frames:
        print("No frames to write to video.")
        return

    frame_height, frame_width = frames[0].shape[:2]
    frame_size = (frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

create_video_from_frames(frames,'output_video',fps=30)
