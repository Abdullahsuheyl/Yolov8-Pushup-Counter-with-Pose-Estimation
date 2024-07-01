import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time

model = YOLO('yolov8n-pose.pt')

video_path = 'pushup.mp4'  
cap = cv.VideoCapture(video_path)
output_video_path = 'output_video.mp4'
original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v')  
out = cv.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

def check_parallel_to_ground(x1, y1, x2, y2, threshold=0.1):
    delta_y = abs(y2 - y1)
    delta_x = abs(x2 - x1)
    if delta_x == 0:  
        return False
    ratio = delta_y / delta_x
    return ratio < threshold  

def draw_circle(frame, x_point, y_point):
    cv.circle(frame, (int(x_point), int(y_point)), radius=5, color=(0, 255, 0), thickness=3)

def draw_line(frame, point1, point2):
    cv.line(frame, point1, point2, color=(0, 0, 255), thickness=2)

if not cap.isOpened():
    print("Error: Video doesn't exist.")
    exit()

pushup_counter = 0
last_pushup_time = None
cooldown = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    cv.putText(frame, f"PushUP: {pushup_counter}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for result in results:  
        if hasattr(result, 'keypoints'):  
            keypoints = result.keypoints.cpu().numpy() 
            for keypoint in keypoints:
                conf = keypoint.conf
                if conf.all() > 0.8:
                    xy = keypoint.xy
                    x7, y7 = xy[0][7]
                    x9, y9 = xy[0][5]
                    draw_circle(frame, x7, y7)
                    draw_circle(frame, x9, y9)
                    draw_line(frame, (int(x7), int(y7)), (int(x9), int(y9)))
                    current_time = time.time()

                    if last_pushup_time is None or (current_time - last_pushup_time) > cooldown:
                        if check_parallel_to_ground(x7, y7, x9, y9):               
                            pushup_counter += 1
                            last_pushup_time = current_time

    out.write(frame)

    cv.imshow('Pose Estimation', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
