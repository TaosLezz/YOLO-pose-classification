import cv2
from ultralytics import YOLO
import pandas as pd
import os
import numpy as np
import torch
from equilib import equi2pers
from utils import plot_one_box, plot_skeleton_kpts

video_path = r'E:\aHieu\pose_recognition\video\yoga.mp4'

lm_list = []
label = "YOGA_TEST"
no_of_frames = 600

frame_count = 0
face_count = 0

cap = cv2.VideoCapture(video_path)

# Kiểm tra xem camera có được mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    
    result_frame = cv2.resize(frame, (800, 600))

    # frame = cv2.resize(frame, (1020, 720))
    results = model.predict(result_frame, conf = 0.3)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data.numpy()):
            plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {box.conf[0]:.3}')
            plot_skeleton_kpts(result_frame, pose, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
            lm_list.append(pose.flatten().tolist())
    frame_count += 1
    print('frame_count:',frame_count)
    cv2.imshow('frame', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()