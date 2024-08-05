from ultralytics import YOLO
import cv2
import numpy as np
from utils import plot_skeleton_kpts, plot_one_box



video_path = r'E:\aHieu\YOLO_pose_sleep\videos\Cam-5_2024-08-02_16-46-30.avi'

model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')
cap = cv2.VideoCapture(video_path)


while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed to Read...')
        break

    # results = model.predict(img)
    results = model.predict(img, conf = 0.3)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data.numpy()):
            plot_one_box(box.xyxy[0], img, (255, 0, 255), f'person {box.conf[0]:.3}')
            plot_skeleton_kpts(img, pose, radius=5, shape=img.shape[:2], confi=0.5, line_thick=2)

    # img = cv2.resize(img, (1080, 720))
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()