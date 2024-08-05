import cv2
from ultralytics import YOLO
import numpy as np
import threading
import tensorflow as tf
from utils import plot_one_box, plot_skeleton_kpts
import time

label = "Warmup...."
n_time_steps = 10
lm_list = []
poses = None

yolo_model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')
model = tf.keras.models.load_model("model.h5")

video = r'E:\aHieu\pose_recognition\video\run_tes.mp4'
cap = cv2.VideoCapture(video)


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    print('results[0][0]', results[0])
    labels = ['RUN', 'YOGA', 'STAND']
    pre = np.argmax(results[0])
    confidence = results[0][pre]
    print("confidence: ", confidence)
    if confidence >= 0.9:
        label = labels[pre]
    else:
        label = "unknown"
    print(label)
    return label


i = 0
warmup_frames = 60
fps_start_time = 0
fps = 0
time_sleep = 0

while True:

    success, img = cap.read()
    width = 800
    height = 600
    dim = (width, height)

    result_frame = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_start_time = fps_end_time
    fps_text = 'FPS: {:.2f}'.format(fps)
    i = i + 1
    if i > warmup_frames:
        if i % 2 == 0:
            print("Start detect....")
            results = yolo_model.predict(result_frame, conf = 0.3)
            for result in results:
                for box, pose in zip(result.boxes, result.keypoints.data.numpy()):
                    plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {box.conf[0]:.3}')
                    plot_skeleton_kpts(result_frame, pose, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
                    poses = pose

                    lm_list.append(pose.flatten().tolist())
                    if len(lm_list) == n_time_steps:
                        label = detect(model, lm_list)
                        lm_list = []
                    
                    if label == "RUN":
                        time_sleep +=1

                    
        else:
            if poses is not None:
                plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {box.conf[0]:.3}')
                plot_skeleton_kpts(result_frame, poses, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
                
    cv2.putText(result_frame, label, (30, 50), 1, 2, (0, 255, 0), 2)
    cv2.putText(result_frame, fps_text, (200, 50), 1, 2, (0, 255, 0), 2)
    cv2.putText(result_frame, str(time_sleep), (500, 50), 1, 2, (0, 255, 0), 2)
    cv2.imshow("Image", result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
