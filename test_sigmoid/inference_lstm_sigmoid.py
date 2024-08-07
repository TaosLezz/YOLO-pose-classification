import sys
import os
import torch
import cv2
from ultralytics import YOLO
import numpy as np
from equilib import equi2pers
import threading
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_one_box, plot_skeleton_kpts
import time

label = "Warmup...."
n_time_steps = 10
lm_list = []
poses = None

yolo_model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')
model = tf.keras.models.load_model(r"E:\aHieu\YOLO_pose_sleep\test_sigmoid\model1.h5")

video = r'E:\aHieu\YOLO_pose_sleep\test_sigmoid\output.avi'
cap = cv2.VideoCapture(video)
# Kiểm tra xem camera có được mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#DATA SLEEP DETECT
rots = {
    'roll': 0,
    'pitch': np.pi/6.5,  # xoay theo trục dọc
    'yaw': np.pi/-7,    # xoay theo trục ngang
}

#sigmoid results[0][0] là kq của lớp positive nhãn (1)
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    print('results[0][0]', results[0])
    prob_normal = results[0][0]
    prob_sleep = 1 - prob_normal
    if prob_sleep > 0.8:
        label = "SLEEP"
    else:
        label = "NORMAL"
    print(label)
    return label


i = 0
warmup_frames = 60
fps_start_time = 0
fps = 0
time_sleep = 0
sleep_start_time = None
while True:

    success, frame = cap.read()
    if not success:
        print("Can not read frame!")
    width = 800
    height = 600
    dim = (width, height)

    result_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # # Chuyển đổi ảnh sang định dạng mà hàm equi2pers yêu cầu (C, H, W)
    # equi_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # equi_img = np.transpose(equi_img, (2, 0, 1))  # Chuyển đổi sang định dạng (C, H, W)
    # tensor_equi_img = torch.from_numpy(equi_img)  # Chuyển đổi thành tensor PyTorch
    # tensor_equi_img = tensor_equi_img.cuda() if torch.cuda.is_available() else tensor_equi_img  # Di chuyển lên GPU nếu có CUDA

    # # Chạy hàm equi2pers
    # pers_img = equi2pers(
    #     equi=tensor_equi_img,  # Truyền tensor đã chuyển đổi
    #     rots=rots,
    #     height=1080,
    #     width=1280,
    #     fov_x=60.0,
    #     mode="bilinear",
    # )

    # # Chuyển đổi lại ảnh từ định dạng (C, H, W) sang (H, W, C)
    # pers_img = pers_img.cpu().numpy()  # Chuyển về numpy array
    # pers_img = np.transpose(pers_img, (1, 2, 0))  # Chuyển đổi sang định dạng (H, W, C)
    # pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)  # Chuyển đổi từ RGB sang BGR để lưu bằng OpenCV

    # # Làm nét ảnh bằng bộ lọc Unsharp Mask
    # gaussian_blur = cv2.GaussianBlur(pers_img, (9, 9), 10.0)
    # sharp_img = cv2.addWeighted(pers_img, 1.5, gaussian_blur, -0.5, 0)
    # result_frame = cv2.resize(sharp_img, (800, 600))
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / (time_diff + 0.000001)
    fps_start_time = fps_end_time
    fps_text = 'FPS: {:.2f}'.format(fps)
    i = i + 1
    if i > warmup_frames:
        if i % 1 == 0:
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
                    
                    if label == "SLEEP":
                        if sleep_start_time is None:
                            sleep_start_time = time.time()
                        time_sleep = time.time() - sleep_start_time
                    else:
                        sleep_start_time = None

                    
        else:
            if poses is not None:
                plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {box.conf[0]:.3}')
                plot_skeleton_kpts(result_frame, poses, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
                
    cv2.putText(result_frame, label, (30, 50), 1, 2, (0, 255, 0), 2)
    cv2.putText(result_frame, fps_text, (200, 50), 1, 2, (0, 255, 0), 2)
    cv2.putText(result_frame, f"Time sleep: {str(time_sleep)}", (500, 50), 1, 2, (0, 255, 0), 2)
    cv2.imshow("Image", result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
