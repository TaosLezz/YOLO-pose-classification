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
from lstm_model import LSTMModel


n_time_steps = 10
poses = None

yolo_model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')

# Khởi tạo và load mô hình đã huấn luyện
input_dim = 17 * 3  # số lượng keypoints * số lượng tọa độ (x, y, score)
hidden_dim = 50
num_layers = 4
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
model.load_state_dict(torch.load(r"E:\aHieu\YOLO_pose_sleep\test_sigmoid\model1.pth"))
model.eval()

# video = r'E:\aHieu\YOLO_pose_sleep\videos\1and2.mp4'
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

def detect(model, lm_list):
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    lm_list_tensor = torch.tensor(lm_list, dtype=torch.float32)
    with torch.no_grad():
        results = model(lm_list_tensor)
    print('results[0][0]', results[0])
    prob_normal = results.item()
    prob_sleep = 1 - prob_normal
    if prob_sleep > 0.8:
        return "SLEEP"
    else:
        return "NORMAL"

i = 0
warmup_frames = 60
fps_start_time = 0
fps = 0
lm_lists = []
time_sleep = []
sleep_start_time = []
labels = ["Warmup...."]

while True:

    success, frame = cap.read()
    if not success:
        print("Can not read frame!")
        break

    width = 800
    height = 600
    dim = (width, height)
    result_frame = cv2.resize(frame, dim)

    # equi_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # equi_img = np.transpose(equi_img, (2, 0, 1))
    # tensor_equi_img = torch.from_numpy(equi_img)
    # tensor_equi_img = tensor_equi_img.cuda() if torch.cuda.is_available() else tensor_equi_img

    # pers_img = equi2pers(
    #     equi=tensor_equi_img,
    #     rots=rots,
    #     height=1080,
    #     width=1280,
    #     fov_x=60.0,
    #     mode="bilinear",
    # )

    # pers_img = pers_img.cpu().numpy()
    # pers_img = np.transpose(pers_img, (1, 2, 0))
    # pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)

    # gaussian_blur = cv2.GaussianBlur(pers_img, (9, 9), 10.0)
    # sharp_img = cv2.addWeighted(pers_img, 1.5, gaussian_blur, -0.5, 0)
    # result_frame = cv2.resize(sharp_img, (800, 600))
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / (time_diff + 0.000001)
    fps_start_time = fps_end_time
    fps_text = 'FPS: {:.2f}'.format(fps)
    i += 1
    if i > warmup_frames:
        print("Start detect....")
        results = yolo_model.predict(result_frame, conf=0.3)
        # Lấy danh sách các tên đối tượng
        names = results[0].names

        # Lấy các bounding boxes
        boxes = results[0].boxes

        # Đếm số lượng người (label = 'person' với id = 0)
        num_people = sum(1 for box in boxes if names[int(box.cls)] == 'person')

        print(f"Số người phát hiện: {num_people}")
        if len(lm_lists) != num_people:
            lm_lists = [[] for _ in range(num_people)]
            time_sleep = [0] * num_people
            sleep_start_time = [None] * num_people
            labels = ["Warmup...."] * num_people

        person_idx = 0
        for result in results:
            for box, pose in zip(result.boxes, result.keypoints.data.numpy()):
                plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {person_idx + 1} {box.conf[0]:.3}')
                plot_skeleton_kpts(result_frame, pose, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
                poses = pose

                lm_lists[person_idx].append(pose.flatten().tolist())
                
                if len(lm_lists[person_idx]) == n_time_steps:
                    labels[person_idx] = detect(model, lm_lists[person_idx])
                    lm_lists[person_idx] = []
                
                if labels[person_idx] == "SLEEP":
                    if sleep_start_time[person_idx] is None:
                        sleep_start_time[person_idx] = time.time()
                    time_sleep[person_idx] = time.time() - sleep_start_time[person_idx]
                else:
                    sleep_start_time[person_idx] = None
                person_idx += 1
                print("person_idx", person_idx)

                
    # for idx, ts in enumerate(time_sleep):
    #     cv2.putText(result_frame, f"Person {idx + 1} Time sleep: {str(ts)}", (400, 50 + 30 * idx), 1, 2, (0, 255, 0), 2)
    for idx, ts in enumerate(time_sleep):
        y_offset = 100 + 30 * idx
        cv2.putText(result_frame, f"Person {idx + 1}", (30, y_offset), 1, 2, (0, 255, 0), 2)
        cv2.putText(result_frame, str(labels[idx]), (200, y_offset), 1, 2, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Time sleep: {ts:.2f}", (500, y_offset), 1, 2, (0, 255, 0), 2)

    
    # cv2.putText(result_frame, str(labels[0]), (30, 50), 1, 2, (0, 255, 0), 2)
    cv2.putText(result_frame, str(fps_text), (30, 50), 1, 2, (0, 255, 0), 2)

    cv2.imshow("Image", result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
