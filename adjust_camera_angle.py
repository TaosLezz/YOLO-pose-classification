import cv2
from ultralytics import YOLO
# import cvzone
import os
import numpy as np
import torch
from equilib import equi2pers
from utils import plot_one_box, plot_skeleton_kpts

#X-RAY
# video_path = r'E:\aHieu\YOLO_pose_sleep\videos\Cam-5_2024-08-02_16-46-30.avi'
#
video_path = r'E:\aHieu\YOLO_pose_sleep\videos\1.avi'

output_dir = "data_faces_facenet_DongVang/Tien1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


frame_count = 0
face_count = 0

cap = cv2.VideoCapture(video_path)

# Kiểm tra xem camera có được mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

rots = {
    'roll': 0,
    'pitch': np.pi/6.5,  # xoay theo trục dọc
    'yaw': np.pi/-7,    # xoay theo trục ngang
}

# rots = {
#     'roll': 0,
#     'pitch': np.pi/6.5,  # xoay theo trục dọc
#     'yaw': np.pi/-5,    # xoay theo trục ngang
# }

# # Đặt các giá trị xoay XRAY-ROOM
# rots = {
#     'roll': 0,
#     'pitch': np.pi/2.5,  # xoay theo trục dọc
#     'yaw': np.pi/-5,    # xoay theo trục ngang
# }
# rots = {
#     'roll': 0,
#     'pitch': np.pi/2.5,  # xoay theo trục dọc
#     'yaw': np.pi/10,    # xoay theo trục ngang
# }
model = YOLO(r'E:\aHieu\YOLO_pose_sleep\models\yolov8m-pose.pt')

while True:
    ret, frame = cap.read()
    # Chuyển đổi ảnh sang định dạng mà hàm equi2pers yêu cầu (C, H, W)
    equi_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    equi_img = np.transpose(equi_img, (2, 0, 1))  # Chuyển đổi sang định dạng (C, H, W)
    tensor_equi_img = torch.from_numpy(equi_img)  # Chuyển đổi thành tensor PyTorch
    tensor_equi_img = tensor_equi_img.cuda() if torch.cuda.is_available() else tensor_equi_img  # Di chuyển lên GPU nếu có CUDA

    # Chạy hàm equi2pers
    pers_img = equi2pers(
        equi=tensor_equi_img,  # Truyền tensor đã chuyển đổi
        rots=rots,
        height=1080,
        width=1280,
        fov_x=60.0,
        mode="bilinear",
    )

    # Chuyển đổi lại ảnh từ định dạng (C, H, W) sang (H, W, C)
    pers_img = pers_img.cpu().numpy()  # Chuyển về numpy array
    pers_img = np.transpose(pers_img, (1, 2, 0))  # Chuyển đổi sang định dạng (H, W, C)
    pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)  # Chuyển đổi từ RGB sang BGR để lưu bằng OpenCV

    # Làm nét ảnh bằng bộ lọc Unsharp Mask
    gaussian_blur = cv2.GaussianBlur(pers_img, (9, 9), 10.0)
    sharp_img = cv2.addWeighted(pers_img, 1.5, gaussian_blur, -0.5, 0)
    result_frame = cv2.resize(sharp_img, (800, 600))

    # frame = cv2.resize(frame, (1020, 720))
    results = model.predict(result_frame, conf = 0.3)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data.numpy()):
            plot_one_box(box.xyxy[0], result_frame, (255, 0, 255), f'person {box.conf[0]:.3}')
            plot_skeleton_kpts(result_frame, pose, radius=5, shape=result_frame.shape[:2], confi=0.5, line_thick=2)
    frame_count += 1
    print('frame_count:',frame_count)
    cv2.imshow('frame', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()