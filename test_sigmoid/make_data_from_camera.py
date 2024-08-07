import cv2

# Mở webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có được mở thành công không
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Lấy thông tin về chiều rộng và chiều cao của khung hình
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Định dạng codec và tạo VideoWriter đối tượng để ghi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, fps, (width, height))

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận khung hình (có thể đã hết)")
        break

    # Ghi khung hình vào tệp video
    out.write(frame)

    # Hiển thị khung hình
    cv2.imshow('Webcam', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tất cả đối tượng
cap.release()
out.release()
cv2.destroyAllWindows()
