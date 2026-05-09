import os

DB_CONFIG = {
    "host": "localhost",
    "database": "your_database_name",
    "user": "postgres",
    "password": "your_password",
    "port": 5432
}

# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "datastudy")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# CẤU HÌNH TRỌNG SỐ TỔ HỢP
# Tổng 3 trọng số này phải bằng 1.0
WEIGHTS = {
    "geometric": 0.4,  # Trọng số cho cấu trúc xương mặt (Landmarks)
    "texture": 0.3,    # Trọng số cho kết cấu da (LBP)
    "shape": 0.3       # Trọng số cho khung hình dáng (HOG)
}

# CẤU HÌNH THUẬT TOÁN TRÍCH XUẤT
LBP_PARAMS = {
    "P": 8,
    "R": 1,
    "method": "uniform"
}

HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "size": (64, 64) # Kích thước ảnh chuẩn trước khi tính HOG
}

# CẤU HÌNH KẾT QUẢ TRẢ VỀ
TOP_K = 5