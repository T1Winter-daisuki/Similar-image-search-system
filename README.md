# Hệ thống CBIR: Nhận diện khuôn mặt trẻ em

Đây là project Bài tập lớn môn Hệ Cơ sở dữ liệu Đa phương tiện. Hệ thống tìm kiếm và nhận diện khuôn mặt trẻ em dựa trên nội dung ảnh (Content-Based Image Retrieval), kết hợp 3 đặc trưng:
- **Hình học (Geometric Landmarks):** Dùng Dlib để đo tỷ lệ các mốc trên khuôn mặt (R1, R2, R3, R4).
- **Kết cấu da (LBP - Local Binary Pattern):** Đánh giá độ mịn màng, đặc điểm bề mặt da.
- **Hình dáng (HOG - Histogram of Oriented Gradients):** Phân tích đường nét và góc cạnh khuôn mặt.

Giao diện cho phép người dùng tự kéo thanh trượt (slider) để tinh chỉnh trọng số của từng đặc trưng lúc tìm kiếm.

## 🛠 Tech Stack
- **Backend/Xử lý ảnh:** Python (OpenCV, Dlib, scikit-image, numpy)
- **Database:** PostgreSQL (Lưu trữ vector đặc trưng dưới dạng JSON)
- **Frontend/Giao diện:** Streamlit

## Cấu trúc thư mục cơ bản
```text
CSDLDPT/
├── be/
│   ├── config.py
│   ├── features.py
│   ├── database.py
│   ├── setup_db.py
|   ├── retrieval.py
│   └── shape_predictor_68_face_landmarks.dat
├── fe/
│   └── app.py
├── data/
│   └── datastudy/
└── README.md

Hướng dẫn cài đặt & Chạy Local
1. Chuẩn bị Database 
Tạo một database mới.
Vào file be/config.py, đổi thông tin user và password cho khớp với PostgreSQL trên máy của bạn.

2. Cài đặt thư viện
Khuyên dùng Python 3.12. Mở terminal và chạy lệnh:

Bash
pip install numpy opencv-python scikit-image scipy psycopg2-binary streamlit python-dotenv

(Nếu cài dlib bị lỗi Failed building wheel, thay thế bằng cách tải bên dưới: 
Lên Google tải file dlib-19.24.2-cp312-cp312-win_amd64.whl về và dùng lệnh pip install <đường-dẫn-file-whl>.)

3. Tải mô hình Dlib
Hệ thống cần file bản đồ khuôn mặt của Dlib.
Tải tại: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Giải nén ra file .dat (gần 100MB) và đưa vào thư mục be của project.

4. Nạp dữ liệu vào Database
Bỏ tất cả ảnh dataset vào thư mục data/datastudy. Sau đó mở terminal ở thư mục gốc và chạy:

Bash
python be/features.py
Lưu ý: Quá trình này sẽ mất một lúc tuỳ vào số lượng ảnh

5. Khởi động Giao diện Web
Khi DB đã có data, chạy lệnh này để bật Streamlit:

Bash
streamlit run fe/app.py
Web sẽ tự pop-up lên ở localhost:8501. Tải một bức ảnh query lên, chỉnh trọng số và xem Top 5 kết quả trả về.