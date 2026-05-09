import cv2
import dlib
import numpy as np
from skimage.feature import local_binary_pattern, hog
from scipy.spatial import distance as dist
import os
import config

# Import file database để lưu chuẩn SQL
from database import FaceDatabase 

class FeatureExtractor:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        # Khởi tạo các công cụ của dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_all(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None: 
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray)

        geo_vec = self._extract_geometric(gray)  # truyền gray
        lbp_vec = self._extract_lbp(gray)
        hog_vec = self._extract_hog(gray)

        if geo_vec is None or lbp_vec is None or hog_vec is None:
            return None

        return {
            "geo_vec": geo_vec.tolist(),
            "lbp_hist": lbp_vec.tolist(),
            "hog_vec": hog_vec.tolist()
        }
    
    def _extract_geometric(self, gray):  # ← đổi tên tham số cho rõ
        """Tính toán 4 tỉ lệ hình học từ 68 điểm mốc (Nhận ảnh GRAY)"""
        
        # Bỏ 2 dòng cv2.cvtColor và ascontiguousarray ở đây vì đã xử lý trong extract_all
        faces = self.detector(gray)
        
        if len(faces) != 1: 
            return None
        
        shape = self.predictor(gray, faces[0])
        coords = np.array([[p.x, p.y] for p in shape.parts()])

        face_width = dist.euclidean(coords[0], coords[16])
        face_height = dist.euclidean(coords[27], coords[8])
        r1 = face_height / face_width if face_width != 0 else 0

        left_eye = np.mean(coords[36:42], axis=0)
        right_eye = np.mean(coords[42:48], axis=0)
        r2 = dist.euclidean(left_eye, right_eye) / face_width

        r3 = dist.euclidean(coords[48], coords[54]) / face_width

        eye_mid = (left_eye + right_eye) / 2
        r4 = dist.euclidean(eye_mid, coords[30]) / (dist.euclidean(coords[30], coords[51]) + 1e-6)

        return np.array([r1, r2, r3, r4])

    def _extract_lbp(self, gray, P=8, R=1):
        """Tính Histogram LBP Uniform"""
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def _extract_hog(self, gray):
        """Tính đặc trưng hình dáng HOG"""
        resized = cv2.resize(gray, (64, 64))
        features = hog(resized, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)
        return features

if __name__ == "__main__":
    print("[INFO] Bắt đầu quá trình trích xuất đặc trưng...")
    
    # Khởi tạo công cụ
    extractor = FeatureExtractor(predictor_path=config.PREDICTOR_PATH)
    
    # Khởi tạo module Database để lưu chuẩn 2 bảng
    db = FaceDatabase(config.DB_CONFIG)
    
    # Quét thư mục ảnh
    image_files = [f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"[INFO] Tìm thấy {total_images} ảnh trong thư mục {config.DATA_DIR}")
    
    success_count = 0
    
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(config.DATA_DIR, filename)
        print(f"Đang xử lý [{idx+1}/{total_images}]: {filename}...", end=" ")
        
        try:
            # Gọi hàm trích xuất
            features = extractor.extract_all(img_path)
            
            if features is not None:
                # Đóng gói thông tin ảnh chuẩn bị lưu
                # (Lấy label tạm thời là tên file, bạn có thể sửa logic này sau nếu thư mục chia theo label)
                image_info = {
                    "label": filename.split('_')[0], 
                    "raw_path": img_path,
                    "clean_path": img_path
                }
                
                # Hàm save_feature sẽ tự động lưu vào bảng images, lấy ra ID, rồi lưu JSON vào bảng features
                db.save_feature(image_info, features)
                
                print("OK")
                success_count += 1
            else:
                print("BỎ QUA (Không tìm thấy mặt hoặc nhiều hơn 1 mặt)")
                
        except Exception as e:
            # Nếu có lỗi (file hỏng, ngoại lệ dlib...), in ra và tiếp tục chạy ảnh sau
            print(f"LỖI ({e})")
            continue
            
    print(f"\n[SUCCESS] Đã trích xuất và lưu thành công {success_count}/{total_images} ảnh vào Database!")