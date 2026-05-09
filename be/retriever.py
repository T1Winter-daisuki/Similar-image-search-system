import config
import numpy as np
import psycopg2
import json

class CBIRRetriever:
    def __init__(self, db_config):
        """
        Khởi tạo hệ thống: Kết nối DB và tải toàn bộ đặc trưng lên RAM 
        """
        self.db_config = db_config
        self.image_ids = []
        self.db_geo = []
        self.db_lbp = []
        self.db_hog = []
        
        # Gọi hàm nạp dữ liệu ngay khi khởi động server
        self._load_database_to_ram()

    def _load_database_to_ram(self):
        """Kết nối PostgreSQL và tải dữ liệu lên RAM để truy vấn siêu tốc"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Truy vấn lấy id và 3 vector đặc trưng
            cursor.execute("SELECT image_id, geo_vec, lbp_hist, hog_vec FROM features")
            rows = cursor.fetchall()
            print(f"[DEBUG] Số rows trong DB: {len(rows)}")
            
            for row in rows:
                self.image_ids.append(row[0])
                
                # Tự động xử lý cả 2 trường hợp: JSON string hoặc list/dict thẳng
                geo = row[1] if isinstance(row[1], list) else json.loads(row[1])
                lbp = row[2] if isinstance(row[2], list) else json.loads(row[2])
                hog = row[3] if isinstance(row[3], list) else json.loads(row[3])
                
                self.db_geo.append(geo)
                self.db_lbp.append(lbp)
                self.db_hog.append(hog)
            
            # Thêm vào sau vòng lặp for, trước khi convert numpy
            print(f"[DEBUG] geo shapes: {set(len(x) for x in self.db_geo)}")
            print(f"[DEBUG] lbp shapes: {set(len(x) for x in self.db_lbp)}")
            print(f"[DEBUG] hog shapes: {set(len(x) for x in self.db_hog)}")
                
            # QUAN TRỌNG: Chuyển đổi tất cả sang Ma trận đa chiều Numpy
            # Giúp tính khoảng cách với hàng ngàn ảnh cùng lúc mà không cần vòng lặp FOR
            self.image_ids = np.array(self.image_ids)
            self.db_geo = np.array(self.db_geo, dtype=np.float32)
            self.db_lbp = np.array(self.db_lbp, dtype=np.float32)
            self.db_hog = np.array(self.db_hog, dtype=np.float32)
            
            cursor.close()
            conn.close()
            print(f"[SUCCESS] Đã nạp {len(self.image_ids)} khuôn mặt lên RAM.")
            
        except Exception as e:
            print(f"[ERROR] Lỗi kết nối Database: {e}")

    # ================= CÁC HÀM TÍNH KHOẢNG CÁCH ================= #

    def _chi_square_distance(self, hist_query, hist_db_matrix):
        """Tính khoảng cách Chi-Square (Mục 2.6.1) - Dùng cho LBP"""
        # Ép kiểu mảng query thành float32
        q = np.array(hist_query, dtype=np.float32)
        
        eps = 1e-10
        diff = q - hist_db_matrix
        add = q + hist_db_matrix
        return 0.5 * np.sum((diff ** 2) / (add + eps), axis=1)

    def _euclidean_distance(self, vec_query, vec_db_matrix):
        """Tính khoảng cách Euclidean (Mục 2.6.2) - Dùng cho Geo và HOG"""
        # Ép kiểu mảng query thành float32
        q = np.array(vec_query, dtype=np.float32)
        
        return np.linalg.norm(vec_db_matrix - q, axis=1)

    def _min_max_normalize(self, dist_array):
        """Chuẩn hóa Min-Max cục bộ về đoạn [0, 1] (Mục 3.4)"""
        min_val = np.min(dist_array)
        max_val = np.max(dist_array)
        if max_val - min_val == 0:
            return np.zeros_like(dist_array)
        return (dist_array - min_val) / (max_val - min_val)

    # ================= HÀM TÌM KIẾM CỐT LÕI ================= #

    def search(self, query_geo, query_lbp, query_hog, weights=(0.4, 0.3, 0.3), top_k=5):
        """
        Thực hiện tìm kiếm khuôn mặt
        Trả về: Danh sách Top K ảnh giống nhất kèm theo điểm số
        """
        w_geo, w_lbp, w_hog = weights
        
        # Bước 1: Tính 3 mảng khoảng cách riêng biệt (So khớp ảnh Query với toàn bộ DB)
        d_geo = self._euclidean_distance(query_geo, self.db_geo)
        d_lbp = self._chi_square_distance(query_lbp, self.db_lbp)
        d_hog = self._euclidean_distance(query_hog, self.db_hog)
        
        # Bước 2: Chuẩn hóa Min-Max
        norm_geo = self._min_max_normalize(d_geo)
        norm_lbp = self._min_max_normalize(d_lbp)
        norm_hog = self._min_max_normalize(d_hog)
        
        # Bước 3: Tổ hợp tuyến tính (Mục 2.6.3)
        total_distance = (w_geo * norm_geo) + (w_lbp * norm_lbp) + (w_hog * norm_hog)
        
        # Bước 4: Sắp xếp (args_sort trả về index của mảng từ nhỏ đến lớn)
        sorted_indices = np.argsort(total_distance)
        
        # Bước 5: Lấy Top K kết quả
        top_k_indices = sorted_indices[:top_k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "image_id": int(self.image_ids[idx]),
                "total_score": float(total_distance[idx]), # Càng gần 0 càng giống
                "detail_scores": {
                    "geo_score": float(norm_geo[idx]),
                    "lbp_score": float(norm_lbp[idx]),
                    "hog_score": float(norm_hog[idx])
                }
            })
            
        return results