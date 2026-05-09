import config
import psycopg2
from psycopg2.extras import execute_values
import json

class FaceDatabase:
    def __init__(self, db_config):
        self.db_config = db_config

    # Tạo bảng
    def create_tables(self):
        commands = (
            """
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                label VARCHAR(255),
                raw_path TEXT,
                clean_path TEXT,
                curated_ok INTEGER DEFAULT 1
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS features (
                image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
                geo_vec JSONB,
                lbp_hist JSONB,
                hog_vec JSONB,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            for command in commands:
                cur.execute(command)
            cur.close()
            conn.commit()
            print("[SUCCESS] Đã khởi tạo cấu trúc bảng PostgreSQL.")
        except Exception as e:
            print(f"[ERROR] Không thể tạo bảng: {e}")
        finally:
            if conn: conn.close()

    # Lưu ttin ảnh/vector đặc trưng vào DB
    def save_feature(self, image_info, features):
        """
        image_info: dict {label, raw_path, clean_path}
        features: dict {geo_vec, lbp_hist, hog_vec}
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Chèn vào bảng images và lấy ID vừa tạo
            cur.execute(
                "INSERT INTO images (label, raw_path, clean_path) VALUES (%s, %s, %s) RETURNING id",
                (image_info['label'], image_info['raw_path'], image_info['clean_path'])
            )
            image_id = cur.fetchone()[0]
            
            # Chèn vào bảng features (Convert list sang JSON string)
            cur.execute(
                "INSERT INTO features (image_id, geo_vec, lbp_hist, hog_vec) VALUES (%s, %s, %s, %s)",
                (
                    image_id, 
                    json.dumps(features['geo_vec']), 
                    json.dumps(features['lbp_hist']), 
                    json.dumps(features['hog_vec'])
                )
            )
            
            conn.commit()
            cur.close()
            return image_id
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu dữ liệu: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    def get_all_features(self):
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        cur.execute("SELECT image_id, geo_vec, lbp_hist, hog_vec FROM features")
        data = cur.fetchall()
        cur.close()
        conn.close()
        return data