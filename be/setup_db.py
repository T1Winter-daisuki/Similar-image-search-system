from database import FaceDatabase
import config

def init():
    db = FaceDatabase(config.DB_CONFIG)
    
    print("--- Đang kết nối và tạo bảng tại PostgreSQL ---")
    db.create_tables()
    print("--- Hoàn tất! ---")

if __name__ == "__main__":
    init()