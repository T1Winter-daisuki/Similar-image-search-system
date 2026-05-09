import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import psycopg2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
be_dir = os.path.abspath(os.path.join(current_dir, '..', 'be'))
sys.path.append(be_dir)

# Import các module do chính bạn viết
import config
from features import FeatureExtractor
from retriever import CBIRRetriever

st.set_page_config(page_title="Nhận diện gương mặt trẻ em", layout="wide")

# =========================
# 1. KHỞI TẠO HỆ THỐNG (DÙNG CACHE ĐỂ CHẠY SIÊU TỐC)
# =========================
@st.cache_resource
def load_system():
    extractor = FeatureExtractor(config.PREDICTOR_PATH)
    retriever = CBIRRetriever(config.DB_CONFIG)
    return extractor, retriever

with st.spinner("Đang khởi động hệ thống và nạp Database lên RAM..."):
    extractor, retriever = load_system()

# Hàm phụ để lấy đường dẫn ảnh từ Database
def get_image_path(image_id):
    conn = psycopg2.connect(**config.DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT clean_path, label FROM images WHERE id = %s", (image_id,))
    result = cur.fetchone()
    conn.close()
    return result

# =========================
# 2. GIAO DIỆN NGƯỜI DÙNG
# =========================
st.title("🔍 Hệ thống CBIR: Nhận diện khuôn mặt trẻ em")
st.markdown("Hệ thống sử dụng tổ hợp 3 đặc trưng: **Hình học (Landmarks), Kết cấu da (LBP) và Hình dáng (HOG)**.")

# --- Sidebar chỉnh trọng số ---
st.sidebar.header("⚙️ Cấu hình trọng số")
w_geo = st.sidebar.slider("Đặc trưng Hình học (Landmarks)", 0.0, 1.0, 0.4, 0.1)
w_lbp = st.sidebar.slider("Đặc trưng Bề mặt da (LBP)", 0.0, 1.0, 0.3, 0.1)
w_hog = st.sidebar.slider("Đặc trưng Hình dáng (HOG)", 0.0, 1.0, 0.3, 0.1)

# Upload ảnh truy vấn
query_file = st.file_uploader("Tải lên ảnh em bé cần tìm", type=["jpg", "png", "jpeg"])

# =========================
# 3. XỬ LÝ TÌM KIẾM
# =========================
if query_file is not None:
    # Tạo key duy nhất cho ảnh upload
    file_key = query_file.name + str(query_file.size)

    if "results" not in st.session_state or st.session_state.get("file_key") != file_key:
        # Chỉ tính lại khi ảnh mới được upload
        query_image = Image.open(query_file).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            query_image.save(tmp_file.name)
            tmp_path = tmp_file.name

        with st.spinner("Đang trích xuất đặc trưng và quét Database..."):
            features = extractor.extract_all(tmp_path)

        os.remove(tmp_path)

        if features is not None:
            results = retriever.search(
                query_geo=features["geo_vec"],
                query_lbp=features["lbp_hist"],
                query_hog=features["hog_vec"],
                weights=(w_geo, w_lbp, w_hog),
                top_k=5
            )
            st.session_state["results"] = results
            st.session_state["file_key"] = file_key
        else:
            st.session_state["results"] = None
            st.session_state["file_key"] = file_key

    # Hiển thị ảnh truy vấn
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Ảnh Truy Vấn")
        st.image(Image.open(query_file).convert("RGB"), use_container_width=True)

    with col2:
        st.subheader("Kết quả tìm kiếm (Top 5)")
        results = st.session_state.get("results")

        if results is None:
            st.error("❌ Không tìm thấy khuôn mặt nào trong ảnh, hoặc ảnh bị mờ. Vui lòng thử ảnh khác!")
        else:
            # --- BƯỚC 1: TẢI ẢNH VÀ RESIZE TOÀN BỘ Ở HẬU TRƯỜNG ---
            # (Gom sẵn vào 1 danh sách để Streamlit không bị render lắt nhắt)
            display_data = []
            for idx, res in enumerate(results):
                img_id = res["image_id"]
                total_score = res["total_score"]

                # Lấy đường dẫn thật từ DB
                row = get_image_path(img_id)
                if row is None:
                    continue
                img_path, label = row

                if os.path.exists(img_path):
                    # Mở ảnh và ép vuông chuẩn 200x200 theo đúng Mục 2.2
                    img = Image.open(img_path).resize((200, 200))
                    
                    # Rút gọn nhãn nếu quá dài (tránh việc rớt dòng gây xô lệch cột)
                    short_label = label[:12] + ".." if len(label) > 12 else label
                    
                    display_data.append((img, short_label, total_score))

            # --- BƯỚC 2: VẼ ĐỒNG LOẠT LÊN GIAO DIỆN ---
            res_cols = st.columns(5)
            for idx, (img, label, score) in enumerate(display_data):
                with res_cols[idx]:
                    st.image(img)
                    st.caption(f"**Top {idx+1}** | Nhãn: `{label}`")
                    st.caption(f"Độ dị biệt: `{score:.3f}`")