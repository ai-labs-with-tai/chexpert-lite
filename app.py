import streamlit as st
import os
import pandas as pd
from PIL import Image
from model_utils import discover_diseases, get_resnet50_model, load_weights, preprocess_image, predict

st.set_page_config(page_title="Demo CheXpert DAM vs CE", layout="wide", page_icon="🫁")

# Define the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# UI: Sidebar
st.sidebar.title("🫁 Demo CheXpert")
st.sidebar.markdown("So sánh các mô hình ResNet50 được huấn luyện bằng phương pháp Deep AUC Maximization (DAM) và Cross-Entropy (CE) tiêu chuẩn.")

# Dynamically discover diseases
available_diseases_eng = discover_diseases(PROJECT_ROOT)

# Vietnamese translation map for diseases
DISEASE_VN_MAP = {
    "Atelectasis": "Xẹp phổi (Atelectasis)",
    "Cardiomegaly": "Tim to (Cardiomegaly)",
    "Consolidation": "Đông đặc phổi (Consolidation)",
    "Edema": "Phù phổi (Edema)",
    "Pleural_Effusion": "Tràn dịch màng phổi (Pleural Effusion)",
    "Pleural Effusion": "Tràn dịch màng phổi (Pleural Effusion)"
}

def get_vn_name(eng_name):
    return DISEASE_VN_MAP.get(eng_name, eng_name)

if not available_diseases_eng:
    st.sidebar.error(f"Không tìm thấy mô hình nào trong thư mục {os.path.join(PROJECT_ROOT, 'CheXpert_DAM')}. Vui lòng kiểm tra lại.")
    st.stop()
    
available_diseases_vn = [get_vn_name(d) for d in available_diseases_eng]

# Mode Selection
mode = st.sidebar.radio("Chế độ Dự Đoán:", ["Một Bệnh Lý (So sánh chi tiết)", "Đa Nhãn (Tất cả bệnh lý)"])

selected_disease_vn = None
selected_disease_eng = None
if mode == "Một Bệnh Lý (So sánh chi tiết)":
    selected_disease_vn = st.sidebar.selectbox("Chọn Bệnh Lý Để Dự Đoán:", available_diseases_vn)
    # Find the corresponding english name
    idx = available_diseases_vn.index(selected_disease_vn)
    selected_disease_eng = available_diseases_eng[idx]

st.sidebar.markdown("---")
st.sidebar.subheader("Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn một ảnh X-quang ngực...", type=["jpg", "jpeg", "png"])

# UI: Main Content
if mode == "Một Bệnh Lý (So sánh chi tiết)":
    st.title(f"Dự Đoán Bệnh Lý: {selected_disease_vn}")
else:
    st.title("Dự Đoán Đa Nhãn (Multi-label Prediction)")

st.write("Tải ảnh lên từ thanh bên trái và nhấn **Chạy Dự Đoán**.")

# Caching models to avoid reloading them on every interaction
@st.cache_resource
def load_models(disease, base_path):
    dam_path = os.path.join(base_path, "CheXpert_DAM", "DAM", f"{disease}_latest.pth")
    ce_path_1 = os.path.join(base_path, "CheXpert_DAM", "CE", f"CE_{disease}.pth")
    ce_path_2 = os.path.join(base_path, "CheXpert_DAM", "CE", f"CE_{disease.replace('_', ' ')}.pth")
    ce_path = ce_path_1 if os.path.exists(ce_path_1) else ce_path_2
    
    model_ce = get_resnet50_model()
    model_dam = get_resnet50_model()
    
    try:
        model_ce = load_weights(model_ce, ce_path)
    except Exception as e:
        st.warning(f"Không thể tải mô hình CE cho {disease}: {e}")
        model_ce = None
        
    try:
        model_dam = load_weights(model_dam, dam_path)
    except Exception as e:
        st.warning(f"Không thể tải mô hình DAM cho {disease}: {e}")
        model_dam = None
        
    return model_ce, model_dam


if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Ảnh Đã Tải Lên")
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh X-quang ngực", use_column_width=True)
        
    with col2:
        st.subheader("So Sánh Mô Hình" if mode == "Một Bệnh Lý (So sánh chi tiết)" else "Kết Quả Đa Nhãn")
        
        if st.button("Chạy Dự Đoán", type="primary"):
            img_tensor = preprocess_image(image)
            
            if mode == "Một Bệnh Lý (So sánh chi tiết)":
                with st.spinner("Đang tải mô hình và dự đoán..."):
                    model_ce, model_dam = load_models(selected_disease_eng, PROJECT_ROOT)
                    
                    if model_ce is None and model_dam is None:
                        st.error("Không thể tải cả hai mô hình. Vui lòng kiểm tra lại các file trọng số.")
                    else:
                        res_cols = st.columns(2)
                        with res_cols[0]:
                            st.markdown("### 🟦 Baseline (CE)")
                            if model_ce:
                                prob_ce = predict(model_ce, img_tensor)
                                st.metric(label="Xác suất", value=f"{prob_ce:.2%}")
                                st.progress(prob_ce)
                            else:
                                st.write("Mô hình không khả dụng.")
                                
                        with res_cols[1]:
                            st.markdown("### 🟧 Optimized (DAM)")
                            if model_dam:
                                prob_dam = predict(model_dam, img_tensor)
                                st.metric(label="Xác suất", value=f"{prob_dam:.2%}")
                                st.progress(prob_dam)
                            else:
                                st.write("Mô hình không khả dụng.")
                                
                        st.success("Dự đoán hoàn tất!")
                        st.info("Lưu ý: Mô hình DAM được tối ưu hóa cho Area Under the ROC Curve (AUC), giúp mô hình phân định xác suất bệnh lý tốt hơn so với Cross-Entropy tiêu chuẩn.")
            
            else:
                # Multi-label mode
                progress_text = "Đang chạy dự đoán qua tất cả các mô hình..."
                my_bar = st.progress(0, text=progress_text)
                
                results = []
                for i, disease_eng in enumerate(available_diseases_eng):
                    disease_vn = get_vn_name(disease_eng)
                    model_ce, model_dam = load_models(disease_eng, PROJECT_ROOT)
                    
                    prob_ce = predict(model_ce, img_tensor) if model_ce else 0.0
                    prob_dam = predict(model_dam, img_tensor) if model_dam else 0.0
                    
                    results.append({
                        "Bệnh lý": disease_vn,
                        "DAM (Tối ưu)": prob_dam,
                        "CE (Cơ sở)": prob_ce
                    })
                    
                    my_bar.progress((i + 1) / len(available_diseases_eng), text=f"Đã xử lý: {disease_vn}")
                
                my_bar.empty() # Remove progress bar when done
                
                # Display Results
                df_results = pd.DataFrame(results).set_index("Bệnh lý")
                
                st.markdown("### Bảng Xác Suất Dự Đoán")
                st.dataframe(df_results.style.format("{:.2%}"), use_container_width=True)
                
                st.markdown("### Biểu Đồ So Sánh Trực Quan")
                st.bar_chart(df_results)
                
                st.success("Dự đoán đa nhãn hoàn tất!")
                st.info("Đã chạy ngầm tổng cộng {} mô hình để tạo ra kết quả đa nhãn (Multi-label) ở trên.".format(len(available_diseases_eng) * 2))

else:
    st.info("Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu.")
