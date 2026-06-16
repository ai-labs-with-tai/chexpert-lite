import streamlit as st
import os
import pandas as pd
from PIL import Image
from model_utils import (
    discover_diseases, 
    get_resnet50_model, 
    get_densenet121_model,
    load_weights, 
    preprocess_image, 
    predict,
    predict_multilabel
)

st.set_page_config(page_title="Demo CheXpert DAM vs CE", layout="wide", page_icon="🫁")

# Define the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# UI: Sidebar
st.sidebar.title("🫁 Demo CheXpert")
st.sidebar.markdown("So sánh các mô hình được huấn luyện trên CheXpert dataset.")

# Model Architecture Selection
model_arch = st.sidebar.radio(
    "Chọn Kiến Trúc Mô Hình:",
    ["ResNet50 (DAM vs CE)", "DenseNet121 (Multi-label)"],
    help="ResNet50: So sánh mô hình DAM vs CE cho từng bệnh\nDenseNet121: Dự đoán đồng thời 5 bệnh lý"
)

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
mode = None
selected_disease_vn = None
selected_disease_eng = None

if model_arch == "ResNet50 (DAM vs CE)":
    mode = st.sidebar.radio("Chế độ Dự Đoán:", ["Một Bệnh Lý (So sánh chi tiết)", "Đa Nhãn (Tất cả bệnh lý)"])
    
    if mode == "Một Bệnh Lý (So sánh chi tiết)":
        selected_disease_vn = st.sidebar.selectbox("Chọn Bệnh Lý Để Dự Đoán:", available_diseases_vn)
        # Find the corresponding english name
        idx = available_diseases_vn.index(selected_disease_vn)
        selected_disease_eng = available_diseases_eng[idx]

st.sidebar.markdown("---")

# Threshold Settings
st.sidebar.subheader("⚙️ Cài Đặt Ngưỡng Phân Loại")
threshold = st.sidebar.slider(
    "Ngưỡng xác suất (Probability Threshold):",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Xác suất >= ngưỡng này sẽ được coi là 'Có khả năng mắc bệnh'. Điều chỉnh dựa trên độ nhạy mong muốn."
)

st.sidebar.info(f"🎯 Ngưỡng hiện tại: **{threshold:.2f}**\n\n"
                f"- **< {threshold:.2f}**: Khả năng thấp\n"
                f"- **≥ {threshold:.2f}**: Khả năng cao")

st.sidebar.markdown("---")
st.sidebar.subheader("Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn một ảnh X-quang ngực...", type=["jpg", "jpeg", "png"])

# UI: Main Content
if model_arch == "DenseNet121 (Multi-label)":
    st.title("DenseNet121 - Dự Đoán Đa Nhãn (5 Bệnh Lý)")
elif mode == "Một Bệnh Lý (So sánh chi tiết)":
    st.title(f"Dự Đoán Bệnh Lý: {selected_disease_vn}")
else:
    st.title("Dự Đoán Đa Nhãn (Multi-label Prediction)")

st.write("Tải ảnh lên từ thanh bên trái và nhấn **Chạy Dự Đoán**.")

# Medical Disclaimer
st.warning("⚕️ **LƯU Ý Y TẾ QUAN TRỌNG:**\n\n"
          "Kết quả từ hệ thống AI này chỉ mang tính **hỗ trợ tham khảo** và **KHÔNG thay thế** "
          "cho chẩn đoán lâm sàng của bác sĩ chuyên khoa. Mọi quyết định điều trị cần được thực hiện "
          "bởi bác sĩ có chứng chỉ hành nghề dựa trên khám lâm sàng toàn diện và các xét nghiệm bổ sung.")

# Caching models to avoid reloading them on every interaction
@st.cache_resource
def load_models(disease, base_path):
    dam_path = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "DAM", f"{disease}_latest.pth")
    ce_path_1 = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "CE", f"CE_{disease}.pth")
    ce_path_2 = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "CE", f"CE_{disease.replace('_', ' ')}.pth")
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

@st.cache_resource
def load_densenet121_model(base_path):
    """Load DenseNet121 model for multi-label classification"""
    model_path = os.path.join(base_path, "CheXpert_DAM", "Densenet121", "best_densenet.pth")
    
    try:
        model = get_densenet121_model(num_classes=5)
        model = load_weights(model, model_path)
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình DenseNet121: {e}")
        return None


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
            
            # DenseNet121 Multi-label Mode
            if model_arch == "DenseNet121 (Multi-label)":
                with st.spinner("Đang tải mô hình DenseNet121 và dự đoán..."):
                    model = load_densenet121_model(PROJECT_ROOT)
                    
                    if model is None:
                        st.error("Không thể tải mô hình DenseNet121. Vui lòng kiểm tra lại file trọng số.")
                    else:
                        results = predict_multilabel(model, img_tensor)
                        
                        # Create DataFrame with Vietnamese names
                        df_results = pd.DataFrame([
                            {
                                "Bệnh lý": get_vn_name(disease),
                                "Xác suất": prob
                            }
                            for disease, prob in results.items()
                        ]).set_index("Bệnh lý")
                        
                        st.markdown("### Bảng Xác Suất Dự Đoán")
                        st.dataframe(df_results.style.format({"Xác suất": "{:.2%}"}), use_container_width=True)
                        
                        st.markdown("### Biểu Đồ So Sánh Trực Quan")
                        st.bar_chart(df_results)
                        
                        st.success("Dự đoán đa nhãn hoàn tất!")
                        st.info("✨ Mô hình DenseNet121 được huấn luyện để dự đoán đồng thời 5 bệnh lý từ một ảnh X-quang.")
            
            # ResNet50 Mode
            elif mode == "Một Bệnh Lý (So sánh chi tiết)":
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
            
            # ResNet50 Multi-label mode
            elif mode == "Đa Nhãn (Tất cả bệnh lý)":
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
