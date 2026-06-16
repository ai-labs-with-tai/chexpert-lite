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

def interpret_probability(prob, threshold):
    """
    Interpret probability based on threshold
    Returns: (interpretation, color, icon)
    """
    if prob >= threshold:
        return "Khả năng cao", "🔴", "#ff4444"
    elif prob >= threshold * 0.7:  # Between 70% and 100% of threshold
        return "Cần theo dõi", "🟡", "#ffaa00"
    else:
        return "Khả năng thấp", "🟢", "#00cc66"

def display_probability_with_interpretation(prob, threshold, label="Xác suất"):
    """Display probability with color-coded interpretation"""
    interpretation, icon, color = interpret_probability(prob, threshold)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(label=label, value=f"{prob:.2%}")
        st.progress(prob)
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 10px;'>"
                   f"<h3 style='color: {color};'>{icon}</h3>"
                   f"<p style='color: {color}; font-weight: bold;'>{interpretation}</p>"
                   f"</div>", unsafe_allow_html=True)


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
                        
                        # Create results with interpretation
                        interpreted_results = []
                        for disease, prob in results.items():
                            interpretation, icon, color = interpret_probability(prob, threshold)
                            interpreted_results.append({
                                "Bệnh lý": get_vn_name(disease),
                                "Xác suất": prob,
                                "Đánh giá": f"{icon} {interpretation}"
                            })
                        
                        df_results = pd.DataFrame(interpreted_results).set_index("Bệnh lý")
                        
                        # Display summary statistics
                        high_risk_count = sum(1 for r in interpreted_results if "Khả năng cao" in r["Đánh giá"])
                        medium_risk_count = sum(1 for r in interpreted_results if "Cần theo dõi" in r["Đánh giá"])
                        
                        if high_risk_count > 0:
                            st.error(f"⚠️ **Phát hiện {high_risk_count} bệnh lý có khả năng cao** (≥ {threshold:.0%})")
                        elif medium_risk_count > 0:
                            st.warning(f"⚠️ **Có {medium_risk_count} bệnh lý cần theo dõi**")
                        else:
                            st.success("✅ **Tất cả các bệnh lý đều ở mức khả năng thấp**")
                        
                        st.markdown("### 📊 Bảng Xác Suất Chi Tiết")
                        st.dataframe(
                            df_results.style.format({"Xác suất": "{:.2%}"}),
                            use_container_width=True
                        )
                        
                        st.markdown("### 📈 Biểu Đồ So Sánh Trực Quan")
                        # Create color-coded bar chart
                        chart_data = pd.DataFrame({
                            "Xác suất": [r["Xác suất"] for r in interpreted_results]
                        }, index=[r["Bệnh lý"] for r in interpreted_results])
                        st.bar_chart(chart_data)
                        
                        # Add threshold line visualization
                        st.markdown(f"<div style='border-top: 2px dashed red; margin-top: -20px; margin-bottom: 10px;'>"
                                   f"<small style='color: red;'>Ngưỡng: {threshold:.0%}</small></div>", 
                                   unsafe_allow_html=True)
                        
                        st.success("Dự đoán đa nhãn hoàn tất!")
                        st.info("✨ **Mô hình DenseNet121** được huấn luyện để dự đoán đồng thời 5 bệnh lý từ một ảnh X-quang. "
                               f"Ngưỡng phân loại hiện tại: **{threshold:.0%}**")
            
            # ResNet50 Mode
            elif mode == "Một Bệnh Lý (So sánh chi tiết)":
                with st.spinner("Đang tải mô hình và dự đoán..."):
                    model_ce, model_dam = load_models(selected_disease_eng, PROJECT_ROOT)
                    
                    if model_ce is None and model_dam is None:
                        st.error("Không thể tải cả hai mô hình. Vui lòng kiểm tra lại các file trọng số.")
                    else:
                        res_cols = st.columns(2)
                        
                        prob_ce = None
                        prob_dam = None
                        
                        with res_cols[0]:
                            st.markdown("### 🟦 Baseline (CE)")
                            if model_ce:
                                prob_ce = predict(model_ce, img_tensor)
                                display_probability_with_interpretation(prob_ce, threshold, "Xác suất")
                            else:
                                st.write("Mô hình không khả dụng.")
                                
                        with res_cols[1]:
                            st.markdown("### 🟧 Optimized (DAM)")
                            if model_dam:
                                prob_dam = predict(model_dam, img_tensor)
                                display_probability_with_interpretation(prob_dam, threshold, "Xác suất")
                            else:
                                st.write("Mô hình không khả dụng.")
                        
                        # Comparison summary
                        if prob_ce is not None and prob_dam is not None:
                            st.markdown("---")
                            st.markdown("### 📊 So Sánh Kết Quả")
                            
                            diff = prob_dam - prob_ce
                            if abs(diff) < 0.01:
                                st.info(f"Cả hai mô hình cho kết quả tương đương (~{prob_dam:.1%})")
                            elif diff > 0:
                                st.success(f"✅ Mô hình DAM đánh giá xác suất cao hơn CE: **+{diff:.2%}**")
                            else:
                                st.warning(f"⚠️ Mô hình CE đánh giá xác suất cao hơn DAM: **+{abs(diff):.2%}**")
                        
                        st.success("Dự đoán hoàn tất!")
                        st.info("💡 **Lưu ý:** Mô hình DAM được tối ưu hóa cho AUC (Area Under ROC Curve), "
                               "giúp phân định xác suất bệnh lý tốt hơn so với Cross-Entropy tiêu chuẩn. "
                               f"Ngưỡng phân loại hiện tại: **{threshold:.0%}**")
            
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
                    
                    # Get interpretation for DAM (optimized model)
                    interpretation_dam, icon_dam, _ = interpret_probability(prob_dam, threshold)
                    
                    results.append({
                        "Bệnh lý": disease_vn,
                        "DAM (Tối ưu)": prob_dam,
                        "CE (Cơ sở)": prob_ce,
                        "Đánh giá": f"{icon_dam} {interpretation_dam}"
                    })
                    
                    my_bar.progress((i + 1) / len(available_diseases_eng), text=f"Đã xử lý: {disease_vn}")
                
                my_bar.empty()
                
                # Display summary
                high_risk = [r for r in results if "Khả năng cao" in r["Đánh giá"]]
                medium_risk = [r for r in results if "Cần theo dõi" in r["Đánh giá"]]
                
                if high_risk:
                    st.error(f"⚠️ **Phát hiện {len(high_risk)} bệnh lý có khả năng cao:**")
                    for r in high_risk:
                        st.write(f"  • {r['Bệnh lý']}: {r['DAM (Tối ưu)']:.1%}")
                elif medium_risk:
                    st.warning(f"⚠️ **Có {len(medium_risk)} bệnh lý cần theo dõi**")
                else:
                    st.success("✅ **Tất cả các bệnh lý đều ở mức khả năng thấp**")
                
                # Display Results
                df_results = pd.DataFrame(results).set_index("Bệnh lý")
                
                st.markdown("### 📊 Bảng Xác Suất Dự Đoán")
                st.dataframe(
                    df_results.style.format({
                        "DAM (Tối ưu)": "{:.2%}",
                        "CE (Cơ sở)": "{:.2%}"
                    }),
                    use_container_width=True
                )
                
                st.markdown("### 📈 Biểu Đồ So Sánh Trực Quan")
                chart_data = df_results[["DAM (Tối ưu)", "CE (Cơ sở)"]]
                st.bar_chart(chart_data)
                
                st.success("Dự đoán đa nhãn hoàn tất!")
                st.info(f"Đã chạy ngầm tổng cộng **{len(available_diseases_eng) * 2} mô hình** "
                       f"để tạo ra kết quả đa nhãn (Multi-label). "
                       f"Ngưỡng phân loại: **{threshold:.0%}**")

else:
    st.info("Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu.")
