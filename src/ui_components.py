import pandas as pd
import streamlit as st

from src.config import DEFAULT_THRESHOLD, SUPPORTED_IMAGE_TYPES
from src.disease_info import get_vn_name
from src.prediction import interpret_probability


def initialize_session_state():
    if "gradcam_model_choice" not in st.session_state:
        st.session_state.gradcam_model_choice = "DAM (Optimized)"


def show_sidebar(available_diseases_eng):
    st.sidebar.title("🫁 Demo CheXpert")
    st.sidebar.markdown("So sánh các mô hình được huấn luyện trên CheXpert dataset.")

    model_arch = st.sidebar.radio(
        "Chọn Kiến Trúc Mô Hình:",
        ["ResNet50 (DAM vs CE)", "DenseNet121 (Multi-label)"],
        help="ResNet50: So sánh mô hình DAM vs CE cho từng bệnh\nDenseNet121: Dự đoán đồng thời 5 bệnh lý",
    )

    mode = None
    selected_disease_eng = None
    selected_disease_vn = None

    if model_arch == "ResNet50 (DAM vs CE)":
        mode = st.sidebar.radio(
            "Chế độ Dự Đoán:",
            ["Một Bệnh Lý (So sánh chi tiết)", "Đa Nhãn (Tất cả bệnh lý)"],
        )

        if mode == "Một Bệnh Lý (So sánh chi tiết)":
            available_diseases_vn = [get_vn_name(disease) for disease in available_diseases_eng]
            selected_disease_vn = st.sidebar.selectbox(
                "Chọn Bệnh Lý Để Dự Đoán:",
                available_diseases_vn,
            )
            selected_index = available_diseases_vn.index(selected_disease_vn)
            selected_disease_eng = available_diseases_eng[selected_index]

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Cài Đặt Ngưỡng Phân Loại")
    threshold = st.sidebar.slider(
        "Ngưỡng xác suất (Probability Threshold):",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Xác suất >= ngưỡng này sẽ được coi là 'Có khả năng mắc bệnh'.",
    )

    st.sidebar.info(
        f"🎯 Ngưỡng hiện tại: **{threshold:.2f}**\n\n"
        f"- **< {threshold:.2f}**: Khả năng thấp\n"
        f"- **≥ {threshold:.2f}**: Khả năng cao"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Grad-CAM Visualization")
    enable_gradcam = st.sidebar.checkbox(
        "Hiển thị vùng quan sát của Model",
        value=True,
        help="Grad-CAM hiển thị vùng mà model đang chú ý để đưa ra dự đoán",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tải Ảnh Lên")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn một ảnh X-quang ngực...",
        type=SUPPORTED_IMAGE_TYPES,
    )

    return model_arch, mode, selected_disease_eng, selected_disease_vn, threshold, enable_gradcam, uploaded_file


def show_page_title(model_arch, mode, selected_disease_vn):
    if model_arch == "DenseNet121 (Multi-label)":
        st.title("DenseNet121 - Dự Đoán Đa Nhãn (5 Bệnh Lý)")
    elif mode == "Một Bệnh Lý (So sánh chi tiết)":
        st.title(f"Dự Đoán Bệnh Lý: {selected_disease_vn}")
    else:
        st.title("Dự Đoán Đa Nhãn (Multi-label Prediction)")

    st.write("Tải ảnh lên từ thanh bên trái và nhấn **Chạy Dự Đoán**.")


def show_medical_disclaimer():
    st.warning(
        "⚕️ **LƯU Ý Y TẾ QUAN TRỌNG:**\n\n"
        "Kết quả từ hệ thống AI này chỉ mang tính **hỗ trợ tham khảo** và **KHÔNG thay thế** "
        "cho chẩn đoán lâm sàng của bác sĩ chuyên khoa. Mọi quyết định điều trị cần được thực hiện "
        "bởi bác sĩ có chứng chỉ hành nghề dựa trên khám lâm sàng toàn diện và các xét nghiệm bổ sung."
    )


def display_uploaded_image(image):
    st.subheader("Ảnh Đã Tải Lên")
    st.image(image, caption="Ảnh X-quang ngực", use_container_width=True)


def display_probability_with_interpretation(probability, threshold, label="Xác suất"):
    interpretation, icon, color = interpret_probability(probability, threshold)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(label=label, value=f"{probability:.2%}")
        st.progress(probability)
    with col2:
        st.markdown(
            f"<div style='text-align: center; padding: 10px;'>"
            f"<h3 style='color: {color};'>{icon}</h3>"
            f"<p style='color: {color}; font-weight: bold;'>{interpretation}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


def build_interpreted_results(results, threshold):
    interpreted_results = []
    for disease, probability in results.items():
        interpretation, icon, _ = interpret_probability(probability, threshold)
        interpreted_results.append(
            {
                "Bệnh lý": get_vn_name(disease),
                "Disease_EN": disease,
                "Xác suất": probability,
                "Đánh giá": f"{icon} {interpretation}",
            }
        )

    return sorted(interpreted_results, key=lambda row: row["Xác suất"], reverse=True)


def display_multilabel_summary(interpreted_results, threshold):
    high_risk_count = sum(1 for row in interpreted_results if "Khả năng cao" in row["Đánh giá"])
    medium_risk_count = sum(1 for row in interpreted_results if "Cần theo dõi" in row["Đánh giá"])

    if high_risk_count > 0:
        st.error(f"⚠️ **Phát hiện {high_risk_count} bệnh lý có khả năng cao** (≥ {threshold:.0%})")
    elif medium_risk_count > 0:
        st.warning(f"⚠️ **Có {medium_risk_count} bệnh lý cần theo dõi**")
    else:
        st.success("✅ **Tất cả các bệnh lý đều ở mức khả năng thấp**")


def display_multilabel_table_and_chart(interpreted_results):
    df_results = pd.DataFrame(interpreted_results)

    st.markdown("### 📊 Bảng Xác Suất Chi Tiết")
    display_df = df_results[["Bệnh lý", "Xác suất", "Đánh giá"]].set_index("Bệnh lý")
    st.dataframe(
        display_df.style.format({"Xác suất": "{:.2%}"}),
        use_container_width=True,
    )

    st.markdown("### 📈 Biểu Đồ So Sánh Trực Quan")
    chart_data = pd.DataFrame(
        {"Xác suất": [row["Xác suất"] for row in interpreted_results]},
        index=[row["Bệnh lý"] for row in interpreted_results],
    )
    st.bar_chart(chart_data)


def display_gradcam_images(image, heatmap, overlay):
    gradcam_cols = st.columns(3)

    with gradcam_cols[0]:
        st.markdown("**Ảnh Gốc**")
        st.image(image, use_container_width=True)

    with gradcam_cols[1]:
        st.markdown("**Heatmap**")
        st.image(heatmap, use_container_width=True)

    with gradcam_cols[2]:
        st.markdown("**Overlay**")
        st.image(overlay, use_container_width=True)
