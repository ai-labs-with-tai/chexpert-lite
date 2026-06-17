import pandas as pd
from PIL import Image
import streamlit as st

from src.config import PROJECT_ROOT, RESNET50_DAM_DIR
from src.disease_info import get_vn_name
from src.gradcam import generate_gradcam_visualization
from src.image_processing import preprocess_image
from src.model_loader import discover_diseases, load_ce_model, load_dam_model
from src.model_loader import load_densenet121_model as load_densenet121_weights
from src.prediction import compare_ce_and_dam, interpret_probability
from src.prediction import predict_all_diseases, predict_single_disease
from src.ui_components import (
    build_interpreted_results,
    display_gradcam_images,
    display_multilabel_summary,
    display_multilabel_table_and_chart,
    display_probability_with_interpretation,
    display_uploaded_image,
    initialize_session_state,
    show_medical_disclaimer,
    show_page_title,
    show_sidebar,
)


st.set_page_config(page_title="Demo CheXpert DAM vs CE", layout="wide", page_icon="🫁")
initialize_session_state()


@st.cache_resource
def load_resnet50_models(disease, base_path):
    model_ce = None
    model_dam = None

    try:
        model_ce = load_ce_model(disease, base_path)
    except Exception as error:
        st.warning(f"Không thể tải mô hình CE cho {disease}: {error}")

    try:
        model_dam = load_dam_model(disease, base_path)
    except Exception as error:
        st.warning(f"Không thể tải mô hình DAM cho {disease}: {error}")

    return model_ce, model_dam


@st.cache_resource
def load_densenet121_model(base_path):
    try:
        return load_densenet121_weights(base_path)
    except Exception as error:
        st.error(f"Không thể tải mô hình DenseNet121: {error}")
        return None


def show_densenet121_prediction(image, image_tensor, threshold, enable_gradcam):
    with st.spinner("Đang tải mô hình DenseNet121 và dự đoán..."):
        model = load_densenet121_model(PROJECT_ROOT)

        if model is None:
            st.error("Không thể tải mô hình DenseNet121. Vui lòng kiểm tra lại file trọng số.")
            return

        results = predict_all_diseases(model, image_tensor)
        interpreted_results = build_interpreted_results(results, threshold)

        display_multilabel_summary(interpreted_results, threshold)
        display_multilabel_table_and_chart(interpreted_results)

        if enable_gradcam:
            show_densenet121_gradcam(model, image, image_tensor, interpreted_results)

        st.success("Dự đoán đa nhãn hoàn tất!")
        st.info(
            "✨ **Mô hình DenseNet121** được huấn luyện để dự đoán đồng thời 5 bệnh lý từ một ảnh X-quang. "
            f"Ngưỡng phân loại hiện tại: **{threshold:.0%}**"
        )


def show_densenet121_gradcam(model, image, image_tensor, interpreted_results):
    st.markdown("---")
    st.markdown("### 🔍 Grad-CAM: Vùng Mà Model Đang Chú Ý")
    st.info(
        "💡 **Grad-CAM** (Gradient-weighted Class Activation Mapping) hiển thị vùng ảnh mà model tập trung "
        "để đưa ra dự đoán. Màu **đỏ/vàng** = quan trọng nhất, màu **xanh/tím** = ít quan trọng."
    )

    disease_names = [row["Bệnh lý"] for row in interpreted_results]
    selected_disease_gradcam = st.selectbox(
        "Chọn bệnh lý để hiển thị Grad-CAM:",
        disease_names,
        key="densenet_disease_selector",
        help="Chọn bệnh lý để xem vùng mà model chú ý cho bệnh đó",
    )
    selected_index = disease_names.index(selected_disease_gradcam)

    with st.spinner(f"Đang tạo Grad-CAM cho {selected_disease_gradcam}..."):
        overlay, heatmap, _ = generate_gradcam_visualization(
            model,
            image,
            image_tensor,
            class_idx=selected_index,
        )

        if overlay is None:
            st.error("Không thể tạo Grad-CAM. Vui lòng thử lại.")
            return

        display_gradcam_images(image, heatmap, overlay)
        st.caption(
            f"🎯 Grad-CAM cho bệnh lý: **{selected_disease_gradcam}** "
            f"(Xác suất: {interpreted_results[selected_index]['Xác suất']:.1%})"
        )


def show_single_disease_prediction(image, image_tensor, selected_disease_eng, selected_disease_vn, threshold, enable_gradcam):
    with st.spinner("Đang tải mô hình và dự đoán..."):
        model_ce, model_dam = load_resnet50_models(selected_disease_eng, PROJECT_ROOT)

        if model_ce is None and model_dam is None:
            st.error("Không thể tải cả hai mô hình. Vui lòng kiểm tra lại các file trọng số.")
            return

        res_cols = st.columns(2)
        prob_ce = None
        prob_dam = None

        with res_cols[0]:
            st.markdown("### 🟦 Baseline (CE)")
            if model_ce:
                prob_ce = predict_single_disease(model_ce, image_tensor)
                display_probability_with_interpretation(prob_ce, threshold, "Xác suất")
            else:
                st.write("Mô hình không khả dụng.")

        with res_cols[1]:
            st.markdown("### 🟧 Optimized (DAM)")
            if model_dam:
                prob_dam = predict_single_disease(model_dam, image_tensor)
                display_probability_with_interpretation(prob_dam, threshold, "Xác suất")
            else:
                st.write("Mô hình không khả dụng.")

        if prob_ce is not None and prob_dam is not None:
            show_ce_dam_comparison(prob_ce, prob_dam)

        if enable_gradcam and model_dam is not None:
            show_resnet50_gradcam(model_ce, model_dam, image, image_tensor, selected_disease_vn, prob_ce, prob_dam)

        st.success("Dự đoán hoàn tất!")
        st.info(
            "💡 **Lưu ý:** Mô hình DAM được tối ưu hóa cho AUC (Area Under ROC Curve), "
            "giúp phân định xác suất bệnh lý tốt hơn so với Cross-Entropy tiêu chuẩn. "
            f"Ngưỡng phân loại hiện tại: **{threshold:.0%}**"
        )


def show_ce_dam_comparison(prob_ce, prob_dam):
    st.markdown("---")
    st.markdown("### 📊 So Sánh Kết Quả")

    difference = compare_ce_and_dam(prob_ce, prob_dam)
    if abs(difference) < 0.01:
        st.info(f"Cả hai mô hình cho kết quả tương đương (~{prob_dam:.1%})")
    elif difference > 0:
        st.success(f"✅ Mô hình DAM đánh giá xác suất cao hơn CE: **+{difference:.2%}**")
    else:
        st.warning(f"⚠️ Mô hình CE đánh giá xác suất cao hơn DAM: **+{abs(difference):.2%}**")


def show_resnet50_gradcam(model_ce, model_dam, image, image_tensor, selected_disease_vn, prob_ce, prob_dam):
    st.markdown("---")
    st.markdown("### 🔍 Grad-CAM: Vùng Mà Model Đang Chú Ý")
    st.info(
        "💡 **Grad-CAM** hiển thị vùng ảnh mà model tập trung để đưa ra dự đoán. "
        "So sánh giữa 2 mô hình để thấy sự khác biệt trong cách nhìn của chúng."
    )

    available_models = ["DAM (Optimized)", "CE (Baseline)"] if model_ce else ["DAM (Optimized)"]
    model_col, _ = st.columns([1, 2])

    with model_col:
        def update_model_choice():
            st.session_state.gradcam_model_choice = st.session_state.temp_model_choice

        current_choice = st.session_state.gradcam_model_choice
        if current_choice not in available_models:
            current_choice = available_models[0]
            st.session_state.gradcam_model_choice = current_choice

        st.radio(
            "Chọn model để xem Grad-CAM:",
            available_models,
            index=available_models.index(current_choice),
            key="temp_model_choice",
            on_change=update_model_choice,
            horizontal=True,
        )

    selected_model = model_dam if "DAM" in st.session_state.gradcam_model_choice else model_ce

    with st.spinner(f"Đang tạo Grad-CAM cho {selected_disease_vn}..."):
        overlay, heatmap, _ = generate_gradcam_visualization(selected_model, image, image_tensor, class_idx=None)

        if overlay is None:
            st.error("Không thể tạo Grad-CAM. Vui lòng thử lại.")
            return

        display_gradcam_images(image, heatmap, overlay)
        prob_display = prob_dam if "DAM" in st.session_state.gradcam_model_choice else prob_ce
        st.caption(
            f"🎯 Grad-CAM từ model **{st.session_state.gradcam_model_choice}** "
            f"cho bệnh lý: **{selected_disease_vn}** "
            f"(Xác suất: {prob_display:.1%})"
        )


def show_resnet50_all_diseases(image_tensor, available_diseases_eng, threshold):
    progress_text = "Đang chạy dự đoán qua tất cả các mô hình..."
    progress_bar = st.progress(0, text=progress_text)

    results = []
    for i, disease_eng in enumerate(available_diseases_eng):
        disease_vn = get_vn_name(disease_eng)
        model_ce, model_dam = load_resnet50_models(disease_eng, PROJECT_ROOT)

        prob_ce = predict_single_disease(model_ce, image_tensor) if model_ce else 0.0
        prob_dam = predict_single_disease(model_dam, image_tensor) if model_dam else 0.0

        interpretation_dam, icon_dam, _ = interpret_probability(prob_dam, threshold)
        results.append(
            {
                "Bệnh lý": disease_vn,
                "DAM (Tối ưu)": prob_dam,
                "CE (Cơ sở)": prob_ce,
                "Đánh giá": f"{icon_dam} {interpretation_dam}",
            }
        )

        progress_bar.progress((i + 1) / len(available_diseases_eng), text=f"Đã xử lý: {disease_vn}")

    progress_bar.empty()
    display_resnet50_all_diseases_results(results, threshold)


def display_resnet50_all_diseases_results(results, threshold):
    high_risk = [row for row in results if "Khả năng cao" in row["Đánh giá"]]
    medium_risk = [row for row in results if "Cần theo dõi" in row["Đánh giá"]]

    if high_risk:
        st.error(f"⚠️ **Phát hiện {len(high_risk)} bệnh lý có khả năng cao:**")
        for row in high_risk:
            st.write(f"  • {row['Bệnh lý']}: {row['DAM (Tối ưu)']:.1%}")
    elif medium_risk:
        st.warning(f"⚠️ **Có {len(medium_risk)} bệnh lý cần theo dõi**")
    else:
        st.success("✅ **Tất cả các bệnh lý đều ở mức khả năng thấp**")

    df_results = pd.DataFrame(results).set_index("Bệnh lý")

    st.markdown("### 📊 Bảng Xác Suất Dự Đoán")
    st.dataframe(
        df_results.style.format(
            {
                "DAM (Tối ưu)": "{:.2%}",
                "CE (Cơ sở)": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### 📈 Biểu Đồ So Sánh Trực Quan")
    chart_data = df_results[["DAM (Tối ưu)", "CE (Cơ sở)"]]
    st.bar_chart(chart_data)

    st.success("Dự đoán đa nhãn hoàn tất!")
    st.info(
        f"Đã chạy ngầm tổng cộng **{len(results) * 2} mô hình** để tạo ra kết quả đa nhãn (Multi-label). "
        f"Ngưỡng phân loại: **{threshold:.0%}**"
    )


available_diseases_eng = discover_diseases(PROJECT_ROOT)
if not available_diseases_eng:
    st.sidebar.error(f"Không tìm thấy mô hình nào trong thư mục {RESNET50_DAM_DIR}. Vui lòng kiểm tra lại.")
    st.stop()

model_arch, mode, selected_disease_eng, selected_disease_vn, threshold, enable_gradcam, uploaded_file = show_sidebar(
    available_diseases_eng
)

show_page_title(model_arch, mode, selected_disease_vn)
show_medical_disclaimer()

if uploaded_file is None:
    st.info("Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu.")
else:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file)
        display_uploaded_image(image)

    with col2:
        st.subheader("So Sánh Mô Hình" if mode == "Một Bệnh Lý (So sánh chi tiết)" else "Kết Quả Đa Nhãn")

        if st.button("Chạy Dự Đoán", type="primary"):
            image_tensor = preprocess_image(image)

            if model_arch == "DenseNet121 (Multi-label)":
                show_densenet121_prediction(image, image_tensor, threshold, enable_gradcam)
            elif mode == "Một Bệnh Lý (So sánh chi tiết)":
                show_single_disease_prediction(
                    image,
                    image_tensor,
                    selected_disease_eng,
                    selected_disease_vn,
                    threshold,
                    enable_gradcam,
                )
            elif mode == "Đa Nhãn (Tất cả bệnh lý)":
                show_resnet50_all_diseases(image_tensor, available_diseases_eng, threshold)
