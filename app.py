import streamlit as st
import os
from PIL import Image
from model_utils import discover_diseases, get_resnet50_model, load_weights, preprocess_image, predict

st.set_page_config(page_title="CheXpert DAM vs CE Demo", layout="wide", page_icon="🫁")

# Define the root directory of the project
# This assumes the app is run from inside CheXpert_GUI or the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# UI: Sidebar
st.sidebar.title("🫁 CheXpert Demo")
st.sidebar.markdown("Compare ResNet50 models trained with Deep AUC Maximization (DAM) and standard Cross-Entropy (CE).")

# Dynamically discover diseases
available_diseases = discover_diseases(PROJECT_ROOT)

if not available_diseases:
    st.sidebar.error(f"No models found in {os.path.join(PROJECT_ROOT, 'CheXpert_DAM')}. Please ensure the models exist.")
    st.stop()

selected_disease = st.sidebar.selectbox("Select Disease to Detect:", available_diseases)

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

# UI: Main Content
st.title(f"Disease Detection: {selected_disease}")
st.write("Upload an image in the sidebar and click **Run Inference** to compare the CE and DAM models.")

# Caching models to avoid reloading them on every interaction
@st.cache_resource
def load_models(disease, base_path):
    # Paths
    dam_path = os.path.join(base_path, "CheXpert_DAM", "DAM", f"{disease}_latest.pth")
    # CE model naming convention has a space instead of underscore for Pleural Effusion? Let's check dynamically.
    # The user directory had "CE_Pleural Effusion.pth". Let's handle this.
    ce_path_1 = os.path.join(base_path, "CheXpert_DAM", "CE", f"CE_{disease}.pth")
    ce_path_2 = os.path.join(base_path, "CheXpert_DAM", "CE", f"CE_{disease.replace('_', ' ')}.pth")
    
    ce_path = ce_path_1 if os.path.exists(ce_path_1) else ce_path_2
    
    model_ce = get_resnet50_model()
    model_dam = get_resnet50_model()
    
    try:
        model_ce = load_weights(model_ce, ce_path)
    except Exception as e:
        st.warning(f"Could not load CE model for {disease}: {e}")
        model_ce = None
        
    try:
        model_dam = load_weights(model_dam, dam_path)
    except Exception as e:
        st.warning(f"Could not load DAM model for {disease}: {e}")
        model_dam = None
        
    return model_ce, model_dam


if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Chest X-Ray", use_container_width=True)
        
    with col2:
        st.subheader("Model Comparison")
        if st.button("Run Inference", type="primary"):
            with st.spinner("Loading models and running inference..."):
                model_ce, model_dam = load_models(selected_disease, PROJECT_ROOT)
                
                if model_ce is None and model_dam is None:
                    st.error("Both models failed to load. Please check your model files.")
                else:
                    # Preprocess
                    img_tensor = preprocess_image(image)
                    
                    # Run Inference
                    res_cols = st.columns(2)
                    
                    with res_cols[0]:
                        st.markdown("### 🟦 Baseline (CE)")
                        if model_ce:
                            prob_ce = predict(model_ce, img_tensor)
                            st.metric(label="Probability", value=f"{prob_ce:.2%}")
                            st.progress(prob_ce)
                        else:
                            st.write("Model not available.")
                            
                    with res_cols[1]:
                        st.markdown("### 🟧 Optimized (DAM)")
                        if model_dam:
                            prob_dam = predict(model_dam, img_tensor)
                            st.metric(label="Probability", value=f"{prob_dam:.2%}")
                            st.progress(prob_dam)
                        else:
                            st.write("Model not available.")
                            
                    st.success("Inference complete!")
                    st.info("Note: The DAM model is optimized for Area Under the ROC Curve, which often leads to a better ranking of probabilities compared to standard Cross-Entropy.")
else:
    st.info("Please upload an image from the sidebar to begin.")
