"""
Test script for Grad-CAM functionality
"""
import os
import sys
from PIL import Image
import torch
from model_utils import (
    get_densenet121_model,
    load_weights,
    preprocess_image,
    generate_gradcam_visualization
)

def test_gradcam():
    print("=" * 60)
    print("KIỂM TRA GRAD-CAM")
    print("=" * 60)
    
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Load model
    print("\n1. Đang load DenseNet121 model...")
    model_path = os.path.join(base_path, "CheXpert_DAM", "Densenet121", "best_densenet_exp1.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        return
    
    try:
        model = get_densenet121_model(num_classes=5, use_dropout=True)
        model = load_weights(model, model_path)
        print("✅ Load model thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return
    
    # Load test image
    print("\n2. Đang tìm ảnh test...")
    
    # Try to find a sample image from train folder
    train_folder = os.path.join(base_path, "train")
    test_image_path = None
    
    if os.path.exists(train_folder):
        # Find first patient folder
        for patient_folder in os.listdir(train_folder):
            patient_path = os.path.join(train_folder, patient_folder)
            if os.path.isdir(patient_path):
                # Find first study
                for study_folder in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study_folder)
                    if os.path.isdir(study_path):
                        # Find first image
                        for img_file in os.listdir(study_path):
                            if img_file.endswith('.jpg'):
                                test_image_path = os.path.join(study_path, img_file)
                                break
                    if test_image_path:
                        break
            if test_image_path:
                break
    
    if not test_image_path:
        print("❌ Không tìm thấy ảnh test trong thư mục train")
        print("💡 Vui lòng đặt một ảnh test vào thư mục CheXpert_GUI với tên 'test_xray.jpg'")
        return
    
    print(f"✅ Tìm thấy ảnh test: {test_image_path}")
    
    # Load and preprocess image
    print("\n3. Đang xử lý ảnh...")
    try:
        image = Image.open(test_image_path).convert('RGB')
        image_tensor = preprocess_image(image)
        print("✅ Xử lý ảnh thành công!")
        print(f"   - Kích thước ảnh gốc: {image.size}")
        print(f"   - Shape tensor: {image_tensor.shape}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh: {e}")
        return
    
    # Test Grad-CAM generation
    print("\n4. Đang tạo Grad-CAM...")
    
    disease_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    
    for idx, disease in enumerate(disease_names):
        print(f"\n   Testing {disease} (class {idx})...")
        try:
            overlay, heatmap, cam = generate_gradcam_visualization(
                model, image, image_tensor, class_idx=idx
            )
            
            if overlay is not None:
                print(f"   ✅ Grad-CAM cho {disease} tạo thành công!")
                print(f"      - CAM shape: {cam.shape}")
                print(f"      - CAM min: {cam.min():.4f}, max: {cam.max():.4f}")
                
                # Optionally save the results
                output_dir = os.path.join(os.path.dirname(__file__), "gradcam_test_output")
                os.makedirs(output_dir, exist_ok=True)
                
                overlay_path = os.path.join(output_dir, f"overlay_{disease}.jpg")
                heatmap_path = os.path.join(output_dir, f"heatmap_{disease}.jpg")
                
                overlay.save(overlay_path)
                heatmap.save(heatmap_path)
                
                print(f"      - Đã lưu: {overlay_path}")
                print(f"      - Đã lưu: {heatmap_path}")
            else:
                print(f"   ❌ Không thể tạo Grad-CAM cho {disease}")
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT KIỂM TRA GRAD-CAM")
    print("=" * 60)
    print(f"\nKiểm tra kết quả trong thư mục: gradcam_test_output/")

if __name__ == "__main__":
    test_gradcam()
