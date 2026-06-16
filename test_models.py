"""
Script để test các models trước khi chạy GUI
"""
import os
import sys
from model_utils import (
    get_resnet50_model,
    get_densenet121_model,
    load_weights,
    discover_diseases
)

def test_model_loading():
    """Test loading các models"""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    print("=" * 60)
    print("KIỂM TRA CẤU TRÚC THỨ MỤC")
    print("=" * 60)
    
    # Test discover diseases
    diseases = discover_diseases(base_path)
    print(f"\n✓ Tìm thấy {len(diseases)} bệnh lý:")
    for d in diseases:
        print(f"  - {d}")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA RESNET50 MODELS")
    print("=" * 60)
    
    # Test ResNet50 models
    for disease in diseases:
        dam_path = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "DAM", f"{disease}_latest.pth")
        ce_path = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "CE", f"CE_{disease}.pth")
        
        print(f"\n{disease}:")
        print(f"  DAM: {'✓' if os.path.exists(dam_path) else '✗'} {dam_path}")
        print(f"  CE:  {'✓' if os.path.exists(ce_path) else '✗'} {ce_path}")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA DENSENET121 MODEL")
    print("=" * 60)
    
    densenet_path = os.path.join(base_path, "CheXpert_DAM", "Densenet121", "best_densenet_exp1.pth")
    print(f"\nDenseNet121: {'✓' if os.path.exists(densenet_path) else '✗'} {densenet_path}")
    
    # Try loading DenseNet121
    if os.path.exists(densenet_path):
        try:
            print("\nĐang thử load DenseNet121...")
            model = get_densenet121_model(num_classes=5, use_dropout=True)
            model = load_weights(model, densenet_path)
            print("✓ Load DenseNet121 thành công!")
            print(f"  - Model architecture: {model.__class__.__name__}")
            print(f"  - Classifier: {model.classifier}")
        except Exception as e:
            print(f"✗ Lỗi khi load DenseNet121: {e}")
    
    # Try loading một ResNet50 model
    if diseases:
        disease = diseases[0]
        dam_path = os.path.join(base_path, "CheXpert_DAM", "Resnet50", "DAM", f"{disease}_latest.pth")
        if os.path.exists(dam_path):
            try:
                print(f"\nĐang thử load ResNet50 DAM ({disease})...")
                model = get_resnet50_model(num_classes=1)
                model = load_weights(model, dam_path)
                print(f"✓ Load ResNet50 thành công!")
                print(f"  - Model architecture: {model.__class__.__name__}")
                print(f"  - FC output: {model.fc.out_features} classes")
            except Exception as e:
                print(f"✗ Lỗi khi load ResNet50: {e}")
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT KIỂM TRA")
    print("=" * 60)

if __name__ == "__main__":
    test_model_loading()
