# CheXpert GUI - Hướng Dẫn Sử Dụng

## Tổng Quan
Ứng dụng Streamlit để demo các mô hình Deep Learning cho phân loại bệnh lý X-quang ngực từ dataset CheXpert.

## Cấu Trúc Thư Mục

```
CheXpert_Lite/
├── CheXpert_DAM/
│   ├── Densenet121/
│   │   └── best_densenet.pth          # Model DenseNet121 đa nhãn (5 bệnh)
│   └── Resnet50/
│       ├── DAM/                        # Models ResNet50 huấn luyện bằng Deep AUC Maximization
│       │   ├── Atelectasis_latest.pth
│       │   ├── Cardiomegaly_latest.pth
│       │   ├── Consolidation_latest.pth
│       │   ├── Edema_latest.pth
│       │   └── Pleural_Effusion_latest.pth
│       └── CE/                         # Models ResNet50 huấn luyện bằng Cross-Entropy
│           ├── CE_Atelectasis.pth
│           ├── CE_Cardiomegaly.pth
│           ├── CE_Consolidation.pth
│           ├── CE_Edema.pth
│           └── CE_Pleural_Effusion.pth
└── CheXpert_GUI/
    ├── app.py                          # Main Streamlit app
    ├── model_utils.py                  # Model utilities
    └── requirements.txt                # Dependencies
```

## Cài Đặt

### 1. Cài đặt dependencies:
```bash
cd CheXpert_GUI
pip install -r requirements.txt
```

### 2. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Tính Năng

### 1. ResNet50 (DAM vs CE)
So sánh hiệu suất giữa 2 phương pháp huấn luyện:
- **DAM (Deep AUC Maximization)**: Tối ưu hóa trực tiếp AUC metric
- **CE (Cross-Entropy)**: Phương pháp huấn luyện truyền thống

**Chế độ:**
- **Một Bệnh Lý**: So sánh chi tiết DAM vs CE cho 1 bệnh cụ thể
- **Đa Nhãn**: Dự đoán tất cả 5 bệnh lý bằng cách chạy 10 models (5 DAM + 5 CE)

### 2. DenseNet121 (Multi-label)
Model duy nhất được huấn luyện để dự đoán đồng thời **5 bệnh lý**:
- Cardiomegaly (Tim to)
- Edema (Phù phổi)
- Consolidation (Đông đặc phổi)
- Atelectasis (Xẹp phổi)
- Pleural Effusion (Tràn dịch màng phổi)

**Ưu điểm:**
- Nhanh hơn (1 model thay vì 5-10 models)
- Hiệu quả hơn về tài nguyên
- Có thể học được mối quan hệ giữa các bệnh lý

## Sử Dụng

1. **Chọn Kiến Trúc Mô Hình:**
   - ResNet50 (DAM vs CE): So sánh 2 phương pháp huấn luyện
   - DenseNet121 (Multi-label): Dự đoán nhanh đa nhãn

2. **Tải Ảnh X-quang:**
   - Định dạng: JPG, JPEG, PNG
   - Ảnh sẽ tự động được resize về 224x224

3. **Nhấn "Chạy Dự Đoán":**
   - Xem kết quả xác suất cho từng bệnh lý
   - Biểu đồ trực quan hóa

## Thông Tin Kỹ Thuật

### ResNet50
- **Input**: RGB image 224x224
- **Output**: 1 class (binary classification cho từng bệnh)
- **Activation**: Sigmoid
- **Số models**: 10 (5 DAM + 5 CE)

### DenseNet121
- **Input**: RGB image 224x224
- **Output**: 5 classes (multi-label classification)
- **Activation**: Sigmoid (cho từng class độc lập)
- **Số models**: 1

## Requirements
- Python 3.8+
- PyTorch 1.12+
- Streamlit
- torchvision
- pandas
- pillow

## Lưu Ý
- Model paths được tự động phát hiện từ cấu trúc thư mục
- Nếu thiếu model, ứng dụng sẽ hiển thị cảnh báo
- DenseNet121 sử dụng Mixed Policy cho xử lý nhãn uncertain (-1.0)

## Tác Giả & Tài Liệu Tham Khảo
- CheXpert Dataset: https://stanfordmlgroup.github.io/competitions/chexpert/
- Deep AUC Maximization: https://arxiv.org/abs/2012.03173
- Notebook training: `D:\CheXpert_Lite\notebook\densenet121\chexpert_densenet.ipynb`
