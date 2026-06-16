# Hướng Dẫn Về Ngưỡng Phân Loại (Threshold)

## 1. Ngưỡng Là Gì?

**Ngưỡng (Threshold)** là giá trị xác suất dùng để quyết định liệu một bệnh lý có "hiện diện" hay không dựa trên output của mô hình AI.

- **Output của model**: Xác suất từ 0.0 đến 1.0 (0% đến 100%)
- **Ngưỡng mặc định**: 0.5 (50%)
- **Quy tắc**: 
  - Xác suất ≥ Ngưỡng → "Khả năng cao"
  - 0.7 × Ngưỡng ≤ Xác suất < Ngưỡng → "Cần theo dõi"
  - Xác suất < 0.7 × Ngưỡng → "Khả năng thấp"

## 2. Tại Sao Không Có Ngưỡng Chuẩn?

Theo nghiên cứu khoa học về CheXpert và Deep AUC Maximization:

### 2.1. Mục Tiêu Của Mô Hình
- **Không phải**: Tự động chẩn đoán bệnh
- **Mà là**: Cung cấp xác suất để **hỗ trợ bác sĩ** ra quyết định

### 2.2. Metric Đánh Giá
Các mô hình được đánh giá bằng **AUC (Area Under ROC Curve)**, không phải accuracy với ngưỡng cố định:

- AUC đo khả năng phân biệt giữa bệnh/không bệnh
- AUC không phụ thuộc vào ngưỡng cụ thể
- AUC thích hợp cho dữ liệu mất cân bằng (imbalanced data)

### 2.3. Bối Cảnh Y Tế
Ngưỡng tối ưu phụ thuộc vào:
- **Mục đích sử dụng**: Screening (tầm soát) hay Diagnosis (chẩn đoán)?
- **Độ nghiêm trọng của bệnh**: Bệnh nguy hiểm cần ngưỡng thấp hơn
- **Chi phí**: Cân nhắc giữa False Positive và False Negative
- **Dân số**: Tỷ lệ bệnh trong cộng đồng

## 3. Cách Chọn Ngưỡng

### 3.1. Screening (Tầm Soát)
**Mục tiêu**: Không bỏ sót bệnh nhân → Ưu tiên Sensitivity cao

```
Ngưỡng khuyến nghị: 0.3 - 0.4
```

**Ưu điểm**:
- Phát hiện được nhiều ca bệnh
- Giảm False Negative (bỏ sót bệnh)

**Nhược điểm**:
- Tăng False Positive (báo động giả)
- Chi phí xét nghiệm bổ sung cao hơn

### 3.2. Diagnosis (Chẩn Đoán Xác Định)
**Mục tiêu**: Chính xác cao → Cân bằng Sensitivity và Specificity

```
Ngưỡng khuyến nghị: 0.5 - 0.6
```

**Ưu điểm**:
- Cân bằng tốt giữa False Positive và False Negative
- Giảm chi phí điều trị không cần thiết

**Nhược điểm**:
- Có thể bỏ sót một số ca bệnh nhẹ

### 3.3. Conservative (Bảo Thủ)
**Mục tiêu**: Chỉ cảnh báo khi rất chắc chắn → Ưu tiên Specificity cao

```
Ngưỡng khuyến nghị: 0.7 - 0.8
```

**Ưu điểm**:
- Giảm False Positive
- Tăng độ tin cậy của kết quả dương tính

**Nhược điểm**:
- Bỏ sót nhiều ca bệnh hơn
- Không phù hợp cho screening

## 4. Bảng Tham Khảo Ngưỡng Theo Bệnh

| Bệnh lý | Screening | Standard | Conservative |
|---------|-----------|----------|--------------|
| **Cardiomegaly** (Tim to) | 0.35 | 0.50 | 0.70 |
| **Edema** (Phù phổi) | 0.30 | 0.45 | 0.65 |
| **Consolidation** (Đông đặc) | 0.40 | 0.55 | 0.75 |
| **Atelectasis** (Xẹp phổi) | 0.35 | 0.50 | 0.70 |
| **Pleural Effusion** (Tràn dịch) | 0.40 | 0.55 | 0.70 |

**Lưu ý**: Đây là ngưỡng gợi ý, cần được xác thực bởi bác sĩ chuyên khoa.

## 5. ROC Curve và Optimal Threshold

### 5.1. Youden Index Method
Ngưỡng tối ưu = điểm trên ROC curve có:
```
J = Sensitivity + Specificity - 1 (maximum)
```

### 5.2. F1-Score Method
Tối ưu hóa cân bằng giữa Precision và Recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 5.3. Cost-Based Method
Tính toán dựa trên chi phí thực tế:
```
Cost = (FP × Cost_FP) + (FN × Cost_FN)
```

## 6. Ví Dụ Thực Tế

### Ví dụ 1: Xác suất = 0.45, Ngưỡng = 0.5
- **Kết quả**: Khả năng thấp (< 0.5)
- **Hành động**: Theo dõi, không cần xét nghiệm bổ sung ngay
- **Lưu ý**: Nếu có triệu chứng lâm sàng → Vẫn cần khám bác sĩ

### Ví dụ 2: Xác suất = 0.72, Ngưỡng = 0.5
- **Kết quả**: Khả năng cao (≥ 0.5)
- **Hành động**: Cần khám bác sĩ chuyên khoa
- **Lưu ý**: Có thể cần X-quang lại, CT scan, hoặc các xét nghiệm khác

### Ví dụ 3: Xác suất = 0.38, Ngưỡng = 0.3 (Screening)
- **Kết quả**: Khả năng cao (≥ 0.3)
- **Hành động**: Chụp X-quang lại hoặc xét nghiệm bổ sung
- **Lý do**: Trong screening, ngưỡng thấp để không bỏ sót

## 7. LƯU Ý QUAN TRỌNG

### ⚠️ Disclaimer Y Tế
```
KẾT QUẢ TỪ AI CHỈ LÀ HỖ TRỢ THAM KHẢO

✓ AI có thể phát hiện pattern mà mắt thường bỏ sót
✓ AI có thể xử lý lượng lớn ảnh nhanh chóng

✗ AI KHÔNG thay thế bác sĩ chuyên khoa
✗ AI có thể sai với các ca bệnh hiếm, ảnh chất lượng kém
✗ AI không có kiến thức lâm sàng tổng thể về bệnh nhân
```

### 🏥 Quy Trình Chuẩn
1. **AI phân tích** → Đưa ra xác suất
2. **Bác sĩ xem xét** → Kết hợp với khám lâm sàng
3. **Xét nghiệm bổ sung** → Nếu cần thiết
4. **Chẩn đoán cuối cùng** → Do bác sĩ quyết định
5. **Điều trị** → Theo phác đồ chuẩn

### 📊 Khi Nào Cần Điều Chỉnh Ngưỡng?

**Tăng ngưỡng (0.6 - 0.8)** khi:
- Có nhiều false alarm (báo động giả)
- Chi phí xét nghiệm bổ sung quá cao
- Muốn tập trung vào ca bệnh rõ ràng

**Giảm ngưỡng (0.3 - 0.4)** khi:
- Đang tầm soát diện rộng
- Bệnh rất nguy hiểm, không được bỏ sót
- Chi phí xét nghiệm bổ sung chấp nhận được

## 8. Tài Liệu Tham Khảo

1. **CheXpert Dataset Paper** (Irvin et al., 2019)
   - "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels"
   - Không đề cập ngưỡng cụ thể, tập trung vào AUC

2. **Deep AUC Maximization** (Yuan et al., 2021)
   - "Large-scale Robust Deep AUC Maximization"
   - Tối ưu hóa AUC, không phải accuracy tại ngưỡng cố định

3. **ROC Analysis in Medical Research**
   - Youden Index cho optimal cutoff
   - Cost-sensitive threshold selection

## 9. Công Cụ Hỗ Trợ Trong GUI

### Slider Ngưỡng
- **Range**: 0.0 - 1.0
- **Step**: 0.05
- **Default**: 0.5
- **Real-time**: Thay đổi ngay khi kéo

### Màu Sắc Phân Loại
- 🔴 **Đỏ**: Khả năng cao (≥ threshold)
- 🟡 **Vàng**: Cần theo dõi (70% - 100% của threshold)
- 🟢 **Xanh**: Khả năng thấp (< 70% của threshold)

### Thống Kê Tổng Hợp
- Số bệnh lý "Khả năng cao"
- Số bệnh lý "Cần theo dõi"
- So sánh giữa các model (DAM vs CE)

## 10. FAQ

### Q1: Tại sao có 3 mức phân loại thay vì 2?
**A**: Để cung cấp thông tin chi tiết hơn:
- Khả năng cao: Cần hành động ngay
- Cần theo dõi: Theo dõi sát, xem xét triệu chứng
- Khả năng thấp: Yên tâm nhưng vẫn cần khám định kỳ

### Q2: Model DAM và CE có ngưỡng khác nhau không?
**A**: Không, cùng một ngưỡng. Nhưng do được tối ưu hóa khác nhau:
- DAM: Tối ưu AUC → Phân tách tốt hơn ở nhiều ngưỡng
- CE: Tối ưu Cross-Entropy → Có thể cần ngưỡng khác để đạt cùng performance

### Q3: DenseNet121 multi-label dùng ngưỡng như thế nào?
**A**: Áp dụng **độc lập** cho từng bệnh lý:
- Mỗi bệnh có xác suất riêng
- Ngưỡng giống nhau cho tất cả bệnh
- Có thể có nhiều bệnh cùng "Khả năng cao"

### Q4: Làm sao biết ngưỡng nào là tốt nhất?
**A**: Cần:
1. Dữ liệu validation với ground truth từ bác sĩ
2. Vẽ ROC curve
3. Tính toán metrics (Sensitivity, Specificity, F1) tại các ngưỡng
4. Tham khảo ý kiến bác sĩ chuyên khoa
5. Pilot test trong môi trường thực tế

---

**Cập nhật**: December 2024  
**Tác giả**: CheXpert GUI Development Team  
**Contact**: Liên hệ bác sĩ chuyên khoa X-quang để được tư vấn về ngưỡng phù hợp
