# Demo phân loại ảnh X-quang ngực CheXpert

## Giới thiệu ngắn gọn

Đây là ứng dụng demo dùng Streamlit để phân loại ảnh X-quang ngực dựa trên các mô hình học sâu đã huấn luyện với dữ liệu CheXpert.

Ứng dụng hỗ trợ so sánh ResNet50 dùng CE và DAM, dự đoán đa nhãn bằng DenseNet121, điều chỉnh ngưỡng xác suất, và hiển thị Grad-CAM để xem vùng ảnh mà model chú ý. Kết quả chỉ dùng cho học tập và minh họa, không phải chẩn đoán y khoa.

## Chức năng chính

- Tải ảnh X-quang ngực lên ứng dụng. Ảnh nên có định dạng JPG, JPEG hoặc PNG.
- Dự đoán xác suất mắc bệnh cho ảnh đã tải lên. Kết quả được hiển thị dưới dạng phần trăm để dễ đọc.
- So sánh hai mô hình ResNet50 CE và DAM cho từng bệnh lý. CE là mô hình baseline, DAM là mô hình tối ưu hơn trong project.
- Dự đoán nhiều bệnh cùng lúc bằng DenseNet121. Chế độ này trả về xác suất cho 5 bệnh lý.
- Điều chỉnh ngưỡng xác suất. Nếu xác suất lớn hơn hoặc bằng ngưỡng, hệ thống đánh giá là khả năng cao.
- Xem Grad-CAM heatmap. Người dùng bấm nút `Tạo Grad-CAM` sau khi dự đoán để tạo bản đồ nhiệt.
- Hiển thị bảng kết quả và biểu đồ để so sánh xác suất giữa các bệnh hoặc giữa các mô hình.

## Công nghệ sử dụng

- Python
- Streamlit
- PyTorch
- Torchvision
- OpenCV
- PIL / Pillow
- NumPy
- Pandas
- ResNet50
- DenseNet121
- Grad-CAM

## Yêu cầu trước khi chạy

- Python 3.10 hoặc mới hơn.
- Git để tải project về máy.
- pip để cài thư viện Python.
- Các file trọng số model `.pth` phải được đặt đúng thư mục.
- Máy CPU bình thường có thể chạy demo, nhưng lần tải model đầu tiên và Grad-CAM có thể chậm.

## Cách tải dự án về máy

Nếu có repository Git, tải project bằng lệnh:

```bash
git clone <repository-url>
cd <project-folder>
```

Sau đó vào thư mục giao diện:

```bash
cd CheXpert_GUI
```

Nếu đã có sẵn thư mục project trên máy, chỉ cần mở terminal tại thư mục `CheXpert_GUI`.

## Tạo môi trường ảo

Trên Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Trên macOS hoặc Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Cài thư viện

Chạy lệnh sau trong thư mục `CheXpert_GUI`:

```bash
pip install -r requirements.txt
```

Nếu thiếu thư viện nào đó, hãy kiểm tra lại môi trường ảo đã được kích hoạt chưa rồi cài lại requirements.

## Chuẩn bị model

Các file trọng số model cần nằm trong thư mục `CheXpert_DAM`, đặt cùng cấp với thư mục `CheXpert_GUI`.

Các thư mục model cần kiểm tra:

- `CheXpert_DAM/Resnet50/DAM/`
- `CheXpert_DAM/Resnet50/CE/`
- `CheXpert_DAM/Densenet121/`

File DAM của ResNet50 dùng cho mô hình tối ưu theo từng bệnh, ví dụ `Cardiomegaly_latest.pth`. File CE của ResNet50 dùng làm baseline, ví dụ `CE_Cardiomegaly.pth`. File DenseNet121 dùng cho dự đoán đa nhãn, ví dụ `best_densenet_exp1.pth`.

Nếu muốn đặt model ở vị trí khác, có thể khai báo biến môi trường `CHEXPERT_PROJECT_ROOT` trỏ đến thư mục chứa `CheXpert_DAM`.

## Chạy chương trình

Chạy lệnh sau trong thư mục `CheXpert_GUI`:

```bash
streamlit run app.py
```

Sau khi chạy, Streamlit sẽ hiển thị địa chỉ local. Mở địa chỉ đó trong trình duyệt để dùng ứng dụng.

## Cách sử dụng ứng dụng

1. Mở ứng dụng trong trình duyệt.
2. Chọn loại mô hình ở thanh bên trái: ResNet50 hoặc DenseNet121.
3. Nếu dùng ResNet50, chọn chế độ một bệnh hoặc nhiều bệnh.
4. Tải ảnh X-quang ngực định dạng JPG/PNG.
5. Điều chỉnh ngưỡng xác suất nếu cần.
6. Bấm `Chạy Dự Đoán`.
7. Xem xác suất, bảng kết quả và biểu đồ.
8. Nếu muốn xem vùng model chú ý, chọn bệnh/model cần xem rồi bấm `Tạo Grad-CAM`.
9. Nếu dùng ResNet50 một bệnh, so sánh kết quả giữa CE và DAM.

## Giải thích ngắn các phần chính trong code

- `app.py` là file chạy giao diện Streamlit và kết nối các chức năng chính.
- `src/image_processing.py` xử lý ảnh đầu vào trước khi đưa vào model.
- `src/model_loader.py` tạo model và tải trọng số đã huấn luyện.
- `src/prediction.py` chạy dự đoán và chuyển output thành xác suất.
- `src/gradcam.py` tạo bản đồ nhiệt Grad-CAM.
- `src/disease_info.py` lưu tên bệnh và tên hiển thị tiếng Việt.
- `src/config.py` lưu đường dẫn và cấu hình chung.
- `src/ui_components.py` chứa các hàm hiển thị giao diện được dùng lại trong app.

## Một số lỗi thường gặp

### Không chạy được `streamlit`

Cài lại Streamlit hoặc cài lại toàn bộ thư viện:

```bash
pip install streamlit
pip install -r requirements.txt
```

### Không tìm thấy model

Kiểm tra thư mục `CheXpert_DAM` có nằm cùng cấp với `CheXpert_GUI` không. Sau đó kiểm tra các file `.pth` trong `Resnet50/DAM`, `Resnet50/CE` và `Densenet121`.

### Upload ảnh bị lỗi

Hãy dùng file JPG, JPEG hoặc PNG hợp lệ. Nếu ảnh bị hỏng hoặc không đúng định dạng, ứng dụng sẽ không đọc được.

### Dự đoán chậm

Lần đầu chạy có thể chậm vì ứng dụng phải tải model từ file `.pth`. Nếu chạy bằng CPU, tốc độ sẽ chậm hơn GPU.

### Grad-CAM chạy lâu

Grad-CAM cần chạy thêm forward pass và backward pass để lấy gradient, nên sẽ lâu hơn dự đoán thông thường.

## Lưu ý y tế

Project này chỉ dùng cho mục đích học tập và trình diễn trong môn học. Kết quả dự đoán của AI không phải là chẩn đoán y khoa. Mọi quyết định khám, chẩn đoán hoặc điều trị phải do bác sĩ có chuyên môn thực hiện.

## Ghi chú cho sinh viên bảo vệ

Sinh viên nên hiểu các ý chính sau trước khi bảo vệ:

- Ảnh được resize về `224x224` vì ResNet50 và DenseNet121 trong project nhận input kích thước này.
- Ảnh X-quang có thể là grayscale, nhưng model dùng 3 kênh nên ảnh được chuyển sang RGB.
- ImageNet normalization được dùng vì kiến trúc model theo chuẩn input của các model torchvision.
- Sigmoid được dùng cho bài toán multi-label vì mỗi bệnh có thể xuất hiện độc lập.
- Softmax có thể được dùng với CE output 2 class để lấy xác suất class dương.
- Threshold là ngưỡng để chuyển xác suất thành đánh giá khả năng thấp/cần theo dõi/khả năng cao.
- Grad-CAM cho biết vùng ảnh đóng góp nhiều vào dự đoán của model; vùng đỏ/vàng thường quan trọng hơn.
- Khi đổi bệnh trong ResNet50, ứng dụng có thể cần tải model khác vì mỗi bệnh có file trọng số riêng.
