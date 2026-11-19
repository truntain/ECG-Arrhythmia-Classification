# Phân tích tín hiệu điện tâm đồ (ECG) và xây dựng mô hình học máy tự động phân loại các dạng rối loạn nhịp tim

**Môn học:** Nhập môn Học máy  
**Nhóm thực hiện:** Group 27  

## 1. Danh sách thành viên & Phân công công việc

| STT | Họ và tên | MSSV | Thư mục làm việc chính | Nhiệm vụ chi tiết |
| :-: | :--- | :---: | :--- | :--- |
| 1 | **Trương Thế Thành** | 20235835 | `data/`, `src/preprocessing/` | Tải dữ liệu MIT-BIH (`wfdb`), lọc nhiễu tín hiệu ECG, phát hiện đỉnh R, cắt nhịp (beat segmentation) và gom 5 nhãn chuẩn AAMI. |
| 2 | **Nguyễn Sỹ Vinh** | 20235875 | `src/features/` | Trích xuất đặc trưng hình thái QRS, khoảng R-R, biến đổi Wavelet (`PyWavelets`), giảm chiều `PCA` và cân bằng dữ liệu (`SMOTE`). |
| 3 | **Nguyễn Phạm Anh Tuấn** | 20235863 | `src/models/` | Xây dựng pipeline huấn luyện, tinh chỉnh siêu tham số và đánh giá mô hình **k-NN** và **SVM**. |
| 4 | **Lê Minh Trọng** | 20239675 | `src/models/` | Xây dựng pipeline huấn luyện, tinh chỉnh cho **Random Forest** và **XGBoost**, tổng hợp bảng so sánh 4 mô hình và vẽ Confusion Matrix. |
| 5 | **Đào Trọng Công** | 20235665 | `app/` | Xây dựng giao diện trực quan hóa sóng ECG bằng `Streamlit`, tích hợp model `.joblib` để phân loại và phát cảnh báo nhịp bất thường. |

---

## 2. Cấu trúc thư mục dự án (Repository Structure)

```text
ECG-Arrhythmia-Classification/
├── app/                        # [Người 5 - Đào Trọng Công] Giao diện Streamlit Web UI
│   └── main.py
├── data/                       # [Người 1 - Trương Thế Thành] Script tải và lưu dữ liệu
│   ├── download_mitbih.py      # Script tự động tải 48 bản ghi MIT-BIH từ PhysioNet
│   ├── raw/                    # Chứa file gốc (.dat, .hea, .atr) - đã thêm vào .gitignore
│   └── processed/              # Chứa dữ liệu đã cắt nhịp / trích xuất đặc trưng (.npz, .csv)
├── results/                    # Lưu biểu đồ, Confusion Matrix, bảng so sánh thực nghiệm
├── saved_models/               # Lưu trọng số mô hình (.joblib, .pkl) + Scaler + PCA
├── src/
│   ├── config.py               # Hằng số chung (tần số 360Hz, ánh xạ 5 lớp AAMI, đường dẫn)
│   ├── preprocessing/          # [Người 1 - Trương Thế Thành] Lọc nhiễu, cắt nhịp, gán nhãn AAMI
│   ├── features/               # [Người 2 - Nguyễn Sỹ Vinh] Đặc trưng QRS, RR, Wavelet, PCA, SMOTE
│   └── models/                 # [Người 3 & 4 - Tuấn & Trọng] Huấn luyện k-NN, SVM, RF, XGBoost
├── overview.md                 # Tài liệu kỹ thuật chi tiết & khung Báo cáo tiến độ 3 trang
├── requirements.txt            # Danh sách thư viện Python chuẩn cho cả nhóm
├── .gitignore                  # Bỏ qua file dữ liệu nặng (.dat, .npz, .csv) và môi trường ảo
└── README.md                   # Hướng dẫn cài đặt và quy ước làm việc nhóm
```

---

## 3. Hướng dẫn cài đặt môi trường (Cho cả 5 thành viên)

### Bước 1: Clone repository về máy
```bash
git clone https://github.com/<username>/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị Python 3.10+)
* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện chuẩn
```bash
pip install -r requirements.txt
```

### Bước 4: Tải bộ dữ liệu MIT-BIH Arrhythmia Database
```bash
python data/download_mitbih.py
```

### Bước 5: Chạy thử giao diện Streamlit
```bash
streamlit run app/main.py
```

---

## 4. Quy ước giao tiếp dữ liệu giữa các thành viên (Data Contract)

Để 5 thành viên có thể code song song mà không cần chờ đợi nhau, định dạng đầu ra của mỗi module được thống nhất như sau:
1. **Output của Người 1 (`src/preprocessing/`) $\rightarrow$ Input của Người 2 (`src/features/`):**
   * File `data/processed/segmented_beats.npz` gồm:
     * `beats`: Mảng numpy kích thước `(N_samples, 250)` (tín hiệu ECG đã lọc nhiễu quanh đỉnh R: `[-90, +160]` mẫu).
     * `rr_features`: Mảng numpy kích thước `(N_samples, 4)` (`[pre_rr, post_rr, local_rr, ratio_rr]`).
     * `labels`: Mảng numpy kích thước `(N_samples,)` chứa nhãn số `{0: N, 1: S, 2: V, 3: F, 4: Q}`.
     * `record_ids`: Mảng chứa mã bản ghi của từng nhịp.
2. **Output của Người 2 (`src/features/`) $\rightarrow$ Input của Người 3 & 4 (`src/models/`):**
   * File `data/processed/train_test_features.npz` gồm:
     * `X_train`, `y_train` (đã qua `StandardScaler`, `PCA` và cân bằng `SMOTE`).
     * `X_test`, `y_test` (đã qua `StandardScaler` và `PCA`, giữ nguyên phân bố thực tế).
   * Lưu `saved_models/scaler.joblib` và `saved_models/pca.joblib` để Người 5 dùng khi chạy inference trên giao diện.
3. **Output của Người 3 & 4 (`src/models/`) $\rightarrow$ Input của Người 5 (`app/`):**
   * Lưu mô hình tốt nhất tại `saved_models/best_model.joblib` và bảng tổng hợp metrics tại `results/model_comparison.csv`.
