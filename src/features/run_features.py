"""
Pipeline Thực thi Module 2 (Feature Engineering, PCA & Imbalance Handling)
Phụ trách: Nguyễn Sỹ Vinh (20235875)
- Đọc file `data/processed/processed_beats.npz` (100.694 nhịp tim) từ Trương Thế Thành.
- Trích xuất 58 đặc trưng đa miền:
  + 5 đặc trưng Khoảng R-R (Pre-RR, Post-RR, Local-RR 10 nhịp, Ratio Pre/Local, Ratio Post/Pre)
  + 12 đặc trưng Hình thái sóng QRS & T (Biên độ R, Q, S, T, độ rộng QRS, diện tích, năng lượng, độ dốc, Skewness, Kurtosis)
  + 41 đặc trưng Biến đổi Wavelet rời rạc DWT (Họ sóng db4 mức 4: 21 hệ số cA4 + 20 thống kê năng lượng & entropy)
- Chia tập Train/Test (80/20 Stratified), chuẩn hóa StandardScaler, giảm chiều PCA (giữ 98% phương sai).
- Xử lý mất cân bằng dữ liệu trên tập Train bằng Hybrid SMOTE + RandomUnderSampler.
- Xuất file `data/processed/train_test_features.npz` bàn giao cho Nguyễn Phạm Anh Tuấn & Lê Minh Trọng.
- Xuất file `saved_models/feature_pipeline.joblib` (chứa scaler, pca, feature_names) bàn giao cho Đào Trọng Công.
- Xuất 2 biểu đồ `fig5_pca_explained_variance.png` và `fig6_smote_class_balance.png` vào `results/`.

Chạy lệnh:
  python src/features/run_features.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import DATA_PROCESSED_DIR, SAVED_MODELS_DIR, ID_TO_LABEL
from src.features.extraction import extract_all_features
from src.features.dimensionality_imbalance import (
    split_scale_and_pca,
    balance_training_data_smote,
    plot_pca_variance,
    plot_smote_comparison
)


def run_feature_engineering_pipeline(
    input_npz: Path = DATA_PROCESSED_DIR / "processed_beats.npz",
    output_npz: Path = DATA_PROCESSED_DIR / "train_test_features.npz",
    pipeline_out: Path = SAVED_MODELS_DIR / "feature_pipeline.joblib",
    pca_variance: float = 0.98
):
    start_time = time.time()
    if not input_npz.exists():
        raise FileNotFoundError(f"Không tìm thấy file {input_npz}. Hãy chạy Module 1 (run_pipeline.py) trước!")

    print("\n" + "=" * 78)
    print("MODULE 2: TRÍCH XUẤT ĐẶC TRƯNG (RR, QRS, WAVELET), GIẢM CHIỀU PCA & CÂN BẰNG SMOTE")
    print("=" * 78)

    # 1. Đọc dữ liệu từ Module 1
    print(f"[*] Đang tải dữ liệu nhịp tim đã cắt từ: {input_npz} ...")
    data = np.load(input_npz)
    X_beats = data["X_beats"]
    pre_rr_samples = data["pre_rr_samples"]
    post_rr_samples = data["post_rr_samples"]
    y_labels = data["y_labels"]
    record_ids = data["record_ids"]
    print(f"    -> Đã tải {X_beats.shape[0]:,d} nhịp tim (mỗi nhịp {X_beats.shape[1]} mẫu @ 360 Hz).")

    # 2. Trích xuất 58 đặc trưng đa miền
    print("\n[*] Bước 1: Trích xuất đặc trưng Khoảng R-R, Hình thái QRS/T và Biến đổi Wavelet (db4, level=4)...")
    t0 = time.time()
    X_features, feature_names = extract_all_features(
        X_beats=X_beats,
        pre_rr_samples=pre_rr_samples,
        post_rr_samples=post_rr_samples,
        record_ids=record_ids,
        wavelet="db4",
        level=4
    )
    print(f"    -> Hoàn tất trích xuất đặc trưng trong {time.time() - t0:.2f}s! Kích thước ma trận đặc trưng: {X_features.shape}")

    # 3. Chia Train/Test (80/20), Chuẩn hóa StandardScaler & Giảm chiều PCA
    print(f"\n[*] Bước 2: Chia Stratified Train/Test (80/20), chuẩn hóa StandardScaler và chạy PCA (giữ {pca_variance*100:.0f}% phương sai)...")
    split_dict, scaler, pca = split_scale_and_pca(
        X_features=X_features,
        y_labels=y_labels,
        test_size=0.20,
        pca_variance=pca_variance
    )

    X_train_pca = split_dict["X_train_pca"]
    X_test_pca = split_dict["X_test_pca"]
    y_train = split_dict["y_train"]
    y_test = split_dict["y_test"]
    explained_pct = np.sum(pca.explained_variance_ratio_) * 100.0

    print(f"    -> Tập Train gốc : {X_train_pca.shape[0]:,d} mẫu | Tập Test: {X_test_pca.shape[0]:,d} mẫu")
    print(f"    -> Số chiều giảm : từ {X_features.shape[1]} đặc trưng gốc -> {pca.n_components_} thành phần chính PCA (giữ {explained_pct:.2f}% phương sai)")

    # Vẽ biểu đồ phương sai PCA
    plot_pca_variance(split_dict["X_train_scaled"], pca)

    # 4. Xử lý mất cân bằng dữ liệu trên tập Train bằng Hybrid SMOTE
    print("\n[*] Bước 3: Xử lý mất cân bằng lớp trên tập Train bằng Hybrid SMOTE + RandomUnderSampler...")
    X_train_smote, y_train_smote = balance_training_data_smote(
        X_train_pca=X_train_pca,
        y_train=y_train,
        majority_cap=25000
    )

    # Vẽ biểu đồ phân bố trước và sau SMOTE
    plot_smote_comparison(y_train, y_train_smote)

    # 5. Lưu file nén train_test_features.npz cho Tuấn và Trọng (kèm cả bản không PCA / không SMOTE cho Ablation Study)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        # Bộ dữ liệu chính đã qua PCA + SMOTE để huấn luyện 4 mô hình
        X_train=X_train_smote,
        y_train=y_train_smote,
        X_test=X_test_pca,
        y_test=y_test,
        # Bộ dữ liệu bổ sung phục vụ Ablation Study (chưa SMOTE và chưa PCA)
        X_train_pca_unbalanced=X_train_pca,
        y_train_unbalanced=y_train,
        X_train_scaled_no_pca=split_dict["X_train_scaled"],
        X_test_scaled_no_pca=split_dict["X_test_scaled"],
        feature_names=np.array(feature_names, dtype="<U32")
    )

    # 6. Lưu đối tượng Scaler + PCA ra file .joblib để Đào Trọng Công dùng trên giao diện Streamlit
    pipeline_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "feature_names": feature_names,
            "wavelet": "db4",
            "wavelet_level": 4
        },
        pipeline_out
    )

    elapsed = time.time() - start_time
    npz_size_mb = output_npz.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 78)
    print("HOÀN TẤT BÀN GIAO DỮ LIỆU MODULE 2 (NGUYỄN SỸ VINH -> TUẤN, TRỌNG, CÔNG)")
    print("=" * 78)
    print(f"  - File đặc trưng Train/Test : {output_npz} ({npz_size_mb:.2f} MB)")
    print(f"  - File Pipeline Scaler+PCA  : {pipeline_out}")
    print(f"  - Số đặc trưng trước PCA    : {X_features.shape[1]} chiều (5 RR + 12 QRS/T + 41 Wavelet db4)")
    print(f"  - Số chiều sau PCA (98%)    : {pca.n_components_} chiều (Giải thích {explained_pct:.2f}% phương sai)")
    print(f"  - Kích thước X_train (SMOTE): {X_train_smote.shape}")
    print(f"  - Kích thước X_test (Thực)  : {X_test_pca.shape}")
    print("  - Phân bố tập Train Trước vs. Sau Hybrid SMOTE:")
    for cls_id in range(5):
        cls_name = ID_TO_LABEL[cls_id]
        cnt_before = int(np.sum(y_train == cls_id))
        cnt_after = int(np.sum(y_train_smote == cls_id))
        cnt_test = int(np.sum(y_test == cls_id))
        print(f"      + Lớp {cls_name} (nhãn {cls_id}): Train gốc = {cnt_before:>6,d} -> Sau SMOTE = {cnt_after:>6,d} | Tập Test = {cnt_test:>6,d}")
    print(f"  - Tổng thời gian thực thi   : {elapsed:.2f} giây")
    print("=" * 78)


if __name__ == "__main__":
    run_feature_engineering_pipeline()
