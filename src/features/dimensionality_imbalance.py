"""
Module Chuẩn hóa, Giảm chiều dữ liệu (PCA) và Xử lý mất cân bằng lớp (SMOTE / Hybrid Sampling)
Phụ trách: Nguyễn Sỹ Vinh (20235875)
- Chia tập dữ liệu Train / Test (Stratified 80/20) TRƯỚC khi fit StandardScaler, PCA và SMOTE để tránh Data Leakage.
- Áp dụng StandardScaler (mean=0, std=1) và PCA (giữ 98% phương sai thông tin).
- Xử lý mất cân bằng lớp trên tập Train bằng Hybrid Sampling (RandomUnderSampler cho lớp đa số N + SMOTE cho 4 lớp thiểu số S, V, F, Q).
- Vẽ 2 biểu đồ chất lượng cao (300 DPI) cho Báo cáo tiến độ:
  1. results/fig5_pca_explained_variance.png (Phương sai tích lũy của PCA)
  2. results/fig6_smote_class_balance.png (Phân bố 5 lớp trước và sau khi cân bằng SMOTE)
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import RESULTS_DIR, ID_TO_LABEL, RANDOM_SEED


def split_scale_and_pca(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    test_size: float = 0.20,
    pca_variance: float = 0.98,
    random_state: int = RANDOM_SEED
) -> Tuple[Dict[str, np.ndarray], StandardScaler, PCA]:
    """
    Thực hiện Stratified Train/Test Split (80/20), chuẩn hóa StandardScaler và giảm chiều PCA.
    Lưu cả bản dữ liệu đã chuẩn hóa nhưng chưa qua PCA (để phục vụ Ablation Study ở bước sau).
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        stratify=y_labels,
        random_state=random_state
    )

    # 1. Chuẩn hóa thang đo StandardScaler (chỉ fit trên X_train_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    # 2. Giảm chiều dữ liệu PCA (chỉ fit trên X_train_scaled)
    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    data_dict = {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "y_train": y_train.astype(np.int64),
        "y_test": y_test.astype(np.int64)
    }
    return data_dict, scaler, pca


def balance_training_data_smote(
    X_train_pca: np.ndarray,
    y_train: np.ndarray,
    majority_cap: int = 25000,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xử lý mất cân bằng dữ liệu cực đoan trên tập Train bằng chiến lược Hybrid:
    - Bước 1 (RandomUnderSampler): Giảm nhẹ lớp đa số N (từ ~72,070 mẫu xuống `majority_cap` = 25,000 mẫu)
      để giảm thời gian huấn luyện SVM bậc hai mà vẫn giữ trọn phân bố lớp N.
    - Bước 2 (SMOTE): Tăng mẫu tổng hợp cho các lớp thiểu số (S, V, F, Q) lên mức cân bằng hợp lý:
      + Lớp N (0): 25,000 mẫu
      + Lớp V (2): 15,000 mẫu
      + Lớp S (1): 12,000 mẫu
      + Lớp F (3):  8,000 mẫu
      + Lớp Q (4):  3,000 mẫu
      (Lưu ý: Lớp Q chỉ có ~12 mẫu trong tập Train nên k_neighbors của SMOTE được đặt tự động = min(5, n_Q - 1)).
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))

    # Bước 1: Undersample lớp đa số N (nhãn 0) nếu vượt quá majority_cap
    under_strategy = {}
    for cls_id, cnt in class_counts.items():
        if cls_id == 0 and cnt > majority_cap:
            under_strategy[cls_id] = majority_cap
        else:
            under_strategy[cls_id] = cnt

    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    X_under, y_under = rus.fit_resample(X_train_pca, y_train)

    # Bước 2: Oversample các lớp thiểu số S, V, F, Q bằng SMOTE
    _, counts_after_under = np.unique(y_under, return_counts=True)
    min_class_count = int(np.min(counts_after_under))
    k_neighbors = max(1, min(5, min_class_count - 1))

    target_smote = {
        0: max(under_strategy.get(0, 25000), 25000),  # N
        1: max(under_strategy.get(1, 0), 12000),      # S
        2: max(under_strategy.get(2, 0), 15000),      # V
        3: max(under_strategy.get(3, 0), 8000),       # F
        4: max(under_strategy.get(4, 0), 3000),       # Q
    }

    smote = SMOTE(
        sampling_strategy=target_smote,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_bal, y_bal = smote.fit_resample(X_under, y_under)
    return X_bal.astype(np.float32), y_bal.astype(np.int64)


def plot_pca_variance(
    X_train_scaled: np.ndarray,
    chosen_pca: PCA,
    save_path: Path = RESULTS_DIR / "fig5_pca_explained_variance.png"
):
    """Vẽ biểu đồ phương sai giải thích tích lũy (Cumulative Explained Variance) của PCA."""
    full_pca = PCA().fit(X_train_scaled)
    cum_var = np.cumsum(full_pca.explained_variance_ratio_) * 100.0
    n_total = len(cum_var)
    n_selected = chosen_pca.n_components_
    selected_var = np.sum(chosen_pca.explained_variance_ratio_) * 100.0

    plt.figure(figsize=(8.8, 4.6), dpi=300)
    components = np.arange(1, n_total + 1)
    plt.bar(
        components,
        full_pca.explained_variance_ratio_ * 100.0,
        alpha=0.55,
        color="#1f77b4",
        label="Phương sai từng thành phần (Individual Variance %)"
    )
    plt.plot(
        components,
        cum_var,
        color="#d62728",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Phương sai tích lũy (Cumulative Variance %)"
    )

    plt.axvline(n_selected, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"Số chiều chọn: k = {n_selected} ({selected_var:.2f}% phương sai)")
    plt.axhline(selected_var, color="#2ca02c", linestyle=":", linewidth=1.3)

    plt.title(f"Phân tích Thành phần chính (PCA): Giảm chiều từ {n_total} -> {n_selected} chiều (Giữ {selected_var:.2f}% phương sai)", fontsize=11, fontweight="bold")
    plt.xlabel("Số lượng thành phần chính (Number of Principal Components)", fontsize=10)
    plt.ylabel("Tỷ lệ phương sai giải thích (%)", fontsize=10)
    plt.xlim(0, n_total + 1)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="center right", fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ phương sai PCA tại: {save_path}")


def plot_smote_comparison(
    y_train_before: np.ndarray,
    y_train_after: np.ndarray,
    save_path: Path = RESULTS_DIR / "fig6_smote_class_balance.png"
):
    """Vẽ biểu đồ đối chiếu phân bố 5 lớp AAMI trên tập Train Trước và Sau khi cân bằng Hybrid SMOTE."""
    classes = [ID_TO_LABEL[i] for i in range(5)]
    before_counts = [int(np.sum(y_train_before == i)) for i in range(5)]
    after_counts = [int(np.sum(y_train_after == i)) for i in range(5)]
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#7f7f7f']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    # Trước khi cân bằng
    bars1 = axes[0].bar(classes, before_counts, color=colors, edgecolor="black", alpha=0.88)
    for bar, cnt in zip(bars1, before_counts):
        pct = cnt / len(y_train_before) * 100.0
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(before_counts) * 0.015,
            f"{cnt:,d}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )
    axes[0].set_title(f"Tập Train TRƯỚC khi cân bằng (n = {len(y_train_before):,d} nhịp)\nMất cân bằng cực đoan (Lớp N chiếm 89.47%)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Nhóm rối loạn nhịp tim (AAMI Classes)", fontsize=9.5)
    axes[0].set_ylabel("Số lượng mẫu (Samples)", fontsize=9.5)
    axes[0].set_ylim(0, max(before_counts) * 1.18)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Sau khi cân bằng Hybrid SMOTE + Undersampling
    bars2 = axes[1].bar(classes, after_counts, color=colors, edgecolor="black", alpha=0.88)
    for bar, cnt in zip(bars2, after_counts):
        pct = cnt / len(y_train_after) * 100.0
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(after_counts) * 0.015,
            f"{cnt:,d}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )
    axes[1].set_title(f"Tập Train SAU khi cân bằng Hybrid SMOTE (n = {len(y_train_after):,d} nhịp)\nTăng cường các lớp bệnh lý hiếm (S, V, F, Q)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Nhóm rối loạn nhịp tim (AAMI Classes)", fontsize=9.5)
    axes[1].set_ylabel("Số lượng mẫu (Samples)", fontsize=9.5)
    axes[1].set_ylim(0, max(after_counts) * 1.18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh Trước/Sau SMOTE tại: {save_path}")
