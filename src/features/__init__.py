"""
Module 2: Trích xuất đặc trưng, Giảm chiều & Cân bằng dữ liệu (Phụ trách: Nguyễn Sỹ Vinh - 20235875)
- Đặc trưng khoảng R-R (Pre-RR, Post-RR, Local-RR, Ratio-RR)
- Đặc trưng hình thái phức bộ QRS & T (Biên độ, độ rộng, diện tích, độ dốc, Skewness, Kurtosis)
- Đặc trưng biến đổi Wavelet rời rạc (DWT - PyWavelets họ db4 mức 4)
- Chuẩn hóa StandardScaler, Giảm chiều PCA & Xử lý mất cân bằng lớp (Hybrid SMOTE)
"""
from src.features.extraction import (
    compute_rr_features,
    compute_morphological_features,
    compute_wavelet_features,
    extract_all_features
)
from src.features.dimensionality_imbalance import (
    split_scale_and_pca,
    balance_training_data_smote,
    plot_pca_variance,
    plot_smote_comparison
)

__all__ = [
    "compute_rr_features",
    "compute_morphological_features",
    "compute_wavelet_features",
    "extract_all_features",
    "split_scale_and_pca",
    "balance_training_data_smote",
    "plot_pca_variance",
    "plot_smote_comparison"
]
