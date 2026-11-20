"""
Module 3 & 4: Huấn luyện và Đánh giá mô hình Học máy
- Nguyễn Phạm Anh Tuấn (20235863): k-NN & SVM
- Lê Minh Trọng (20239675): Random Forest & XGBoost, Tổng hợp Metrics & Confusion Matrix
"""
from src.models.train_baseline import (
    get_baseline_models,
    run_baseline_experiments,
    plot_confusion_matrices,
    plot_model_comparison_bars
)

__all__ = [
    "get_baseline_models",
    "run_baseline_experiments",
    "plot_confusion_matrices",
    "plot_model_comparison_bars"
]
