"""
Module 3 & 4: Huấn luyện và Đánh giá 4 Mô hình Học máy (Baseline Experiments)
Phụ trách:
  - Nguyễn Phạm Anh Tuấn (20235863): k-NN & SVM
  - Lê Minh Trọng (20239675): Random Forest & XGBoost, Tổng hợp Metrics & Confusion Matrix
Đọc dữ liệu từ `data/processed/train_test_features.npz` (do Nguyễn Sỹ Vinh bàn giao):
  - X_train (63,000 x 26) đã qua StandardScaler + PCA + Hybrid SMOTE
  - X_test (20,139 x 26) giữ nguyên phân bố thực tế
Đánh giá cả 4 mô hình qua:
  - Accuracy (%)
  - Macro F1-Score (%) & Weighted F1-Score (%)
  - F1-Score chi tiết trên từng lớp AAMI (N, S, V, F, Q)
  - Confusion Matrix (5x5)
  - Thời gian huấn luyện (Training Time - s) và Thời gian dự đoán (Inference Time - ms / 1000 nhịp)
Xuất kết quả:
  - `results/model_comparison.csv`
  - `results/per_class_f1.csv`
  - `results/fig7_confusion_matrices_4models.png`
  - `results/fig8_model_comparison_metrics.png`
  - `saved_models/best_model.joblib` (cùng từng model `.joblib`)

Chạy lệnh:
  python src/models/train_baseline.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import (
    DATA_PROCESSED_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    ID_TO_LABEL,
    RANDOM_SEED
)


def get_baseline_models():
    """Khởi tạo 4 mô hình Baseline theo đúng đề tài."""
    return {
        "k-NN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=24,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }


def plot_confusion_matrices(
    cms_dict: dict,
    metrics_df: pd.DataFrame,
    save_path: Path = RESULTS_DIR / "fig7_confusion_matrices_4models.png"
):
    """Vẽ lưới 2x2 Confusion Matrix (5x5) chuẩn hóa theo hàng (% Recall) kèm số lượng mẫu thực tế."""
    class_labels = [ID_TO_LABEL[i] for i in range(5)]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), dpi=300)
    axes = axes.flatten()

    for idx, (model_name, cm) in enumerate(cms_dict.items()):
        ax = axes[idx]
        # Tính phần trăm theo từng lớp thực tế (True label row-wise normalization)
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_pct = cm / row_sums * 100.0

        # Tạo chuỗi hiển thị gồm cả số lượng mẫu và tỷ lệ %
        annot_matrix = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_matrix[i, j] = f"{cm[i, j]:,d}\n({cm_pct[i, j]:.1f}%)"

        row_info = metrics_df[metrics_df["Model"] == model_name].iloc[0]
        acc = row_info["Accuracy (%)"]
        mf1 = row_info["Macro F1 (%)"]

        sns.heatmap(
            cm_pct,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 8.5, "weight": "bold"},
            linewidths=0.6,
            linecolor="lightgray"
        )
        ax.set_title(
            f"{model_name}  |  Accuracy: {acc:.2f}%  |  Macro F1: {mf1:.2f}%",
            fontsize=11,
            fontweight="bold",
            pad=8
        )
        ax.set_xlabel("Nhãn dự đoán (Predicted Class)", fontsize=9.5)
        ax.set_ylabel("Nhãn thực tế (True Class)", fontsize=9.5)

    plt.suptitle(
        "Ma trận nhầm lẫn (Confusion Matrix 5x5) của 4 Mô hình trên Tập Test (20,139 nhịp tim)",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ Confusion Matrix của 4 mô hình tại: {save_path}")


def plot_model_comparison_bars(
    metrics_df: pd.DataFrame,
    save_path: Path = RESULTS_DIR / "fig8_model_comparison_metrics.png"
):
    """Vẽ biểu đồ so sánh Accuracy, Macro F1-Score và Thời gian thực thi của 4 mô hình."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    models = metrics_df["Model"].tolist()
    acc = metrics_df["Accuracy (%)"].tolist()
    macro_f1 = metrics_df["Macro F1 (%)"].tolist()
    x = np.arange(len(models))
    width = 0.36

    # Subplot 1: Accuracy vs Macro F1
    b1 = axes[0].bar(x - width / 2, acc, width, label="Accuracy (%)", color="#1f77b4", edgecolor="black", alpha=0.9)
    b2 = axes[0].bar(x + width / 2, macro_f1, width, label="Macro F1-Score (%)", color="#ff7f0e", edgecolor="black", alpha=0.9)

    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b2:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    axes[0].set_title("So sánh Hiệu năng Phân loại: Accuracy vs. Macro F1-Score", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Điểm số (%)", fontsize=10)
    axes[0].set_ylim(50, 108)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=9.5)

    # Subplot 2: Thời gian huấn luyện (s) & Thời gian dự đoán (ms / 1000 nhịp)
    train_t = metrics_df["Train Time (s)"].tolist()
    infer_t = metrics_df["Inference Time (ms/1k beats)"].tolist()

    b3 = axes[1].bar(x - width / 2, train_t, width, label="Training Time (giây)", color="#2ca02c", edgecolor="black", alpha=0.9)
    b4 = axes[1].bar(x + width / 2, infer_t, width, label="Inference Time (ms / 1000 nhịp)", color="#9467bd", edgecolor="black", alpha=0.9)

    max_t = max(max(train_t), max(infer_t), 1.0)
    for bar in b3:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_t * 0.02, f"{bar.get_height():.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b4:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_t * 0.02, f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    axes[1].set_title("So sánh Thời gian Huấn luyện & Tốc độ Dự đoán (Inference)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Thời gian (s hoặc ms / 1000 nhịp)", fontsize=10)
    axes[1].set_ylim(0, max_t * 1.18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh hiệu năng 4 mô hình tại: {save_path}")


def run_baseline_experiments(
    features_npz: Path = DATA_PROCESSED_DIR / "train_test_features.npz"
):
    if not features_npz.exists():
        raise FileNotFoundError(f"Không tìm thấy {features_npz}. Hãy chạy Module 2 (run_features.py) trước!")

    print("\n" + "=" * 80)
    print("MODULE 3 & 4: HUẤN LUYỆN & ĐÁNH GIÁ 4 MÔ HÌNH HỌC MÁY (k-NN, SVM, RF, XGBoost)")
    print("=" * 80)

    data = np.load(features_npz)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"[*] Dữ liệu nạp vào: X_train = {X_train.shape} (PCA + Hybrid SMOTE) | X_test = {X_test.shape} (Thực tế)")

    models = get_baseline_models()
    summary_rows = []
    per_class_rows = []
    cms_dict = {}

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        print(f"\n[-] Đang huấn luyện mô hình: {name} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_total_sec = time.perf_counter() - t1
        # Quy đổi ra thời gian dự đoán trung bình cho 1.000 nhịp tim (ms / 1000 beats)
        infer_ms_per_1k = (infer_total_sec / len(X_test)) * 1000.0 * 1000.0

        acc = accuracy_score(y_test, y_pred) * 100.0
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
        f1_per_cls = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0) * 100.0
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

        cms_dict[name] = cm
        summary_rows.append({
            "Model": name,
            "Accuracy (%)": round(acc, 2),
            "Macro F1 (%)": round(macro_f1, 2),
            "Weighted F1 (%)": round(weighted_f1, 2),
            "Macro Precision (%)": round(macro_prec, 2),
            "Macro Recall (%)": round(macro_rec, 2),
            "Train Time (s)": round(train_time_sec, 2),
            "Inference Time (ms/1k beats)": round(infer_ms_per_1k, 2)
        })

        per_class_rows.append({
            "Model": name,
            "F1_N (%)": round(f1_per_cls[0], 2),
            "F1_S (%)": round(f1_per_cls[1], 2),
            "F1_V (%)": round(f1_per_cls[2], 2),
            "F1_F (%)": round(f1_per_cls[3], 2),
            "F1_Q (%)": round(f1_per_cls[4], 2),
            "Macro_F1 (%)": round(macro_f1, 2)
        })

        print(
            f"    -> Hoàn tất {name}: Accuracy = {acc:.2f}% | Macro F1 = {macro_f1:.2f}% | "
            f"Weighted F1 = {weighted_f1:.2f}% | Train = {train_time_sec:.2f}s | Infer = {infer_ms_per_1k:.2f} ms/1k nhịp"
        )
        print(
            f"       F1 từng lớp: N={f1_per_cls[0]:.2f}%, S={f1_per_cls[1]:.2f}%, "
            f"V={f1_per_cls[2]:.2f}%, F={f1_per_cls[3]:.2f}%, Q={f1_per_cls[4]:.2f}%"
        )

        # Lưu từng model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        joblib.dump(model, SAVED_MODELS_DIR / f"{safe_name}_model.joblib")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model_obj = model

    # Lưu mô hình tốt nhất cho Đào Trọng Công tích hợp vào Streamlit UI
    joblib.dump(
        {"model_name": best_model_name, "model": best_model_obj, "macro_f1": best_macro_f1},
        SAVED_MODELS_DIR / "best_model.joblib"
    )

    df_summary = pd.DataFrame(summary_rows)
    df_per_class = pd.DataFrame(per_class_rows)

    df_summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    df_per_class.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)

    # Vẽ 2 biểu đồ đánh giá tổng hợp
    plot_confusion_matrices(cms_dict, df_summary)
    plot_model_comparison_bars(df_summary)

    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM 4 MÔ HÌNH TRÊN TẬP TEST (20,139 NHỊP TIM)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("\n--- F1-SCORE CHI TIẾT TỪNG LỚP AAMI ---")
    print(df_per_class.to_string(index=False))
    print(f"\n[*] Mô hình đạt Macro F1-Score cao nhất: {best_model_name} ({best_macro_f1:.2f}%) -> Đã lưu tại saved_models/best_model.joblib")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_experiments()
