"""
Module Khám phá dữ liệu (EDA) & Trực quan hóa tín hiệu ECG cho Báo cáo Tiến độ
Phụ trách: Trương Thế Thành (20235835)
- Kiểm tra kênh chuyển đạo MLII trên các bản ghi MIT-BIH (sau khi loại bỏ 102, 104, 107, 217).
- Vẽ đoạn tín hiệu ECG thô (10 giây) kèm đánh dấu đỉnh R và nhãn nhịp từ file .atr.
- Vẽ biểu đồ so sánh tín hiệu Trước và Sau khi lọc Bandpass Butterworth bậc 4 (0.5 - 45 Hz).
- Vẽ biểu đồ hình thái sóng trung bình (+/- std) của 5 nhóm nhịp chuẩn AAMI (N, S, V, F, Q) và biểu đồ phân bố lớp.
Chạy lệnh: python src/preprocessing/eda.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import (
    DATA_RAW_DIR,
    RESULTS_DIR,
    SAMPLING_RATE,
    WINDOW_BEFORE_R,
    WINDOW_AFTER_R,
    MIT_BIH_RECORDS,
    AAMI_EXCLUDED_RECORDS,
    ID_TO_LABEL,
)
from src.preprocessing.denoising import bandpass_filter
from src.preprocessing.segmentation import select_mlii_channel, map_symbol_to_aami


def get_available_aami_records(raw_dir: Path = DATA_RAW_DIR):
    """Lấy danh sách các bản ghi chuẩn AAMI (bỏ 102, 104, 107, 217) hiện có trong data/raw/."""
    valid_records = [r for r in MIT_BIH_RECORDS if r not in AAMI_EXCLUDED_RECORDS]
    available = [
        r for r in valid_records
        if (raw_dir / f"{r}.dat").exists() and (raw_dir / f"{r}.hea").exists() and (raw_dir / f"{r}.atr").exists()
    ]
    return available


def inspect_channels_and_annotations(records, raw_dir: Path = DATA_RAW_DIR):
    """Kiểm tra vị trí kênh MLII và thống kê phân bố nhãn gốc -> 5 lớp AAMI."""
    channel_report = []
    symbol_counts = {}
    aami_counts = {'N': 0, 'S': 0, 'V': 0, 'F': 0, 'Q': 0}

    for rec_id in records:
        rec_path = str(raw_dir / rec_id)
        hdr = wfdb.rdheader(rec_path)
        ann = wfdb.rdann(rec_path, 'atr')

        sig_names = [s.upper().strip() for s in hdr.sig_name]
        mlii_idx = sig_names.index("MLII") if "MLII" in sig_names else -1
        channel_report.append({
            "record_id": rec_id,
            "sig_name_0": hdr.sig_name[0],
            "sig_name_1": hdr.sig_name[1] if len(hdr.sig_name) > 1 else "N/A",
            "selected_channel": hdr.sig_name[mlii_idx] if mlii_idx != -1 else hdr.sig_name[0],
            "selected_index": mlii_idx if mlii_idx != -1 else 0
        })

        for sym in ann.symbol:
            aami = map_symbol_to_aami(sym)
            if aami is not None:
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                aami_counts[aami] += 1

    df_channels = pd.DataFrame(channel_report)
    non_zero_mlii = df_channels[df_channels["selected_index"] != 0]
    print(f"[*] Đã kiểm tra {len(records)} bản ghi chuẩn AAMI (đã loại bỏ {AAMI_EXCLUDED_RECORDS}).")
    if len(non_zero_mlii) > 0:
        print("[!] Phát hiện các bản ghi có kênh MLII KHÔNG nằm ở kênh 0 (đã tự động chuyển sang kênh đúng):")
        for _, row in non_zero_mlii.iterrows():
            print(f"    - Bản ghi {row['record_id']}: Kênh 0 = {row['sig_name_0']}, Kênh 1 = {row['sig_name_1']} -> Chọn kênh {row['selected_index']} ({row['selected_channel']})")

    total_beats = sum(aami_counts.values())
    print("\n[*] Thống kê phân bố 5 lớp chuẩn AAMI EC57 trên các bản ghi hiện có:")
    dist_rows = []
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        cnt = aami_counts[cls]
        pct = (cnt / total_beats * 100.0) if total_beats > 0 else 0.0
        print(f"    - Lớp {cls}: {cnt:>7,d} nhịp ({pct:6.2f}%)")
        dist_rows.append({"AAMI_Class": cls, "Count": cnt, "Percentage": round(pct, 2)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(RESULTS_DIR / "aami_class_distribution.csv", index=False)
    return df_channels, df_dist


def plot_raw_ecg_with_r_peaks(
    record_id: str = "200",
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    raw_dir: Path = DATA_RAW_DIR,
    save_path: Path = RESULTS_DIR / "fig1_raw_ecg_10s_rpeaks.png"
):
    """Vẽ đoạn tín hiệu ECG thô (10 giây) kèm đánh dấu các đỉnh R và nhãn từ file .atr."""
    rec_path = str(raw_dir / record_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')

    raw_sig, ch_name, ch_idx = select_mlii_channel(record)
    start_samp = int(start_sec * SAMPLING_RATE)
    end_samp = int((start_sec + duration_sec) * SAMPLING_RATE)

    t = np.arange(start_samp, end_samp) / SAMPLING_RATE
    sig_segment = raw_sig[start_samp:end_samp]

    # Lọc các đỉnh R nằm trong khoảng 10 giây
    mask = (ann.sample >= start_samp) & (ann.sample < end_samp)
    r_samples = ann.sample[mask]
    r_symbols = np.array(ann.symbol)[mask]

    beat_r_samples = []
    beat_aami_labels = []
    for s, sym in zip(r_samples, r_symbols):
        aami = map_symbol_to_aami(sym)
        if aami is not None:
            beat_r_samples.append(s)
            beat_aami_labels.append(f"{aami}({sym})" if aami != sym else aami)

    beat_r_samples = np.array(beat_r_samples, dtype=np.int32)
    r_times = beat_r_samples / SAMPLING_RATE
    r_amps = raw_sig[beat_r_samples]

    plt.figure(figsize=(13, 4.2), dpi=300)
    plt.plot(t, sig_segment, color="#1f77b4", linewidth=1.1, label=f"Raw ECG Signal (Record {record_id} - Channel {ch_name})")
    plt.scatter(r_times, r_amps, color="#d62728", s=38, zorder=5, label="Annotated R-peaks (.atr)")

    for rt, ra, lbl in zip(r_times, r_amps, beat_aami_labels):
        color = "#d62728" if not lbl.startswith("N") else "#2ca02c"
        plt.annotate(
            lbl,
            xy=(rt, ra),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color
        )

    plt.title(f"Khám phá tín hiệu ECG thô 10 giây & Đỉnh R (MIT-BIH Record {record_id}, Kênh {ch_name}[{ch_idx}])", fontsize=12, fontweight="bold")
    plt.xlabel("Thời gian (Giây - Seconds)", fontsize=10)
    plt.ylabel("Biên độ điện thế (mV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ khám phá tín hiệu 10s tại: {save_path}")


def plot_denoising_comparison(
    record_id: str = "200",
    start_sec: float = 0.0,
    duration_sec: float = 8.0,
    raw_dir: Path = DATA_RAW_DIR,
    save_path: Path = RESULTS_DIR / "fig2_denoising_bandpass_comparison.png"
):
    """Vẽ biểu đồ so sánh tín hiệu ECG Trước và Sau khi lọc Bandpass Butterworth bậc 4 (0.5 - 45 Hz)."""
    rec_path = str(raw_dir / record_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')

    raw_sig, ch_name, _ = select_mlii_channel(record)
    filtered_sig = bandpass_filter(raw_sig, lowcut=0.5, highcut=45.0, fs=SAMPLING_RATE, order=4)

    start_samp = int(start_sec * SAMPLING_RATE)
    end_samp = int((start_sec + duration_sec) * SAMPLING_RATE)
    t = np.arange(start_samp, end_samp) / SAMPLING_RATE

    mask = (ann.sample >= start_samp) & (ann.sample < end_samp)
    r_samples = [s for s, sym in zip(ann.sample[mask], np.array(ann.symbol)[mask]) if map_symbol_to_aami(sym) is not None]
    r_samples = np.array(r_samples, dtype=np.int32)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, dpi=300)

    # Subplot 1: Tín hiệu thô
    axes[0].plot(t, raw_sig[start_samp:end_samp], color="#7f7f7f", linewidth=1.0, label="Tín hiệu ECG thô (Lẫn nhiễu trôi đường cơ sở & nhiễu cao tần)")
    axes[0].scatter(r_samples / SAMPLING_RATE, raw_sig[r_samples], color="#d62728", s=28, zorder=5, label="Đỉnh R gốc (.atr)")
    axes[0].set_title(f"Trước khi lọc nhiễu: Tín hiệu ECG thô (Record {record_id} - Kênh {ch_name})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Biên độ (mV)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=9)

    # Subplot 2: Tín hiệu sau lọc Bandpass 0.5 - 45 Hz (filtfilt)
    axes[1].plot(t, filtered_sig[start_samp:end_samp], color="#1f77b4", linewidth=1.15, label="Tín hiệu sau lọc Bandpass Butterworth bậc 4 (0.5 - 45 Hz, Zero-phase filtfilt)")
    axes[1].scatter(r_samples / SAMPLING_RATE, filtered_sig[r_samples], color="#d62728", s=28, zorder=5, label="Đỉnh R giữ nguyên pha (Không bị trễ)")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Sau khi lọc nhiễu: Đường cơ sở ổn định về 0 mV, khử sạch nhiễu cao tần và bảo toàn vị trí đỉnh R", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Giây - Seconds)", fontsize=10)
    axes[1].set_ylabel("Biên độ (mV)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh Trước/Sau lọc nhiễu tại: {save_path}")


def plot_segmented_beats_and_distribution(
    X_beats: np.ndarray,
    y_labels: np.ndarray,
    save_morphology_path: Path = RESULTS_DIR / "fig3_5classes_beat_morphology.png",
    save_dist_path: Path = RESULTS_DIR / "fig4_aami_class_distribution.png"
):
    """Vẽ hình thái sóng trung bình của 5 lớp nhịp tim (250 mẫu) và biểu đồ phân bố số lượng mẫu."""
    time_ms = (np.arange(X_beats.shape[1]) - WINDOW_BEFORE_R) / SAMPLING_RATE * 1000.0
    colors = {'N': '#2ca02c', 'S': '#ff7f0e', 'V': '#d62728', 'F': '#9467bd', 'Q': '#7f7f7f'}
    titles = {
        'N': 'Lớp N (Normal Beat)',
        'S': 'Lớp S (Supraventricular Ectopic)',
        'V': 'Lớp V (Ventricular Ectopic)',
        'F': 'Lớp F (Fusion Beat)',
        'Q': 'Lớp Q (Unknown / Unclassifiable)'
    }

    # Hình 3: Hình thái sóng 5 lớp sau khi cắt [-90, +160] mẫu quanh đỉnh R
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True, dpi=300)
    for idx in range(5):
        cls_sym = ID_TO_LABEL[idx]
        ax = axes[idx]
        cls_beats = X_beats[y_labels == idx]
        n_count = len(cls_beats)

        if n_count > 0:
            mean_beat = np.mean(cls_beats, axis=0)
            std_beat = np.std(cls_beats, axis=0)
            ax.plot(time_ms, mean_beat, color=colors[cls_sym], linewidth=1.8, label="Mean waveform")
            ax.fill_between(time_ms, mean_beat - std_beat, mean_beat + std_beat, color=colors[cls_sym], alpha=0.22, label="±1 Std")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(f"{titles[cls_sym]}\n(n = {n_count:,d} beats)", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Thời gian quanh đỉnh R (ms)", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("Biên độ (mV)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.suptitle("Hình thái sóng ECG sau phân đoạn [-90, +160] mẫu (250 mẫu / nhịp @ 360 Hz) theo 5 lớp chuẩn AAMI", fontsize=11.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_morphology_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_morphology_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ hình thái 5 lớp nhịp tim tại: {save_morphology_path}")

    # Hình 4: Biểu đồ cột phân bố 5 lớp chuẩn AAMI
    counts = [int(np.sum(y_labels == i)) for i in range(5)]
    total = max(sum(counts), 1)
    cls_labels = [ID_TO_LABEL[i] for i in range(5)]

    plt.figure(figsize=(8.5, 4.5), dpi=300)
    bars = plt.bar(cls_labels, counts, color=[colors[c] for c in cls_labels], edgecolor="black", alpha=0.88)
    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100.0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            f"{cnt:,d}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
    plt.title("Phân bố số lượng nhịp tim theo 5 lớp chuẩn AAMI EC57 (MIT-BIH Dataset)", fontsize=11.5, fontweight="bold")
    plt.xlabel("Nhóm rối loạn nhịp tim (AAMI Classes)", fontsize=10)
    plt.ylabel("Số lượng nhịp tim (Beats)", fontsize=10)
    plt.ylim(0, max(counts) * 1.18 if max(counts) > 0 else 10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dist_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ phân bố 5 lớp AAMI tại: {save_dist_path}")


if __name__ == "__main__":
    available_records = get_available_aami_records(DATA_RAW_DIR)
    if not available_records:
        print("[!] Chưa có bản ghi nào trong data/raw/. Hãy chạy 'python data/download_mitbih.py' trước.")
        sys.exit(1)

    inspect_channels_and_annotations(available_records, DATA_RAW_DIR)
    sample_rec = "200" if "200" in available_records else available_records[0]
    plot_raw_ecg_with_r_peaks(record_id=sample_rec, start_sec=0.0, duration_sec=10.0)
    plot_denoising_comparison(record_id=sample_rec, start_sec=0.0, duration_sec=8.0)
