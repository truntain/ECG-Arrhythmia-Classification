"""
Pipeline Tổng hợp Bước 2 & Bước 3: Tiền xử lý, Phân đoạn nhịp & Bàn giao dữ liệu (Data Handoff)
Phụ trách: Trương Thế Thành (20235835)
- Tự động tải dữ liệu MIT-BIH (nếu chưa có trong data/raw/).
- Loại bỏ 4 bản ghi máy tạo nhịp (102, 104, 107, 217) theo chuẩn AAMI EC57 -> xử lý 44 bản ghi chuẩn.
- Chọn đúng kênh chuyển đạo MLII cho từng bản ghi.
- Lọc nhiễu Bandpass Butterworth bậc 4 (0.5 - 45 Hz) với zero-phase filtfilt.
- Ánh xạ nhãn sang 5 lớp AAMI (N, S, V, F, Q) và cắt nhịp [-90, +160] mẫu quanh đỉnh R (250 mẫu / nhịp).
- Xuất file nén `data/processed/processed_beats.npz` kèm đầy đủ `X_beats`, `r_peaks`, `pre_rr_samples`,
  `post_rr_samples`, `y_labels`, `y_symbols`, `record_ids` để bàn giao cho Nguyễn Sỹ Vinh (Feature Engineering).
- Tự động sinh 4 biểu đồ EDA & Tiền xử lý chất lượng cao (300 DPI) trong thư mục `results/` để đưa vào báo cáo.

Chạy lệnh:
  python src/preprocessing/run_pipeline.py
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from data.download_mitbih import download_dataset
from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    MIT_BIH_RECORDS,
    AAMI_EXCLUDED_RECORDS,
    ID_TO_LABEL
)
from src.preprocessing.segmentation import segment_record_beats
from src.preprocessing.eda import (
    get_available_aami_records,
    inspect_channels_and_annotations,
    plot_raw_ecg_with_r_peaks,
    plot_denoising_comparison,
    plot_segmented_beats_and_distribution
)


def run_preprocessing_pipeline(
    records=None,
    raw_dir: Path = DATA_RAW_DIR,
    output_file: Path = DATA_PROCESSED_DIR / "processed_beats.npz",
    auto_download: bool = True
):
    """Chạy toàn bộ pipeline tiền xử lý và xuất file processed_beats.npz bàn giao cho Module 2."""
    start_time = time.time()

    # 1. Xác định danh sách 44 bản ghi chuẩn AAMI (loại bỏ 102, 104, 107, 217)
    if records is None:
        target_records = [r for r in MIT_BIH_RECORDS if r not in AAMI_EXCLUDED_RECORDS]
    else:
        target_records = [r for r in records if r not in AAMI_EXCLUDED_RECORDS]

    if auto_download:
        download_dataset(records=target_records, target_dir=raw_dir)

    available_records = [r for r in target_records if (raw_dir / f"{r}.dat").exists()]
    if not available_records:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu MIT-BIH trong {raw_dir}. Hãy kiểm tra kết nối mạng.")

    # 2. Chạy EDA kiểm tra kênh MLII & thống kê nhãn
    print("\n" + "=" * 75)
    print("BƯỚC 2: KHÁM PHÁ DỮ LIỆU (EDA) & KIỂM TRA KÊNH CHUYỂN ĐẠO MLII")
    print("=" * 75)
    inspect_channels_and_annotations(available_records, raw_dir)

    sample_rec = "200" if "200" in available_records else available_records[0]
    plot_raw_ecg_with_r_peaks(record_id=sample_rec, start_sec=0.0, duration_sec=10.0, raw_dir=raw_dir)
    plot_denoising_comparison(record_id=sample_rec, start_sec=0.0, duration_sec=8.0, raw_dir=raw_dir)

    # 3. Chạy lọc nhiễu Bandpass 0.5-45Hz & cắt nhịp [-90, +160] mẫu trên tất cả bản ghi
    print("\n" + "=" * 75)
    print("BƯỚC 3: LỌC NHIỄU BANDPASS (0.5-45 Hz), ÁNH XẠ 5 LỚP AAMI & CẮT NHỊP (250 MẪU)")
    print("=" * 75)

    all_X_beats = []
    all_r_peaks = []
    all_pre_rr = []
    all_post_rr = []
    all_y_labels = []
    all_y_symbols = []
    all_record_ids = []

    for rec_id in tqdm(available_records, desc="Đang xử lý bản ghi MIT-BIH"):
        res = segment_record_beats(
            record_id=rec_id,
            raw_dir=raw_dir,
            apply_filter=True,
            lowcut=0.5,
            highcut=45.0
        )
        all_X_beats.append(res["X_beats"])
        all_r_peaks.append(res["r_peaks"])
        all_pre_rr.append(res["pre_rr_samples"])
        all_post_rr.append(res["post_rr_samples"])
        all_y_labels.append(res["y_labels"])
        all_y_symbols.append(res["y_symbols"])
        all_record_ids.append(res["record_ids"])

    X_beats = np.vstack(all_X_beats).astype(np.float32)
    r_peaks = np.concatenate(all_r_peaks).astype(np.int32)
    pre_rr_samples = np.concatenate(all_pre_rr).astype(np.int32)
    post_rr_samples = np.concatenate(all_post_rr).astype(np.int32)
    y_labels = np.concatenate(all_y_labels).astype(np.int64)
    y_symbols = np.concatenate(all_y_symbols)
    record_ids = np.concatenate(all_record_ids)

    # 4. Vẽ biểu đồ hình thái 5 lớp & phân bố dữ liệu sau khi cắt nhịp
    plot_segmented_beats_and_distribution(X_beats, y_labels)

    # 5. Lưu file nén processed_beats.npz bàn giao cho bạn Nguyễn Sỹ Vinh
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        X_beats=X_beats,
        r_peaks=r_peaks,
        pre_rr_samples=pre_rr_samples,
        post_rr_samples=post_rr_samples,
        y_labels=y_labels,
        y_symbols=y_symbols,
        record_ids=record_ids
    )

    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 75)
    print("HOÀN TẤT BÀN GIAO DỮ LIỆU MODULE 1 (DATA HANDOFF SUMMARY)")
    print("=" * 75)
    print(f"  - File xuất ra          : {output_file} ({file_size_mb:.2f} MB)")
    print(f"  - Số bản ghi đã xử lý   : {len(available_records)} bản ghi chuẩn AAMI")
    print(f"  - Kích thước X_beats    : {X_beats.shape} (float32, [-90, +160] mẫu quanh đỉnh R)")
    print(f"  - Kích thước r_peaks    : {r_peaks.shape} (int32, tọa độ mẫu đỉnh R)")
    print(f"  - Kích thước y_labels   : {y_labels.shape} (int64: 0=N, 1=S, 2=V, 3=F, 4=Q)")
    print("  - Phân bố số nhịp đã cắt:")
    for cls_id in range(5):
        cls_name = ID_TO_LABEL[cls_id]
        cnt = int(np.sum(y_labels == cls_id))
        pct = cnt / len(y_labels) * 100.0
        print(f"      + Lớp {cls_name} (nhãn {cls_id}): {cnt:>7,d} nhịp ({pct:6.2f}%)")
    print(f"  - Tổng thời gian chạy   : {elapsed:.2f} giây")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Bước 2 & 3: EDA, Lọc nhiễu, Cắt nhịp và Xuất dữ liệu")
    parser.add_argument(
        "--records",
        nargs="+",
        default=None,
        help="Danh sách bản ghi cụ thể (mặc định: toàn bộ 44 bản ghi chuẩn AAMI)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Không tự động gọi tải dữ liệu từ PhysioNet"
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(
        records=args.records,
        auto_download=not args.no_download
    )
