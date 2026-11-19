"""
Script tải tự động bộ dữ liệu MIT-BIH Arrhythmia Database từ PhysioNet về thư mục data/raw/.
Hỗ trợ kiểm tra các bản ghi đã tồn tại để tránh tải lại và cho phép tải toàn bộ hoặc tải thử nghiệm.
Chạy lệnh:
  - Tải toàn bộ 48 bản ghi: python data/download_mitbih.py
  - Chỉ tải 44 bản ghi chuẩn AAMI (bỏ 102, 104, 107, 217): python data/download_mitbih.py --aami-only
  - Tải nhanh vài bản ghi để test pipeline: python data/download_mitbih.py --records 100 101 200 208
"""
import argparse
import sys
from pathlib import Path
import wfdb

# Thêm thư mục gốc vào sys.path để import src.config
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import DATA_RAW_DIR, MIT_BIH_RECORDS, AAMI_EXCLUDED_RECORDS


def get_missing_records(records, target_dir: Path):
    """Kiểm tra những bản ghi nào chưa có đủ 3 file (.dat, .hea, .atr) trong thư mục."""
    missing = []
    for rec in records:
        dat_file = target_dir / f"{rec}.dat"
        hea_file = target_dir / f"{rec}.hea"
        atr_file = target_dir / f"{rec}.atr"
        if not (dat_file.exists() and hea_file.exists() and atr_file.exists()):
            missing.append(rec)
    return missing


def download_dataset(records=None, target_dir=DATA_RAW_DIR, force=False):
    """Tải danh sách bản ghi MIT-BIH từ PhysioNet."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = records or MIT_BIH_RECORDS

    if not force:
        to_download = get_missing_records(records, target_dir)
        already_have = len(records) - len(to_download)
        if already_have > 0:
            print(f"[i] Đã tìm thấy {already_have}/{len(records)} bản ghi trong '{target_dir}'.")
    else:
        to_download = list(records)

    if not to_download:
        print("[+] Toàn bộ bản ghi yêu cầu đã có sẵn trên máy!")
        return

    print(f"[*] Bắt đầu tải {len(to_download)} bản ghi MIT-BIH từ PhysioNet về: {target_dir}")
    print(f"    Danh sách tải: {', '.join(to_download)}")
    wfdb.dl_database('mitdb', dl_dir=str(target_dir), records=to_download)
    print("[+] Hoàn tất tải dữ liệu MIT-BIH Arrhythmia Database!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dữ liệu MIT-BIH Arrhythmia từ PhysioNet")
    parser.add_argument(
        "--aami-only",
        action="store_true",
        help="Chỉ tải 44 bản ghi chuẩn AAMI (bỏ qua 102, 104, 107, 217)"
    )
    parser.add_argument(
        "--records",
        nargs="+",
        default=None,
        help="Danh sách mã bản ghi cụ thể cần tải (ví dụ: --records 100 101 200)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải đè lại kể cả khi file đã tồn tại"
    )
    args = parser.parse_args()

    if args.records:
        selected_records = args.records
    elif args.aami_only:
        selected_records = [r for r in MIT_BIH_RECORDS if r not in AAMI_EXCLUDED_RECORDS]
    else:
        selected_records = MIT_BIH_RECORDS

    download_dataset(records=selected_records, force=args.force)
