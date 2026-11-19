"""
Script tải tự động bộ dữ liệu MIT-BIH Arrhythmia Database từ PhysioNet về thư mục data/raw/.
Chạy lệnh: python data/download_mitbih.py
"""
import sys
from pathlib import Path
import wfdb

# Thêm thư mục gốc vào sys.path để import src.config
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import DATA_RAW_DIR, MIT_BIH_RECORDS


def download_dataset(records=None, target_dir=DATA_RAW_DIR):
    """Tải danh sách bản ghi MIT-BIH từ PhysioNet."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = records or MIT_BIH_RECORDS

    print(f"[*] Bắt đầu tải {len(records)} bản ghi MIT-BIH về: {target_dir}")
    wfdb.dl_database('mitdb', dl_dir=str(target_dir), records=records)
    print("[+] Hoàn tất tải dữ liệu MIT-BIH Arrhythmia Database!")


if __name__ == "__main__":
    download_dataset()
