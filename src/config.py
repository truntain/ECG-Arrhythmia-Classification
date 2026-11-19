"""
Cấu hình hằng số dùng chung cho toàn bộ dự án phân loại rối loạn nhịp tim ECG (Group 27).
"""
from pathlib import Path

# Đường dẫn thư mục gốc của dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Thông số tín hiệu MIT-BIH Arrhythmia Database
SAMPLING_RATE = 360  # Hz
WINDOW_BEFORE_R = 90   # Số mẫu trước đỉnh R (~250 ms)
WINDOW_AFTER_R = 160   # Số mẫu sau đỉnh R (~444 ms)
BEAT_LENGTH = WINDOW_BEFORE_R + WINDOW_AFTER_R  # 250 mẫu / nhịp

# Danh sách 48 bản ghi chuẩn của MIT-BIH
MIT_BIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# 4 bản ghi chứa nhịp máy tạo nhịp (paced beats) thường được loại trừ theo chuẩn AAMI EC57
AAMI_EXCLUDED_RECORDS = ['102', '104', '107', '217']

# Ánh xạ nhãn gốc MIT-BIH sang 5 lớp chuẩn AAMI EC57
AAMI_MAPPING = {
    # N: Normal & Bundle branch block beats
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S: Supraventricular ectopic beats (SVEB)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V: Ventricular ectopic beats (VEB)
    'V': 'V', 'E': 'V',
    # F: Fusion beats
    'F': 'F',
    # Q: Unknown / Paced / Unclassifiable beats
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

# Ánh xạ 5 lớp AAMI sang số nguyên để huấn luyện mô hình
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

CLASS_NAMES = {
    'N': 'Normal (Bình thường)',
    'S': 'Supraventricular Ectopic (Ngoại tâm thu trên thất)',
    'V': 'Ventricular Ectopic (Ngoại tâm thu thất - Nguy hiểm)',
    'F': 'Fusion Beat (Nhịp hỗn hợp)',
    'Q': 'Unknown / Paced (Không xác định)'
}

RANDOM_SEED = 42
