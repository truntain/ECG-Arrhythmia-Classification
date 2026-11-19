"""
Module Ánh xạ nhãn chuẩn AAMI EC57 & Phân đoạn nhịp tim (Beat Segmentation)
Phụ trách: Trương Thế Thành (20235835)
- Tự động chọn đúng kênh chuyển đạo MLII từ bản ghi MIT-BIH.
- Ánh xạ nhãn gốc MIT-BIH sang 5 nhóm chuẩn AAMI (N, S, V, F, Q), bỏ qua ký hiệu phi nhịp tim (+, ~, |...).
- Cắt cửa sổ [-90, +160] mẫu quanh đỉnh R (250 mẫu / nhịp @ 360 Hz).
- Lưu kèm tọa độ đỉnh R (r_peaks) và khoảng cách đỉnh R liền trước/liền sau chính xác trong từng bản ghi.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import wfdb

from src.config import (
    DATA_RAW_DIR,
    SAMPLING_RATE,
    WINDOW_BEFORE_R,
    WINDOW_AFTER_R,
    BEAT_LENGTH,
    AAMI_MAPPING,
    LABEL_TO_ID
)
from src.preprocessing.denoising import bandpass_filter


def select_mlii_channel(record: wfdb.Record) -> Tuple[np.ndarray, str, int]:
    """
    Tìm và trích xuất kênh chuyển đạo MLII từ bản ghi MIT-BIH.
    Lưu ý: Hầu hết bản ghi có MLII ở kênh 0, nhưng một số bản ghi (như 114) có MLII ở kênh 1,
    và bản ghi 102, 104 (nếu không bị loại) chỉ có V5/V2.

    Returns
    -------
    signal : np.ndarray (1D)
        Tín hiệu của kênh được chọn.
    ch_name : str
        Tên kênh được chọn (ví dụ 'MLII').
    ch_idx : int
        Chỉ số kênh (0 hoặc 1).
    """
    sig_names = [name.upper().strip() for name in record.sig_name]
    if "MLII" in sig_names:
        ch_idx = sig_names.index("MLII")
        return record.p_signal[:, ch_idx].astype(np.float32), record.sig_name[ch_idx], ch_idx

    # Fallback về kênh 0 nếu bản ghi không có kênh MLII
    return record.p_signal[:, 0].astype(np.float32), record.sig_name[0], 0


def map_symbol_to_aami(symbol: str) -> Optional[str]:
    """
    Chuyển đổi ký hiệu annotation gốc của MIT-BIH sang 5 lớp chuẩn AAMI (N, S, V, F, Q).
    Trả về None nếu ký hiệu không phải là nhịp tim hợp lệ (ví dụ: '+', '~', '|', '!', '"', '[', ']').
    """
    return AAMI_MAPPING.get(symbol, None)


def segment_record_beats(
    record_id: str,
    raw_dir: Path = DATA_RAW_DIR,
    apply_filter: bool = True,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    window_before: int = WINDOW_BEFORE_R,
    window_after: int = WINDOW_AFTER_R,
    normalize_beat: bool = False
) -> Dict[str, np.ndarray]:
    """
    Đọc 1 bản ghi MIT-BIH, lọc nhiễu Bandpass (0.5-45 Hz), ánh xạ nhãn AAMI
    và cắt từng nhịp tim [-90, +160] mẫu quanh đỉnh R.

    Returns
    -------
    result : dict chứa các mảng numpy:
        - 'X_beats': (N_valid_beats, 250) float32 - các đoạn tín hiệu nhịp tim
        - 'r_peaks': (N_valid_beats,) int32 - tọa độ mẫu r_k của đỉnh R
        - 'pre_rr_samples': (N_valid_beats,) int32 - khoảng cách (mẫu) tới đỉnh R ngay trước đó
        - 'post_rr_samples': (N_valid_beats,) int32 - khoảng cách (mẫu) tới đỉnh R ngay sau đó
        - 'y_labels': (N_valid_beats,) int64 - nhãn số (0: N, 1: S, 2: V, 3: F, 4: Q)
        - 'y_symbols': (N_valid_beats,) <U1 - nhãn ký tự AAMI ('N', 'S', 'V', 'F', 'Q')
        - 'orig_symbols': (N_valid_beats,) <U2 - nhãn ký tự gốc trong MIT-BIH
        - 'record_ids': (N_valid_beats,) <U3 - mã bản ghi (ví dụ '100')
        - 'channel_name': str - tên kênh chuyển đạo đã dùng
    """
    record_path = str(Path(raw_dir) / str(record_id))
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    raw_signal, ch_name, _ = select_mlii_channel(record)
    if apply_filter:
        signal = bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=SAMPLING_RATE)
    else:
        signal = raw_signal.copy()

    sig_len = len(signal)
    ann_samples = np.asarray(annotation.sample, dtype=np.int32)
    ann_symbols = annotation.symbol

    # Bước 1: Lọc ra tất cả các đỉnh R thực sự là nhịp tim (có trong AAMI_MAPPING)
    # Việc lọc trước giúp tính Pre-RR và Post-RR giữa các nhịp tim liên tiếp chính xác tuyệt đối,
    # không bị xen giữa bởi các ký hiệu nhiễu '+', '~'.
    beat_r_locations: List[int] = []
    beat_aami_symbols: List[str] = []
    beat_orig_symbols: List[str] = []

    for samp, sym in zip(ann_samples, ann_symbols):
        aami_sym = map_symbol_to_aami(sym)
        if aami_sym is not None:
            beat_r_locations.append(int(samp))
            beat_aami_symbols.append(aami_sym)
            beat_orig_symbols.append(sym)

    num_total_beats = len(beat_r_locations)
    X_beats = []
    r_peaks = []
    pre_rr_list = []
    post_rr_list = []
    y_labels = []
    y_symbols = []
    orig_symbols = []

    # Bước 2: Cắt cửa sổ [-window_before, +window_after] quanh từng đỉnh R
    for i in range(num_total_beats):
        r_k = beat_r_locations[i]
        start_idx = r_k - window_before
        end_idx = r_k + window_after

        # Bỏ qua nhịp nằm quá sát biên đầu hoặc biên cuối của bản ghi 30 phút
        if start_idx < 0 or end_idx > sig_len:
            continue

        beat_segment = signal[start_idx:end_idx]
        if len(beat_segment) != BEAT_LENGTH:
            continue

        if normalize_beat:
            std = np.std(beat_segment)
            if std > 1e-6:
                beat_segment = (beat_segment - np.mean(beat_segment)) / std

        # Tính khoảng cách R-R (tính theo số mẫu @ 360Hz) với nhịp liền trước và liền sau
        if i > 0:
            pre_rr = r_k - beat_r_locations[i - 1]
        elif num_total_beats > 1:
            pre_rr = beat_r_locations[1] - r_k
        else:
            pre_rr = int(0.8 * SAMPLING_RATE)

        if i < num_total_beats - 1:
            post_rr = beat_r_locations[i + 1] - r_k
        elif i > 0:
            post_rr = r_k - beat_r_locations[i - 1]
        else:
            post_rr = int(0.8 * SAMPLING_RATE)

        aami_cls = beat_aami_symbols[i]
        X_beats.append(beat_segment)
        r_peaks.append(r_k)
        pre_rr_list.append(pre_rr)
        post_rr_list.append(post_rr)
        y_labels.append(LABEL_TO_ID[aami_cls])
        y_symbols.append(aami_cls)
        orig_symbols.append(beat_orig_symbols[i])

    n_valid = len(X_beats)
    return {
        "X_beats": np.asarray(X_beats, dtype=np.float32).reshape(n_valid, BEAT_LENGTH),
        "r_peaks": np.asarray(r_peaks, dtype=np.int32),
        "pre_rr_samples": np.asarray(pre_rr_list, dtype=np.int32),
        "post_rr_samples": np.asarray(post_rr_list, dtype=np.int32),
        "y_labels": np.asarray(y_labels, dtype=np.int64),
        "y_symbols": np.asarray(y_symbols, dtype="<U1"),
        "orig_symbols": np.asarray(orig_symbols, dtype="<U2"),
        "record_ids": np.full(n_valid, str(record_id), dtype="<U3"),
        "channel_name": ch_name,
    }
