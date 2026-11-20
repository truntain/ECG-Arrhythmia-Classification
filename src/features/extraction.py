"""
Module Trích xuất đặc trưng đa miền (Multi-domain Feature Extraction)
Phụ trách: Nguyễn Sỹ Vinh (20235875)
Trích xuất 57 đặc trưng từ mỗi nhịp tim ECG (250 mẫu quanh đỉnh R) thuộc 3 nhóm:
1. Nhóm đặc trưng Khoảng R-R (5 chiều): pre_rr, post_rr, local_rr (10 nhịp), ratio_pre_local, ratio_post_pre.
2. Nhóm đặc trưng Hình thái sóng QRS & T (11 chiều): r_amp, q_amp, s_amp, qrs_ptp, qrs_width, qrs_area,
   t_amp, beat_energy, max_pos_slope, max_neg_slope, skewness, kurtosis.
3. Nhóm đặc trưng Biến đổi Wavelet rời rạc DWT (41 chiều): Họ sóng Daubechies 4 (db4) phân rã 4 mức (level=4),
   gồm 21 hệ số xấp xỉ cA4 + 20 chỉ số thống kê (MAV, Std, Log-Energy, Shannon Entropy) trên 5 băng tần (cA4, cD4..cD1).
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis

from src.config import SAMPLING_RATE, WINDOW_BEFORE_R


def compute_rr_features(
    pre_rr_samples: np.ndarray,
    post_rr_samples: np.ndarray,
    record_ids: np.ndarray,
    fs: float = SAMPLING_RATE
) -> Tuple[np.ndarray, List[str]]:
    """
    Tính 5 đặc trưng khoảng R-R theo đơn vị giây (s) và tỷ lệ chuẩn hóa cục bộ theo từng bản ghi.
    """
    pre_rr_sec = pre_rr_samples.astype(np.float32) / fs
    post_rr_sec = post_rr_samples.astype(np.float32) / fs

    # Giới hạn giá trị ngoại lai (ví dụ khoảng nghỉ quá dài > 3 giây hoặc quá ngắn < 0.15 giây)
    pre_rr_sec = np.clip(pre_rr_sec, 0.15, 3.0)
    post_rr_sec = np.clip(post_rr_sec, 0.15, 3.0)

    # Tính Local-RR: trung bình trượt 10 nhịp xung quanh bên trong cùng 1 bản ghi (record_id)
    df_rr = pd.DataFrame({"rec": record_ids, "pre_rr": pre_rr_sec})
    local_rr_sec = (
        df_rr.groupby("rec")["pre_rr"]
        .transform(lambda s: s.rolling(window=10, min_periods=1, center=True).mean())
        .to_numpy(dtype=np.float32)
    )

    eps = 1e-6
    ratio_pre_local = pre_rr_sec / (local_rr_sec + eps)
    ratio_post_pre = post_rr_sec / (pre_rr_sec + eps)

    rr_matrix = np.column_stack([
        pre_rr_sec,
        post_rr_sec,
        local_rr_sec,
        ratio_pre_local,
        ratio_post_pre
    ]).astype(np.float32)

    names = [
        "rr_pre_sec",
        "rr_post_sec",
        "rr_local10_sec",
        "rr_ratio_pre_local",
        "rr_ratio_post_pre"
    ]
    return rr_matrix, names


def compute_morphological_features(
    X_beats: np.ndarray,
    r_idx: int = WINDOW_BEFORE_R,
    fs: float = SAMPLING_RATE
) -> Tuple[np.ndarray, List[str]]:
    """
    Tính 11 đặc trưng hình thái thời gian (QRS complex & T-wave morphology) theo phương pháp vector hóa
    trên toàn bộ ma trận X_beats (N_samples, 250).
    """
    # Vùng cửa sổ quanh đỉnh R (r_idx = 90)
    q_window = X_beats[:, r_idx - 25: r_idx]        # [-69ms, 0ms]: Vùng sóng Q
    r_window = X_beats[:, r_idx - 6: r_idx + 7]     # [-16ms, +16ms]: Vùng đỉnh R
    s_window = X_beats[:, r_idx: r_idx + 30]        # [0ms, +83ms]: Vùng sóng S
    qrs_window = X_beats[:, r_idx - 25: r_idx + 30] # [-69ms, +83ms]: Toàn bộ phức bộ QRS (55 mẫu)
    t_window = X_beats[:, r_idx + 35: r_idx + 140]  # [+97ms, +388ms]: Vùng sóng T

    r_amp = np.max(r_window, axis=1)
    q_amp = np.min(q_window, axis=1)
    s_amp = np.min(s_window, axis=1)
    qrs_ptp = np.ptp(qrs_window, axis=1)
    t_max_amp = np.max(t_window, axis=1)

    # Độ rộng phức bộ QRS (số mẫu có biên độ tuyệt đối vượt 25% biên độ cực đại trong vùng QRS, quy đổi ra giây)
    qrs_abs = np.abs(qrs_window)
    thresholds = 0.25 * np.max(qrs_abs, axis=1, keepdims=True)
    qrs_width_sec = np.sum(qrs_abs > thresholds, axis=1).astype(np.float32) / fs

    # Diện tích dưới đường cong QRS & Năng lượng toàn nhịp
    qrs_area = np.sum(qrs_abs, axis=1) / fs
    beat_energy = np.sum(X_beats ** 2, axis=1) / fs

    # Độ dốc lên cực đại (QR) và độ dốc xuống cực đại (RS) trong vùng QRS
    qrs_diff = np.diff(qrs_window, axis=1) * fs
    max_pos_slope = np.max(qrs_diff, axis=1)
    max_neg_slope = np.min(qrs_diff, axis=1)

    # Độ bất đối xứng (Skewness) & Độ nhọn (Kurtosis) của nhịp tim
    beat_skew = skew(X_beats, axis=1, bias=False).astype(np.float32)
    beat_kurt = kurtosis(X_beats, axis=1, bias=False).astype(np.float32)
    beat_skew = np.nan_to_num(beat_skew, nan=0.0, posinf=0.0, neginf=0.0)
    beat_kurt = np.nan_to_num(beat_kurt, nan=0.0, posinf=0.0, neginf=0.0)

    morph_matrix = np.column_stack([
        r_amp,
        q_amp,
        s_amp,
        qrs_ptp,
        qrs_width_sec,
        qrs_area,
        t_max_amp,
        beat_energy,
        max_pos_slope,
        max_neg_slope,
        beat_skew,
        beat_kurt
    ]).astype(np.float32)

    names = [
        "morph_r_amp",
        "morph_q_amp",
        "morph_s_amp",
        "morph_qrs_ptp",
        "morph_qrs_width_sec",
        "morph_qrs_area",
        "morph_t_max_amp",
        "morph_beat_energy",
        "morph_max_pos_slope",
        "morph_max_neg_slope",
        "morph_skewness",
        "morph_kurtosis"
    ]
    return morph_matrix, names


def compute_wavelet_features(
    X_beats: np.ndarray,
    wavelet: str = "db4",
    level: int = 4
) -> Tuple[np.ndarray, List[str]]:
    """
    Tính đặc trưng biến đổi Wavelet rời rạc (DWT) họ sóng db4 mức 4 trên toàn bộ ma trận X_beats (N_samples, 250).
    PyWavelets hỗ trợ truyền trực tiếp mảng 2D với axis=1 nên tốc độ tính cho 100.694 nhịp chỉ mất ~1 giây!
    - Lấy trực tiếp 21 hệ số xấp xỉ cA4 (nắm bắt hình dạng sóng tần số thấp).
    - Trên mỗi băng tần [cA4, cD4, cD3, cD2, cD1], tính 4 chỉ số thống kê:
      1. Mean Absolute Value (MAV)
      2. Standard Deviation (Std)
      3. Log10 Energy
      4. Normalized Shannon Entropy
    Tổng cộng: 21 + 5 * 4 = 41 đặc trưng Wavelet.
    """
    # Phân rã Wavelet 4 mức dọc theo chiều thời gian (axis=1)
    coeffs = pywt.wavedec(X_beats, wavelet=wavelet, level=level, axis=1)
    # coeffs = [cA4, cD4, cD3, cD2, cD1]
    cA4 = coeffs[0].astype(np.float32)
    n_cA = cA4.shape[1]

    band_names = [f"cA{level}"] + [f"cD{lv}" for lv in range(level, 0, -1)]
    stat_list = [cA4]
    feature_names = [f"dwt_cA{level}_coef_{i}" for i in range(n_cA)]

    eps = 1e-10
    for band_coef, bname in zip(coeffs, band_names):
        band_coef = band_coef.astype(np.float32)
        mav = np.mean(np.abs(band_coef), axis=1)
        std = np.std(band_coef, axis=1)

        energy_raw = np.sum(band_coef ** 2, axis=1)
        log_energy = np.log10(energy_raw + eps)

        # Shannon Entropy trên phân bố năng lượng chuẩn hóa của băng tần
        p = (band_coef ** 2) / (energy_raw[:, None] + eps)
        shannon_ent = -np.sum(p * np.log2(p + eps), axis=1)

        stat_list.append(np.column_stack([mav, std, log_energy, shannon_ent]).astype(np.float32))
        feature_names.extend([
            f"dwt_{bname}_mav",
            f"dwt_{bname}_std",
            f"dwt_{bname}_log_energy",
            f"dwt_{bname}_entropy"
        ])

    dwt_matrix = np.hstack(stat_list).astype(np.float32)
    return dwt_matrix, feature_names


def extract_all_features(
    X_beats: np.ndarray,
    pre_rr_samples: np.ndarray,
    post_rr_samples: np.ndarray,
    record_ids: np.ndarray,
    wavelet: str = "db4",
    level: int = 4
) -> Tuple[np.ndarray, List[str]]:
    """
    Hàm tổng hợp trích xuất toàn bộ 58 đặc trưng (5 RR + 12 Hình thái QRS/T + 41 Wavelet DWT).
    """
    rr_feats, rr_names = compute_rr_features(pre_rr_samples, post_rr_samples, record_ids)
    morph_feats, morph_names = compute_morphological_features(X_beats)
    dwt_feats, dwt_names = compute_wavelet_features(X_beats, wavelet=wavelet, level=level)

    X_features = np.hstack([rr_feats, morph_feats, dwt_feats]).astype(np.float32)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_names = rr_names + morph_names + dwt_names
    return X_features, all_names
