"""
Module Khử nhiễu tín hiệu ECG (Denoising) - Phụ trách: Trương Thế Thành (20235835)
- Bộ lọc thông dải Bandpass Butterworth bậc 4 (0.5 - 45 Hz) với zero-phase filtering (filtfilt).
- Loại bỏ trôi đường cơ sở (< 0.5 Hz), nhiễu lưới điện (50/60 Hz) và nhiễu cơ EMG tần số cao
  mà không làm dịch chuyển vị trí đỉnh R.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from src.config import SAMPLING_RATE


def butter_bandpass(lowcut: float = 0.5, highcut: float = 45.0, fs: float = SAMPLING_RATE, order: int = 4):
    """Thiết kế hệ số bộ lọc thông dải Butterworth (b, a)."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = SAMPLING_RATE,
    order: int = 4
) -> np.ndarray:
    """
    Áp dụng bộ lọc thông dải Butterworth bậc 4 hai chiều (zero-phase filtfilt) lên tín hiệu ECG 1D.

    Parameters
    ----------
    signal : np.ndarray
        Mảng 1D chứa tín hiệu ECG thô.
    lowcut : float
        Tần số cắt dưới (Hz), mặc định 0.5 Hz để khử trôi đường cơ sở (baseline wander).
    highcut : float
        Tần số cắt trên (Hz), mặc định 45.0 Hz để khử nhiễu điện lưới 50/60Hz và nhiễu cơ EMG.
    fs : float
        Tần số lấy mẫu (Hz), mặc định 360 Hz cho MIT-BIH.
    order : int
        Bậc của bộ lọc Butterworth, mặc định bậc 4.

    Returns
    -------
    filtered_signal : np.ndarray (float32)
        Tín hiệu ECG đã làm sạch, giữ nguyên pha và vị trí đỉnh R.
    """
    signal = np.asarray(signal, dtype=np.float64)
    # Xử lý trường hợp tín hiệu có chứa NaN (hiếm gặp)
    if np.isnan(signal).any():
        nans = np.isnan(signal)
        not_nans = ~nans
        signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), signal[not_nans])

    b, a = butter_bandpass(lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered.astype(np.float32)
