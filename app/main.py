"""
Module 5: Giao diện trực quan hóa tín hiệu ECG & Cảnh báo bất thường
(Phụ trách: Đào Trọng Công - 20235665)
Chạy ứng dụng: streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title="ECG Arrhythmia Classification - Group 27",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Hệ thống Phân tích Tín hiệu ECG & Cảnh báo Rối loạn Nhịp tim")
st.caption("Đề tài Nhập môn Học máy - Nhóm 27 | Bộ dữ liệu chuẩn MIT-BIH Arrhythmia (5 lớp AAMI: N, S, V, F, Q)")
st.info("Giao diện đang trong quá trình tích hợp với pipeline tiền xử lý và mô hình học máy.")
