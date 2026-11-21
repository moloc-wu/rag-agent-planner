"""
知识库上传界面（Streamlit）
===========================

基于 Streamlit 的知识库文件上传界面，支持：
- 上传文本文件到知识库
- 文件信息展示（文件名、类型、大小）
- 上传结果反馈

运行方式（从项目根目录执行）:
    streamlit run app_file_uploader.py
"""

import time

import streamlit as st

from rag.knowledge_base import KnowledgeBase

st.title("知识库更新服务")
uploaded_file = st.file_uploader("上传文件", type=["pdf", "docx", "txt"], accept_multiple_files=False)

# 初始化知识库服务（缓存在 session_state 避免每次 rerun 重建）
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBase()

if uploaded_file is not None:
    # 获取文件信息
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024

    # 展示文件信息
    st.subheader(f"文件名: {file_name}")
    st.write(f"文件类型: {file_type}, 文件大小: {file_size:.2f} KB")

    # 解析文件内容（先尝试 utf-8，失败回退 gbk，兼容中文 Windows 文件）
    try:
        text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        text = uploaded_file.getvalue().decode("gbk")

    # 上传到知识库（内部会做 MD5 去重 + 文本分割 + 向量化入库）
    with st.spinner("更新中..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
