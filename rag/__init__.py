"""
RAG 检索增强生成子包
====================

本包封装所有与向量检索、知识库管理、对话历史相关的功能：

- vector_stores:    Chroma 向量数据库封装
- knowledge_base:   知识库上传/去重/分割
- file_history_store: 文件持久化会话历史
- rag:              RAG 服务主类（含 retrieve 前置检索方法）
- app_qa:           Streamlit 智能客服问答界面
- app_file_uploader: Streamlit 知识库上传界面

包内模块互相导入统一使用相对导入（from .module import xxx），
外部访问使用绝对导入（from rag.rag import RagService）。
"""

# 暴露包级别的常用接口，方便外部直接 from rag import RagService
from .rag import RagService
from .vector_stores import VectorStoreService
from .knowledge_base import KnowledgeBase

__all__ = ["RagService", "VectorStoreService", "KnowledgeBase"]
