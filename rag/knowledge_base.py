"""
知识库管理模块
==============

该模块提供知识库的上传、去重和存储功能，支持：
- 文本内容上传（自动分割成 chunk）
- MD5 去重检测（避免同一文本重复入库）
- 文本分割（RecursiveCharacterTextSplitter）
- 向量数据库存储（Chroma）

核心组件：
- KnowledgeBase: 知识库管理类
- check_md5: 检查 MD5 是否已存在
- save_md5: 保存 MD5 到文件
- get_string_md5: 计算字符串的 MD5 值

使用方式（包外）:
    from rag.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    result = kb.upload_by_str("文本内容", "filename.txt")
"""

import os
import hashlib
from datetime import datetime

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG


# ==================== MD5 去重工具函数 ====================

def check_md5(md5_str: str) -> bool:
    """检查 MD5 值是否已存在于记录文件中

    原理：每入库一段文本，就把它的 MD5 追加到 md5_path 文件。
    下次入库前先扫一遍文件，命中则跳过，避免重复向量化。

    参数:
        md5_str: MD5 字符串

    返回:
        bool: True 表示已存在（跳过），False 表示不存在（可入库）
    """
    # 文件不存在则先创建空文件，并返回 False
    if not os.path.exists(CONFIG.rag_md5_path):
        open(CONFIG.rag_md5_path, "w", encoding="utf-8").close()
        return False

    # 逐行比对，命中即返回 True
    with open(CONFIG.rag_md5_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip() == md5_str:
                return True
    return False


def save_md5(md5_str: str) -> None:
    """保存 MD5 值到记录文件（追加模式）

    参数:
        md5_str: MD5 字符串
    """
    with open(CONFIG.rag_md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    """计算字符串的 MD5 值

    参数:
        input_str: 输入字符串
        encoding: 编码格式，默认 utf-8

    返回:
        str: MD5 十六进制字符串（32 位）
    """
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5(str_bytes)
    return md5_obj.hexdigest()


# ==================== 知识库主类 ====================

class KnowledgeBase(object):
    """
    知识库管理类

    提供文本上传、去重检测、文本分割和向量存储功能。
    一个 KnowledgeBase 实例对应一个 Chroma 集合。
    """

    def __init__(self):
        """初始化知识库

        - 创建持久化目录（若不存在）
        - 初始化 Chroma 客户端
        - 初始化文本分割器（按 separators 优先级递归切分）
        """
        os.makedirs(CONFIG.rag_persist_directory, exist_ok=True)

        self.chrom = Chroma(
            collection_name=CONFIG.rag_collection_name,
            embedding_function=DashScopeEmbeddings(model=CONFIG.rag_embedding_model_name),
            persist_directory=CONFIG.rag_persist_directory,
        )

        # RecursiveCharacterTextSplitter 会按 separators 顺序递归切分：
        # 先尝试按 "\n\n" 切，切完仍超 chunk_size 再按 "\n" 切，依此类推
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.rag_chunk_size,
            chunk_overlap=CONFIG.rag_chunk_overlap,
            separators=CONFIG.rag_separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        """上传文本内容到知识库

        流程:
            1. 计算文本 MD5
            2. 检查是否已存在（去重）
            3. 超过 max_split_char_number 则分割成 chunk
            4. 携带 metadata 写入 Chroma（自动向量化并落盘）

        参数:
            data: 文本内容
            filename: 文件名（写入 metadata.source，便于追溯来源）

        返回:
            str: "已存在" 或 "上传成功"
        """
        # 1. MD5 去重
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "已存在"

        # 2. 文本分割：长文本切块，短文本整段入库
        if len(data) > CONFIG.rag_max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 3. 元数据：每个 chunk 都附带同样的来源信息，便于后续过滤
        metadata = {
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": filename,
            "operator": "小吴",
        }

        # 4. 写入 Chroma（内部会调用 embedding_function 把文本转向量）
        self.chrom.add_texts(
            knowledge_chunks,
            metadata=[metadata for _ in knowledge_chunks],
        )

        # 5. 记录 MD5，下次同内容自动跳过
        save_md5(md5_hex)
        return "上传成功"


if __name__ == "__main__":
    # 自测：往知识库写入一条测试文本
    # 注意：直接运行本文件时，需要从项目根目录执行 python -m rag.knowledge_base
    # 或先 sys.path.insert 项目根目录
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    kb = KnowledgeBase()
    kb.upload_by_str("你好", "test.txt")
