"""
向量存储服务模块
================

该模块封装了基于 Chroma 的向量数据库操作，提供：
- 向量数据库初始化（与持久化目录绑定）
- 检索器获取（Retriever，供 RAG 链调用）

设计要点：
- Chroma 是一个本地嵌入式向量数据库，无需启动服务，数据落盘到 persist_directory
- embedding_function 负责把"文本"翻译成"向量"，相似度检索就靠它来计算
- 配置统一来自根目录 config.py 的 CONFIG

使用方式（包外）:
    from rag.vector_stores import VectorStoreService
    from config import CONFIG

    service = VectorStoreService(embedding=CONFIG.create_embedding())
    retriever = service.get_retriever()

使用方式（包内相对导入）:
    from .vector_stores import VectorStoreService
"""

from langchain_chroma import Chroma
from config import CONFIG


class VectorStoreService(object):
    """
    向量存储服务类

    封装了 Chroma 向量数据库的初始化和检索功能。
    一个 VectorStoreService 实例对应一个 Chroma 集合（collection）。

    参数:
        embedding: 嵌入模型对象，用于文本向量化
                   （通常传 CONFIG.create_embedding()）
    """

    def __init__(self, embedding):
        """初始化向量存储服务

        参数:
            embedding: 嵌入模型对象，用于文本向量化
        """
        self.embedding = embedding
        # Chroma 实例：绑定集合名 + 嵌入函数 + 持久化目录
        # 同名集合再次打开时会自动加载已有向量，不会丢数据
        self.vector_store = Chroma(
            collection_name=CONFIG.rag_collection_name,
            embedding_function=embedding,
            persist_directory=CONFIG.rag_persist_directory,
        )

    def get_retriever(self):
        """获取检索器

        返回:
            Retriever: 向量数据库检索器，用于相似度搜索。
                       调用 retriever.invoke(query) 即可拿到 top_k 相关文档。

        说明:
            search_kwargs={"k": N} 控制返回的文档数量，
            N 越大召回越多但噪声也越多，需要平衡。
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": CONFIG.rag_retrieval_top_k}
        )
