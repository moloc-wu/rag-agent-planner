"""
RAG 检索增强生成服务模块
========================

该模块实现了基于 LangChain 的检索增强生成（RAG）服务，支持：
- 向量数据库检索（Chroma + DashScope Embeddings）
- 会话历史管理（FileChatMessageHistory 持久化到 JSON）
- 流式回答输出（对接 Streamlit write_stream）
- 参考资料格式化（page_content + metadata 拼接）

融合方案：
- 该模块的 RagService 既保留独立问答能力（app_qa.py 使用），
  也对外暴露 retrieve(query) 方法，供 Agent 前置检索使用。
- LLM 统一使用 CONFIG.create_llm()，与 Agent 共享 qwen3-max 模型。

核心组件：
- RagService: RAG 服务主类，封装完整的问答流程
- RunnableWithMessageHistory: 带会话记忆的链式调用
- VectorStoreService: 向量存储与检索服务
- FileChatMessageHistory: 文件持久化会话历史

使用方式（独立问答）:
    from rag.rag import RagService

    rag = RagService()
    result = rag.chain.invoke(
        {"input": "用户问题"},
        config={"configurable": {"session_id": "session001"}}
    )

使用方式（前置检索，给 Agent 用）:
    rag = RagService()
    context_str = rag.retrieve("长沙 3 日游 历史 文化")  # 返回字符串
"""

from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import CONFIG
from .file_history_store import get_history
from .vector_stores import VectorStoreService


class RagService(object):
    """
    RAG 检索增强生成服务类

    封装了完整的 RAG 问答流程，包括：
    1. 向量数据库检索
    2. 提示词模板构建
    3. 大语言模型调用
    4. 会话历史管理

    属性:
        vector_store_service: 向量存储服务（封装 Chroma）
        prompt_template:      聊天提示词模板（含 context / history / input 三个槽）
        chat_model:           对话 LLM（与 Agent 共用 CONFIG.create_llm()）
        chain:                完整的 RAG 链（带会话历史）
        retriever:            向量检索器（来自 vector_store_service）
    """

    def __init__(self):
        """初始化 RAG 服务

        - 创建向量存储服务（Chroma + Embeddings）
        - 构建聊天提示词模板
        - 创建对话模型（与 Agent 共用 LLM）
        - 组装 RAG 链
        """
        # 1. 向量存储服务：负责把文本变向量、做相似度检索
        self.vector_store_service = VectorStoreService(
            embedding=CONFIG.create_embedding()
        )

        # 2. 提示词模板：三个槽位
        #    - context:  检索到的参考资料（系统消息）
        #    - history:  多轮对话历史（MessagesPlaceholder 自动展开）
        #    - input:    用户当前问题
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的已知参考资料为基础，回答用户的问题，参考资料为：{context}"),
            ("system", "并且我提供用户的对话历史记录，如下："),
            MessagesPlaceholder("history"),
            ("user", "请根据以上参考资料回答用户问题：{input}")
        ])

        # 3. 对话模型：与 Agent 共用 CONFIG.create_llm()
        #    不再使用独立的 ChatTongyi(model="qwen-max")，统一为 qwen3-max
        self.chat_model = CONFIG.create_llm()

        # 4. 检索器：retriever.invoke(query) -> list[Document]
        self.retriever = self.vector_store_service.get_retriever()

        # 5. 组装完整 RAG 链
        self.chain = self._build_chain()

    # ==================== 链构建 ====================

    def _build_chain(self):
        """构建 RAG 链式调用

        链路结构（LCEL 表达式）:
            {input, context, history}
                ↓ prompt_template 填充槽位
                ↓ chat_model 生成回复
                ↓ StrOutputParser 提取纯文本
                ↓ RunnableWithMessageHistory 包一层自动历史管理

        返回:
            RunnableWithMessageHistory: 带会话记忆的完整 RAG 链
        """
        # 把 Document 列表格式化成纯字符串，塞进 prompt 的 context 槽
        def format_document(docs: list[Document]) -> str:
            """格式化检索到的文档列表

            参数:
                docs: 文档列表

            返回:
                str: 格式化后的文档内容字符串
            """
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += (
                    f"文档片段: {doc.page_content}\n"
                    f"文档源数据: {doc.metadata}\n\n"
                )
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            """从输入字典提取用于检索的文本

            参数:
                value: 输入字典（含 "input" 键）

            返回:
                str: 用户输入文本
            """
            return value["input"]

        def format_for_prompt_template(value):
            """重组字典，适配提示词模板的三个槽位

            参数:
                value: 链式调用中间结果字典（含 input / context / history）

            返回:
                dict: 包含 input、history、context 的格式化字典
            """
            return {
                "input": value["input"],
                "history": value["history"],
                "context": value["context"],
            }

        # LCEL 链：并行执行 input/context/history 三个分支，再汇入 prompt → model → parser
        chain = (
            {
                "input": RunnableLambda(lambda x: x["input"]),
                "context": RunnableLambda(format_for_retriever) | self.retriever | format_document,
                "history": RunnableLambda(lambda x: x["history"]),
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        # 包一层 RunnableWithMessageHistory，自动管理多轮对话历史
        # - input_messages_key: 输入中"用户消息"对应的键
        # - history_messages_key: prompt 模板中"历史消息"对应的占位符
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

    # ==================== 前置检索（给 Agent 用）====================

    def retrieve(self, query: str) -> str:
        """前置检索：根据 query 检索知识库，返回格式化的参考资料文本

        该方法不调用 LLM，只做向量检索 + 文本格式化，适合作为 Agent 的前置步骤：
            user_input → retrieve() → 上下文文本 → Agent 据此生成规划

        参数:
            query: 检索查询字符串（通常是用户原始输入或其改写）

        返回:
            str: 格式化后的参考资料文本；若知识库为空则返回提示字符串
        """
        docs = self.retriever.invoke(query)
        if not docs:
            return "无相关参考资料"

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"[参考{i}] 来源: {doc.metadata.get('source', '未知')}\n"
                f"内容: {doc.page_content}"
            )
        return "\n\n".join(parts)


if __name__ == "__main__":
    # 自测：用 session_id=001 跑一次问答
    # 注意：直接运行本文件需要从项目根目录执行 python -m rag.rag
    rag = RagService()
    result = rag.chain.invoke(
        {"input": "你好"},
        config={"configurable": {"session_id": CONFIG.rag_session_id}},
    )
    print(result)
