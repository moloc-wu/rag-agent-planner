"""
RAG 前置检索适配器。

作为 RAG 与 Agent 融合的桥梁层，封装 RagService 提供简洁的检索接口，
供 Planner 在调用 LLM 前进行前置检索，将知识库内容作为参考资料注入。

流程:
    用户输入
        ↓
    RagRetriever.retrieve(user_input)
        ↓ 返回参考资料文本
    拼接到用户输入前作为上下文
        ↓
    TripPlanner.invoke / stream
        ↓
    生成最终旅行计划
"""

from rag.rag import RagService

_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    """获取全局共享的 RagService 实例（懒加载单例）。

    嵌入模型与向量库初始化有一定开销，使用单例避免重复创建。

    返回:
        RagService: 全局唯一的 RAG 服务实例。
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


class RagRetriever:
    """RAG 前置检索器（供 Agent 使用）。

    封装 RagService.retrieve，增加容错与空结果处理。
    仅提供检索与格式化能力，不暴露 LLM 调用。

    示例:
        >>> retriever = RagRetriever()
        >>> context = retriever.retrieve("长沙3日游 自然风光")
        >>> # context 可直接拼接到 Agent 的 prompt 中
    """

    def __init__(self):
        """初始化，获取全局共享的 RagService。"""
        self.rag = get_rag_service()

    def retrieve(self, query: str) -> str:
        """检索知识库，返回格式化的参考资料文本。

        参数:
            query: 检索查询字符串，通常为用户原始输入。

        返回:
            str: 参考资料文本。若知识库无内容或检索失败，返回空字符串。

        容错策略:
            - 检索抛异常时捕获并返回空串，不阻断 Agent 主流程。
            - 知识库无相关内容时返回空串，避免污染 prompt。
        """
        try:
            result = self.rag.retrieve(query)
        except Exception as e:
            print(f"[RagRetriever] 检索失败，跳过 RAG 增强: {e}")
            return ""

        if not result or result == "无相关参考资料":
            return ""

        return result


if __name__ == "__main__":
    r = RagRetriever()
    print("检索结果:")
    print(r.retrieve("长沙 历史文化 景点"))
