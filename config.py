"""
全局配置中心。

统一管理 API 密钥、LLM 实例、MCP 连接参数、RAG 知识库参数等，
通过模块级 CONFIG 单例提供访问。

示例:
    >>> from config import CONFIG
    >>> llm = CONFIG.create_llm()
    >>> embedding = CONFIG.create_embedding()
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()


def _patched_subtract(self, resp, prev_resp):
    """修复 ChatTongyi 流式 tool_calls 的 KeyError。

    上游 Bug：subtract_client_response 在访问 prev_function["name"] /
    ["arguments"] 前没有检查 key 是否存在，而流式首个 tool_call 分片
    可能不含这些 key，导致 KeyError。

    参数:
        self: ChatTongyi 实例。
        resp: 当前响应字典。
        prev_resp: 前一个响应字典。

    返回:
        dict: 计算增量后的响应（包含内容增量和 tool_calls 增量）。
    """
    import json

    resp_copy = json.loads(json.dumps(resp))
    message = resp_copy["output"]["choices"][0]["message"]
    prev_message = json.loads(json.dumps(prev_resp))["output"]["choices"][0]["message"]

    message["content"] = message["content"].replace(
        prev_message.get("content", "") or "", ""
    )

    if message.get("tool_calls") and prev_message.get("tool_calls"):
        for index, tool_call in enumerate(message["tool_calls"]):
            if index >= len(prev_message["tool_calls"]):
                continue
            function = tool_call["function"]
            prev_function = prev_message["tool_calls"][index]["function"]

            if "name" in function and "name" in prev_function:
                function["name"] = function["name"].replace(prev_function["name"], "")
            if "arguments" in function and "arguments" in prev_function:
                function["arguments"] = function["arguments"].replace(
                    prev_function["arguments"], ""
                )

    return resp_copy


ChatTongyi.subtract_client_response = _patched_subtract


@dataclass
class Config:
    """全局配置单例。

    所有配置项在导入时从环境变量读取，可通过 ``.env`` 文件覆盖。
    """

    api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )

    model_name: str = "qwen-max"
    temperature: float = 0.7

    mcp_transport: str = "http"
    mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"

    tool_domains: dict = field(default_factory=lambda: {
        "poi":     ["maps_text_search", "maps_search_detail"],
        "weather": ["maps_weather"],
        "route":   [
            "maps_direction_walking_by_address",
            "maps_direction_driving_by_address",
            "maps_direction_transit_integrated_by_address",
        ],
    })

    rag_md5_path: str = "./data/md5.txt"
    rag_collection_name: str = "rag"
    rag_persist_directory: str = "./data/chroma_db"
    rag_chat_history_directory: str = "./data/chat_history"
    rag_chunk_size: int = 1024
    rag_chunk_overlap: int = 128
    rag_separators: list = field(default_factory=lambda: ["\n", "\n\n", ".", "!", "?", "。", "？", "!"])
    rag_max_split_char_number: int = 1000
    rag_retrieval_top_k: int = 3
    rag_embedding_model_name: str = "text-embedding-v1"
    rag_session_id: str = "001"

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

    def create_llm(self) -> ChatTongyi:
        """创建对话模型实例。

        返回:
            ChatTongyi: 已配置的流式对话模型。
        """
        return ChatTongyi(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            streaming=True,
        )

    def create_embedding(self) -> DashScopeEmbeddings:
        """创建嵌入模型实例。

        返回:
            DashScopeEmbeddings: 已配置的嵌入模型。
        """
        return DashScopeEmbeddings(model=self.rag_embedding_model_name)


CONFIG = Config()
