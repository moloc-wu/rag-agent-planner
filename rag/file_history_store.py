"""
会话历史存储模块
================

该模块提供基于文件的会话历史持久化功能，支持：
- 会话历史的读写操作
- 多会话隔离（每个 session_id 对应一个 JSON 文件）
- JSON 格式存储（跨进程、跨重启可恢复）

核心组件：
- FileChatMessageHistory: 基于文件的会话历史存储类
- get_history: 获取指定会话的历史记录（供 RunnableWithMessageHistory 使用）

使用方式（包外）:
    from rag.file_history_store import get_history

    history = get_history("session001")
    history.add_message(message)
"""

import os
import json
from typing import Sequence, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict
from config import CONFIG


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件的会话历史存储类

    将会话历史持久化到本地 JSON 文件，支持跨会话保持历史记录。
    每个会话对应一个 JSON 文件，文件名格式为 <session_id>.json。

    继承 BaseChatMessageHistory 是为了对接 LangChain 的
    RunnableWithMessageHistory，后者会自动调用 add_message / messages。

    参数:
        session_id: 会话标识符
        storage_path: 存储目录路径，默认使用 CONFIG.rag_chat_history_directory
    """

    def __init__(self, session_id: str, storage_path: str | None = None):
        self.session_id = session_id
        # 路径解析：
        # - 显式传 storage_path 则用之
        # - 否则用 CONFIG.rag_chat_history_directory（统一管理，不再硬编码）
        self.storage_path = storage_path or CONFIG.rag_chat_history_directory
        self.file_path = os.path.join(self.storage_path, f"{session_id}.json")
        os.makedirs(self.storage_path, exist_ok=True)

    def add_message(self, message: BaseMessage) -> None:
        """添加单条消息到历史记录

        实现策略：读取全部 → 追加新消息 → 全量覆写。
        简单可靠，适合中小规模历史；超大量历史应改用数据库。
        """
        all_messages = list(self.messages)
        all_messages.append(message)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(all_messages), f, ensure_ascii=False)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """添加多条消息到历史记录"""
        all_messages = list(self.messages)
        all_messages.extend(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(all_messages), f, ensure_ascii=False)

    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有历史消息

        文件不存在时返回空列表（首次对话场景）
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                message_data = json.load(f)
                return messages_from_dict(message_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        """清空所有历史消息（写入空数组）"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)


def get_history(session_id: str) -> FileChatMessageHistory:
    """
    获取指定会话的历史记录

    这是 RunnableWithMessageHistory 需要的工厂函数：
    每次调用链时，LangChain 会用 session_id 调用本函数拿到 history 实例，
    再自动把历史消息塞进 prompt 的 history 占位符。

    参数:
        session_id: 会话标识符，用于区分不同会话

    返回:
        FileChatMessageHistory: 会话历史对象
    """
    return FileChatMessageHistory(session_id)
