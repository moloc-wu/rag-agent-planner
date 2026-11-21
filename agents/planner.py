"""
行程规划总控 Agent。

持有子 Agent 作为工具，统一编排并流式输出最终行程。

RAG 融合:
    在调用 LLM 之前，先用 RagRetriever 检索知识库（用户上传的攻略、笔记等），
    将检索到的参考资料作为上下文拼接到用户输入前，使 Planner 在生成规划时
    能够参考知识库内容。
"""

import re
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel

from mcp_client import McpClientManager
from agents.specialist import SpecialistAgent
from rag_retriever import RagRetriever
from prompts import (
    HOTEL_AGENT_PROMPT,
    ATTRACTION_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
)

TOOL_LABELS = {
    "query_weather":     ("🌤️", "查询天气"),
    "search_hotel":      ("🏨", "搜索酒店"),
    "search_attraction": ("🏛️", "搜索景点"),
    "maps_direction_walking_by_address":             ("🚶", "规划步行路线"),
    "maps_direction_driving_by_address":             ("🚗", "规划驾车路线"),
    "maps_direction_transit_integrated_by_address":  ("🚌", "规划公交路线"),
}

_TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:[^\]]*\]")


class TripPlanner:
    """旅行规划总控智能体。

    架构:
        用户输入
          ↓
        RagRetriever.retrieve()          ← RAG 前置检索（融合点）
          ↓ 返回参考资料文本
        拼接系统上下文 + 用户输入
          ↓
        Planner（总控 Agent）
          ├── search_hotel      → HotelAgent      → MCP: maps_text_search
          ├── search_attraction → AttractionAgent → MCP: maps_text_search
          ├── query_weather     → WeatherAgent    → MCP: maps_weather
          └── maps_direction_*  → 直接 MCP 路线工具

    示例:
        >>> planner = TripPlanner(llm)
        >>> await planner.build()
        >>> result = await planner.invoke("杭州3日游...")
        >>>
        >>> async for token in planner.stream("杭州3日游..."):
        ...     print(token, end="", flush=True)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.mcp = McpClientManager()

        self._rag_retriever: RagRetriever | None = None

        self._hotel_agent: SpecialistAgent | None = None
        self._attraction_agent: SpecialistAgent | None = None
        self._weather_agent: SpecialistAgent | None = None

        self._agent = None

    async def build(self):
        """初始化所有子 Agent、RAG 检索器，并组装 Planner。"""
        if self._agent is not None:
            return

        try:
            self._rag_retriever = RagRetriever()
        except Exception as e:
            print(f"[TripPlanner] RAG 检索器初始化失败，将跳过 RAG 增强: {e}")
            self._rag_retriever = None

        poi_tools = await self.mcp.get_tools_for("poi")
        weather_tools = await self.mcp.get_tools_for("weather")
        route_tools = await self.mcp.get_tools_for("route")

        self._hotel_agent = SpecialistAgent(
            self.llm, "HotelAgent", HOTEL_AGENT_PROMPT, poi_tools
        )
        self._attraction_agent = SpecialistAgent(
            self.llm, "AttractionAgent", ATTRACTION_AGENT_PROMPT, poi_tools
        )
        self._weather_agent = SpecialistAgent(
            self.llm, "WeatherAgent", WEATHER_AGENT_PROMPT, weather_tools
        )
        await self._hotel_agent.build()
        await self._attraction_agent.build()
        await self._weather_agent.build()

        @tool
        async def search_hotel(query: str) -> str:
            """搜索酒店。输入城市+偏好，返回酒店列表。"""
            return await self._hotel_agent.invoke(query)

        @tool
        async def search_attraction(query: str) -> str:
            """搜索景点。输入城市+类型偏好，返回景点列表。"""
            return await self._attraction_agent.invoke(query)

        @tool
        async def query_weather(query: str) -> str:
            """查询天气。输入城市+日期，返回天气概况。"""
            return await self._weather_agent.invoke(query)

        all_tools = [search_hotel, search_attraction, query_weather, *route_tools]

        self._agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=PLANNER_AGENT_PROMPT,
        )

    def _augment_input(self, user_input: str) -> str:
        """用 RAG 检索结果增强用户输入。

        流程:
            1. 以 user_input 为查询词检索知识库。
            2. 若存在相关参考资料，拼接为「参考资料 + 用户需求」的复合输入。
            3. 若无相关资料或检索器不可用，原样返回 user_input。

        参数:
            user_input: 用户原始输入。

        返回:
            str: 增强后的输入文本。
        """
        if self._rag_retriever is None:
            return user_input

        context = self._rag_retriever.retrieve(user_input)
        if not context:
            return user_input

        return (
            f"【参考资料】（来自知识库，请优先参考其内容进行规划，"
            f"但天气/酒店/景点等实时信息仍以工具调用结果为准）:\n"
            f"{context}\n\n"
            f"【用户需求】:\n{user_input}"
        )

    async def invoke(self, user_input: str) -> str:
        """输入自然语言需求，返回完整旅行计划。

        流程:
            1. RAG 前置检索，增强输入。
            2. 调用 Planner Agent 生成规划。
        """
        await self.build()
        augmented_input = self._augment_input(user_input)
        result = await self._agent.ainvoke({
            "messages": [{"role": "user", "content": augmented_input}]
        })
        return result["messages"][-1].content

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """流式输出旅行计划。

        逐 token yield 文本及工具调用状态标记：
          - 普通 token: 直接 yield（过滤子 Agent 内部 TOOL_CALL 泄漏）
          - 工具开始:   yield 状态标记行
          - 工具结束:   静默处理
        """
        await self.build()
        augmented_input = self._augment_input(user_input)

        async for event in self._agent.astream_events(
            {"messages": [{"role": "user", "content": augmented_input}]},
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    content = _TOOL_CALL_PATTERN.sub("", content)
                    if content.strip():
                        yield content

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                emoji, label = TOOL_LABELS.get(name, ("🔧", name))
                yield f"\n{emoji} {label}...\n"

            elif kind == "on_tool_end":
                name = event.get("name", "unknown")
                if name in TOOL_LABELS:
                    pass
