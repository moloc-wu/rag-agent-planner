# 🧳 智能旅行助手 (AI Travel Agent)

> 基于 Multi-Agent 架构的智能旅行规划系统，集成高德地图 MCP 服务与 RAG 知识库检索增强，支持 CLI 和 Web 双界面。输入目的地、日期和偏好，AI 自动规划包含天气、景点、酒店、餐饮、交通和预算的完整旅行方案。

---

## 目录

- [项目概览](#项目概览)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [核心模块详解](#核心模块详解)
  - [config.py — 配置中心](#configpy--配置中心)
  - [mcp_client.py — MCP 连接管理（单例模式）](#mcp_clientpy--mcp-连接管理单例模式)
  - [specialist.py — 领域专家 Agent](#specialistpy--领域专家-agent)
  - [planner.py — 总控编排 Agent](#plannerpy--总控编排-agent)
  - [rag_retriever.py — RAG 前置检索适配器](#rag_retrieverpy--rag-前置检索适配器)
  - [rag/ — 检索增强生成子包](#rag--检索增强生成子包)
  - [prompts.py — 提示词中心](#promptspy--提示词中心)
  - [render.py — 渲染引擎](#renderpy--渲染引擎)
  - [入口文件](#入口文件)
- [数据流全景](#数据流全景)
- [设计模式](#设计模式)
- [快速开始](#快速开始)
- [关键 Bug 修复记录](#关键-bug-修复记录)

---

## 项目概览

本项目实现了一个**多层 Multi-Agent 智能体系统**。用户用自然语言描述旅行需求，系统自动调用高德地图 API 查询天气、搜索景点酒店、规划路线，最终输出结构化的旅行计划 JSON + 格式化可读文本 + 可下载 Markdown。

**核心能力：**

| 能力 | 说明 |
|------|------|
| 🌤️ 天气查询 | 通过高德 MCP `maps_weather` 获取目的地实时天气预报 |
| 🏛️ 景点搜索 | 通过 `maps_text_search` 按城市+偏好搜索 POI |
| 🏨 酒店推荐 | 统一 POI 搜索，按位置推荐附近酒店 |
| 🗺️ 路线规划 | 支持步行/驾车/公交三种方式的路径规划 |
| 📊 预算汇总 | 自动汇总景点门票、酒店、餐饮、交通各项费用 |
| 📥 导出下载 | Web 界面支持 Markdown 格式下载 |

**运行方式：**

```bash
# CLI 模式（命令行输出，融合 RAG + Agent）
python cli.py

# Web 模式（Streamlit 图形界面，融合 RAG + Agent）
streamlit run app.py

# 智能客服问答（仅 RAG）
streamlit run app_qa.py

# 知识库上传（往向量库写入攻略/笔记）
streamlit run app_file_uploader.py
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户界面层                           │
│   cli.py (CLI)            app.py (Streamlit Web)         │
│   app_qa.py (客服问答)    app_file_uploader.py (知识库)   │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
┌──────────────▼──────────────────────▼────────────────────┐
│                   TripPlanner (总控 Agent)                │
│                                                          │
│   ① RAG 前置检索：RagRetriever.retrieve(user_input)      │
│      ↓ 返回知识库参考资料文本                             │
│   ② _augment_input：拼接【参考资料】+【用户需求】          │
│      ↓                                                   │
│   ③ system_prompt: PLANNER_AGENT_PROMPT                  │
│   ④ tools: [search_hotel, search_attraction,             │
│             query_weather, maps_direction_*]              │
│                                                          │
│   职责：RAG 检索 → 编排子 Agent → 整合结果 → JSON         │
└──┬──────────────┬──────────────┬─────────────────────────┘
   │              │              │
   ▼              ▼              ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌─────────────────────┐
│Hotel │  │Attrc │  │Weathr│  │ MCP 路线工具（直接）  │
│Agent │  │Agent │  │Agent │  │ walking/driving/     │
│      │  │      │  │      │  │ transit              │
└──┬───┘  └──┬───┘  └──┬───┘  └──────────┬──────────┘
   │         │         │                  │
   │    ┌────┘    ┌────┘                  │
   ▼    ▼         ▼                       │
┌─────────────────────────────────────────▼──────────────┐
│               McpClientManager (单例)                    │
│                                                         │
│   transport: http                                       │
│   url: dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp│
│                                                         │
│   ★ 工具按领域分发: poi / weather / route               │
│   ★ 懒加载 + 缓存: 首次访问时才建立连接并缓存工具列表    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              阿里百炼 MCP 服务 (高德地图)                  │
│                                                          │
│   15 个工具: maps_text_search, maps_weather,              │
│   maps_direction_*, maps_geo, maps_distance, ...         │
└──────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────┐
        │       RAG 前置检索（融合点）      │
        │                                  │
        │  RagRetriever (rag_retriever.py) │
        │       ↓                          │
        │  RagService (rag/rag.py)         │
        │       ↓ retrieve(query)          │
        │  VectorStoreService              │
        │       ↓                          │
        │  Chroma + DashScopeEmbeddings    │
        │  (data/chroma_db/)               │
        └──────────────────────────────────┘
```

**架构特点：**

1. **RAG + Agent 融合**：用户输入先经 RAG 检索知识库（用户上传的攻略/笔记），再作为参考资料喂给 Planner
2. **三层 Agent 嵌套**：Planner (总控) → SpecialistAgent (领域) → MCP Tools (底层 API)
3. **子 Agent 作为 Tool**：Hotel/Attraction/Weather Agent 被 `@tool` 装饰器包装，对 Planner 透明
4. **混合工具来源**：Planner 的工具来自子 Agent 包装 + 直接 MCP 路线工具

---

## 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| LLM | 通义千问 (qwen-max) | - | 推理与生成，通过阿里百炼 DashScope API 调用 |
| Agent 框架 | LangChain + LangGraph | latest | Agent 创建、工具编排、ReAct 循环 |
| LLM 集成 | langchain_community.ChatTongyi | latest | 通义千问的 LangChain 适配器 |
| MCP 协议 | langchain_mcp_adapters | latest | MCP 客户端，连接高德地图服务 |
| Web 界面 | Streamlit | 1.32.0 | 声明式 Web UI，侧边栏 + 主区域布局 |
| 配置管理 | python-dotenv | latest | .env 环境变量加载 |
| 运行环境 | Python 3.12+ | - | 异步 (asyncio)、类型注解 (PEP 604) |

---

## 项目结构

```
RAG+AGENT/
├── config.py                # 配置中心：API Key、模型参数、MCP 连接 + RAG 配置 + Monkey-Patch
├── mcp_client.py            # MCP 客户端管理器（单例模式）
├── prompts.py               # System Prompt 集中管理
├── render.py                # 渲染引擎：JSON 解析 + CLI 格式化
├── rag_retriever.py         # RAG 前置检索适配器（连接 RAG 与 Agent）
│
├── cli.py                   # CLI 入口（原 Agent.py，snake_case 规范化）
├── app.py                   # Streamlit Web 入口（旅行规划）
├── app_qa.py                # Streamlit 智能客服问答入口
├── app_file_uploader.py     # Streamlit 知识库上传入口
│
├── agents/                  # Agent 子包
│   ├── __init__.py
│   ├── planner.py           # 总控 Agent：编排子 Agent + RAG 前置检索 + 流式输出
│   └── specialist.py        # 领域专家 Agent：POI/天气/酒店
│
├── rag/                     # RAG 检索增强生成子包
│   ├── __init__.py          # 包标记，导出 RagService/VectorStoreService/KnowledgeBase
│   ├── rag.py               # RAG 服务主类（含 retrieve 前置检索方法）
│   ├── vector_stores.py     # Chroma 向量数据库封装
│   ├── knowledge_base.py    # 知识库上传/去重/分割（原 knowledeg_base.py，修复拼写）
│   └── file_history_store.py# 文件持久化会话历史（原 File_history_store.py，snake_case）
│
├── data/                    # 数据目录（统一管理，已 .gitignore）
│   ├── chroma_db/           # Chroma 向量数据库持久化
│   ├── chat_history/        # 会话历史 JSON 文件
│   └── md5.txt              # 知识库 MD5 去重记录
│
├── requirements.txt         # Python 依赖清单
├── .env.example             # 环境变量示例
├── .gitignore               # Git 忽略规则
└── README.md
```

---

## 核心模块详解

### config.py — 配置中心

**职责：** 全局配置单例 + 修复 langchain_community 的流式 tool_calls bug + RAG 配置整合。

**设计要点：**

- **`@dataclass`** 定义 `Config` 类，模块级 `CONFIG` 实例作为单例
- `tool_domains` 字典实现**工具领域映射**：将 15 个 MCP 工具按功能分为 poi/weather/route 三组
- `rag_*` 字段统一管理 RAG 相关配置（路径/分割/检索/嵌入模型/会话 ID），原 `RAG/config_data.py` 已并入此处
- `__post_init__` 自动校验 `DASHSCOPE_API_KEY` 是否设置
- `create_llm()` 工厂方法统一创建 `ChatTongyi` 实例，Agent 与 RAG 共用同一 LLM
- `create_embedding()` 工厂方法创建 `DashScopeEmbeddings`，供 RAG 使用

**Monkey-Patch：** 这是本项目最关键的工程修复。`langchain_community` 的 `ChatTongyi.subtract_client_response()` 方法在处理流式 tool_calls 增量时，只检查了当前 chunk 的 `function` 是否含 `name`/`arguments` key，但**未检查**前一个 chunk 的 `prev_function` 是否也含这些 key。流式 API 返回的第一个 tool_call 增量 chunk 通常只包含 `arguments` 的开头部分，不含 `name`，导致 `KeyError: 'name'`。

修复方式：在 `_patched_subtract()` 中添加守卫条件 `"name" in prev_function` 和 `"arguments" in prev_function`，然后将修补后的方法替换到 `ChatTongyi.subtract_client_response`。

---

### mcp_client.py — MCP 连接管理（单例模式）

**职责：** 管理与阿里百炼高德地图 MCP 服务器的生命周期，按领域分发工具。

**设计模式：单例模式（`__new__` + `_initialized` 标志）**

```
McpClientManager()
  ├── __new__: 保证全局只有一个实例
  ├── __init__: _initialized 标志防止重复初始化
  ├── _get_client: 懒加载 MultiServerMCPClient
  ├── get_all_tools: 首次调用时请求 MCP 工具列表并缓存
  └── get_tools_for: 领域过滤，返回子集
```

**关键实现细节：**

- **双重单例保护**：`_instance` (类变量) 确保 `__new__` 返回同一对象；`_initialized` (实例变量) 确保 `__init__` 只执行一次。两者缺一不可——`TripPlanner.__init__` 中可安全创建 `McpClientManager()`，不会产生多个实例
- **MCP 连接配置**：通过 `transport: "http"` 连接到阿里百炼托管的 MCP 服务器，认证头为 `Bearer {DASHSCOPE_API_KEY}`
- **工具缓存**：`_tools_cache` 字典按 key 缓存——`"all"` 存完整列表，全量工具只请求一次

---

### specialist.py — 领域专家 Agent

**职责：** 封装单一职责的 ReAct Agent，专门处理 POI 搜索或天气查询。

```python
class SpecialistAgent:
    def __init__(self, llm, name, system_prompt, tools):
        # 存储配置，_agent 延迟到 build() 时创建

    async def build(self):
        # create_agent(model, tools, system_prompt) → LangGraph Agent

    async def invoke(self, user_input) -> str:
        # 非流式：ainvoke → 返回最后一条消息的 content

    async def stream(self, user_input):
        # 流式：astream_events v2 → 逐 token yield
```

**设计意图：** 每个 SpecialistAgent 是一个**独立的 LangGraph Agent**，有自己的 system_prompt 和受限的工具集。例如 WeatherAgent 只持有 `maps_weather` 工具，HotelAgent 只持有 `maps_text_search` + `maps_search_detail`。这种**最小权限原则**防止 Agent 误用不属于自己领域的工具。

**System Prompt 的角色：** prompt 中要求子 Agent 必须使用工具、不要编造信息，并给出固定的工具调用格式示例。这本质上是**约束式 Prompt Engineering**——通过限制输出格式来提高可靠性。

---

### planner.py — 总控编排 Agent

**职责：** 这是整个系统的核心。TripPlanner 完成：初始化 RAG 检索器 → 加载 MCP 工具 → 创建子 Agent → 包装为 Tool → 创建 Planner Agent → 执行 RAG 前置检索 + 流式输出。

**`build()` 的五个阶段：**

```
Phase 0: 初始化 RAG 前置检索器
  self._rag_retriever = RagRetriever()    # 失败不阻断主流程

Phase 1: 按领域加载 MCP 工具
  poi_tools   = mcp.get_tools_for("poi")     # maps_text_search, maps_search_detail
  weather_tools = mcp.get_tools_for("weather")  # maps_weather
  route_tools = mcp.get_tools_for("route")   # maps_direction_*_by_address

Phase 2: 创建 3 个 SpecialistAgent
  HotelAgent(llm, "HotelAgent", HOTEL_AGENT_PROMPT, poi_tools)
  AttractionAgent(llm, "AttractionAgent", ATTRACTION_AGENT_PROMPT, poi_tools)
  WeatherAgent(llm, "WeatherAgent", WEATHER_AGENT_PROMPT, weather_tools)

Phase 3: 将子 Agent 包装为 @tool 函数
  search_hotel    → self._hotel_agent.invoke(query)
  search_attraction → self._attraction_agent.invoke(query)
  query_weather   → self._weather_agent.invoke(query)

Phase 4: 创建总控 Agent
  create_agent(llm, [search_hotel, search_attraction, query_weather, *route_tools],
               PLANNER_AGENT_PROMPT)
```

**RAG 前置检索（`_augment_input`）：** 这是 RAG + Agent 融合的核心。
- 用 `RagRetriever.retrieve(user_input)` 检索知识库
- 把检索到的参考资料拼接为【参考资料】+【用户需求】的复合输入
- 让 Planner 在生成规划时参考用户上传的攻略/笔记
- 实时信息（天气/酒店/景点）仍以工具调用结果为准

**子 Agent 作为 Tool 的设计：** 这是本项目的核心架构决策。LangChain 的 `create_agent` 原生期望工具是无状态的函数。通过 `@tool` 装饰器将 SpecialistAgent 的 `invoke()` 封装为异步函数，Planner 看到的就是三个 "黑盒工具"——它不需要知道 HotelAgent 内部也是一个 Agent。

**stream() 的事件处理：**

| 事件类型 | 处理方式 |
|---------|---------|
| `on_chat_model_stream` | 过滤 `[TOOL_CALL:...]` 模式 → yield 纯文本 token |
| `on_tool_start` | 映射到 `TOOL_LABELS` 中文标签 → yield 状态行 |
| `on_tool_end` | 静默（不输出 `[完成: xxx]`），避免刷屏 |

**`[TOOL_CALL:...]` 泄漏问题与修复：** 子 Agent 的 system prompt 要求输出 `[TOOL_CALL:amap_maps_xxx:...]` 格式。当子 Agent 通过 `ainvoke()` 返回结果时，其内部的 ReAct 循环产生的中间文本不会直接流出。但在某些模型行为下，子 Agent 的工具调用文本可能被追加到返回内容中。修复方式：在 `stream()` 中用正则 `_TOOL_CALL_PATTERN` 过滤，去除这些纯噪音文本。

---

### rag_retriever.py — RAG 前置检索适配器

**职责：** RAG 与 Agent 融合的桥梁。封装 `RagService`，提供更简洁的检索接口。

**设计要点：**

- **单例化 RagService**：通过 `get_rag_service()` 模块级单例，避免重复初始化 embedding 模型
- **容错检索**：`retrieve()` 失败时捕获异常返回空字符串，不阻断 Agent 流程
- **空结果处理**：知识库无相关内容时返回空字符串，不污染 prompt
- **不调 LLM**：只做向量检索 + 文本格式化，把生成阶段完全交给 Agent

**使用方式：**
```python
from rag_retriever import RagRetriever

retriever = RagRetriever()
context = retriever.retrieve("长沙3日游 自然风光")  # 返回字符串
```

---

### rag/ — 检索增强生成子包

**职责：** 封装所有与向量检索、知识库管理、对话历史相关的功能。

**模块清单：**

| 模块 | 职责 |
|------|------|
| `rag/__init__.py` | 包标记，导出 `RagService`/`VectorStoreService`/`KnowledgeBase` |
| `rag/rag.py` | RAG 服务主类，含 `retrieve()` 前置检索方法 |
| `rag/vector_stores.py` | Chroma 向量数据库封装 |
| `rag/knowledge_base.py` | 知识库上传/去重/分割 |
| `rag/file_history_store.py` | 文件持久化会话历史 |

**包内导入规范：**
- 包内模块互相导入使用**相对导入**（`from .file_history_store import get_history`）
- 外部访问使用**绝对导入**（`from rag.rag import RagService`）
- 不再使用 `sys.path.insert` hack（已全部移除）

**`RagService.retrieve(query)`：** 该方法是 RAG + Agent 融合的关键。
- 不调用 LLM，只做向量检索 + 文本格式化
- 返回格式化的参考资料文本（含来源标记）
- 适合作为 Agent 的前置步骤：`user_input → retrieve() → 上下文文本 → Agent 据此生成规划`

---

### prompts.py — 提示词中心

集中管理 5 个 System Prompt，便于调优和版本管理：

| 变量 | 目标 Agent | 核心指令 |
|------|-----------|---------|
| `WEATHER_AGENT_PROMPT` | WeatherAgent | 必须使用 `maps_weather` 工具，输出固定格式 |
| `ATTRACTION_AGENT_PROMPT` | AttractionAgent | 必须使用 `maps_text_search` 工具搜索景点 |
| `HOTEL_AGENT_PROMPT` | HotelAgent | 使用 `maps_text_search` 以"酒店"为关键词搜索 |
| `PLANNER_AGENT_PROMPT` | TripPlanner | 详细的 JSON 输出 Schema + 工作流程 + 预算要求 |

Planner 的 prompt 最为复杂，定义了完整的三阶段工作流（查天气 → 搜酒店景点 → 规划路线）和严格的 JSON Schema（包括 weather_info、hotel、attractions、meals、budget 的字段规范）。

---

### render.py — 渲染引擎

将 Planner 输出的 JSON 字符串转换为可视化格式：

- **`parse_plan(text)`**：从混合文本中提取 JSON（找到第一个 `{` 到最后一个 `}`），`json.loads` 解析，容错处理
- **`format_plan_cli(text)`**：生成终端友好格式——Unicode 边框标题、天气图标映射、每日行程缩进、预算汇总对齐、建议列表
- **`_weather_icon(weather)`**：天气文字→emoji 映射（晴→☀️、雨→🌧️ 等）

---

### 入口文件

项目有 4 个入口文件，统一放在项目根目录：

| 入口 | 用途 | 启动命令 |
|------|------|---------|
| `cli.py` | CLI 旅行规划（原 `Agent.py`） | `python cli.py` |
| `app.py` | Streamlit Web 旅行规划 | `streamlit run app.py` |
| `app_qa.py` | Streamlit 智能客服问答 | `streamlit run app_qa.py` |
| `app_file_uploader.py` | Streamlit 知识库上传 | `streamlit run app_file_uploader.py` |

**`cli.py` 的两种模式：**
- `demo_stream`：async for 迭代 `planner.stream()`，实时打印 token，最后调用 `format_plan_cli()` 美化输出
- `demo_invoke`：直接 `planner.invoke()` 获取完整结果，格式化输出

**`app.py` 的关键技术点：**

- **`@st.cache_resource`**：缓存 TripPlanner 实例，避免每次点击按钮都重新创建 LLM 和 MCP 连接
- **`st.session_state`**：存储 `plan_data`、`plan_raw`、`status_lines`，使规划结果在多次 rerun 间持久化
- **`asyncio.run(_collect())`**：将异步的 `planner.stream()` 转换为同步调用
- **`st.tabs()`**：每日行程用 Tab 切换，避免页面过长
- **`st.metric()`**：预算汇总用 5 列 metric 卡片展示

---

## 数据流全景

```
用户输入自然语言
      │
      ▼
┌─────────────────┐
│  build_prompt()  │  结构化参数 → 自然语言 prompt 字符串
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  TripPlanner._augment_input()           │  ← RAG 融合点
│                                         │
│  1. RagRetriever.retrieve(user_input)   │
│       ↓                                 │
│     Chroma 向量检索                     │
│       ↓                                 │
│     返回知识库参考资料文本               │
│                                         │
│  2. 拼接【参考资料】+【用户需求】         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ TripPlanner     │
│   .stream()     │
│                 │
│  Planner LLM    │  根据 system_prompt 决定调用哪些工具
│   ↓             │
│  query_weather  │  → WeatherAgent.invoke() → MCP maps_weather
│   ↓             │     返回天气数据 JSON
│  search_hotel   │  → HotelAgent.invoke() → MCP maps_text_search
│   ↓             │     返回酒店列表
│  search_attrcn  │  → AttractionAgent.invoke() → MCP maps_text_search
│   ↓             │     返回景点列表
│  maps_direction │  → 直接调用 MCP 路线工具
│   ↓             │
│  Planner LLM    │  整合所有工具结果，按 Schema 生成 JSON
│   ↓             │
│  逐 token       │  ──── stream ────→ CLI 终端 或 Web spinner
│  yield          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  parse_plan()    │  从混合文本中提取 JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  格式化渲染      │
│  · CLI: format_plan_cli()  → Unicode 边框 + emoji
│  · Web: st.markdown + CSS  → 卡片 + Tab + Metric
└─────────────────┘
```

---

## 设计模式

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **单例模式** | `McpClientManager` | `__new__` + `_initialized` 双重保护，保证全局唯一 MCP 连接 |
| **工厂方法** | `Config.create_llm()` | 统一 LLM 实例创建，隔离构造细节 |
| **门面模式** | `TripPlanner` | 对外暴露简单的 `invoke()`/`stream()` 接口，隐藏内部多 Agent 复杂性 |
| **装饰器模式** | `@tool` 包装子 Agent | 将 SpecialistAgent 适配为 LangChain Tool 接口 |
| **策略模式** | `tool_domains` 字典 | 工具按领域分组，运行时按需选择策略 |
| **模板方法** | `SpecialistAgent.build()` | 子类可覆盖构建逻辑（预留扩展点） |
| **适配器模式** | `render.py` | 将 JSON 数据适配为 CLI 格式 / Streamlit 组件两种视图 |

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n Agent python=3.12
conda activate Agent

# 安装依赖（一次到位）
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填入真实 API Key：

```bash
cp .env.example .env
```

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

> 阿里百炼 API Key 申请地址：https://dashscope.console.aliyun.com/

### 3. 运行（从项目根目录执行）

```bash
# CLI 旅行规划（融合 RAG + Agent）
python cli.py

# Web 旅行规划（融合 RAG + Agent）
streamlit run app.py

# 智能客服问答（仅 RAG）
streamlit run app_qa.py

# 知识库上传（往向量库写入攻略/笔记）
streamlit run app_file_uploader.py
```

### 4. 使用

**CLI 模式**：修改 [cli.py](cli.py) 中 `main()` 函数的 `user_input` 变量

**Web 模式**：在侧边栏填写目的地、日期、偏好，点击「开始规划」

**RAG 融合**：先用 `app_file_uploader.py` 上传旅行攻略/笔记到知识库，再运行 `cli.py` 或 `app.py`，Planner 会自动检索知识库作为参考资料

---

## 关键 Bug 修复记录

### Bug 1：ChatTongyi 流式 tool_calls 的 KeyError

**症状：** `KeyError: 'name'` at `tongyi.py:606`

**根因：** `langchain_community` 的 `subtract_client_response()` 方法未对 `prev_function` 做 key 存在性检查，流式首个 chunk 的 `prev_function` 不含 `name` key

**修复：** [config.py:16-39](config.py) — Monkey-patch `ChatTongyi.subtract_client_response`，添加 `"name" in prev_function` 和 `"arguments" in prev_function` 守卫

### Bug 2：`nonlocal` 作用域错误

**症状：** `SyntaxError: no binding for nonlocal 'full_text' found`

**根因：** `app.py` 的 `_collect()` 嵌套函数使用 `nonlocal` 引用模块级 `if` 块中的变量，而 `nonlocal` 只能引用外层**函数**作用域的变量

**修复：** 将 `_collect()` 改为只 return 结果，变量处理移到外层同步代码

### Bug 3：天气卡片文字不可见

**症状：** 天气预报卡片区域文字颜色过浅，与背景色对比度不足

**根因：** `.weather-card` CSS 只定义了 `background: #E3F2FD`，未设置 `color`，Streamlit 暗色主题下文字默认白色

**修复：** 添加 `color: #1a1a1a` 到 `.weather-card` 和 `.budget-card`
