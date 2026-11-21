"""
智能客服问答界面（Streamlit）
=============================

基于 Streamlit 的智能客服问答界面，支持：
- 多轮对话
- 流式回答输出
- 会话历史展示
- RAG 检索增强回答

运行方式（从项目根目录执行）:
    streamlit run app_qa.py
"""

import streamlit as st

from rag.rag import RagService
from config import CONFIG

st.title("智能客服")
st.divider()

# 初始化消息历史
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"}]

# 初始化 RAG 服务（缓存在 session_state 避免每次 rerun 重建）
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 展示对话历史
for message in st.session_state["message"]:
    st.chat_message(message["role"],).write(message["content"])

# 用户输入
prompt = st.chat_input()

if prompt:
    # 展示用户消息
    st.chat_message("user",).write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 调用 RAG 服务获取回答
    ai_res_list = []
    with st.spinner("思考中..."):
        # session_config 通过 RunnableWithMessageHistory 的 config 参数传入，
        # 用于指定本次对话的 session_id，从而隔离不同会话的历史
        res = st.session_state["rag"].chain.stream(
            {"input": prompt},
            config={"configurable": {"session_id": CONFIG.rag_session_id}},
        )

        # 捕获流式输出并展示
        def capture(generation, cache_list):
            for chunk in generation:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant",).write_stream(capture(res, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
