import os
import asyncio
import streamlit as st
import ollama
import logging
import numpy as np
from typing import Dict, Tuple

# 🚀 引入 LightRAG 核心组件与官方异步 Ollama 客户端
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete
from ollama import AsyncClient

from config import settings
from lightrag_import import lightrag_config
from lightrag_runtime import AsyncLightRAGClient
from logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ==========================================
# 1. 🚀 核心组件：鲁棒性更强的 Ollama 向量化函数
# ==========================================
async def ollama_embed_core(texts: list[str], embed_model: str = "nomic-embed-text:latest", **kwargs) -> np.ndarray:
    """封装官方 AsyncClient，支持批量向量化并确保 1-1 数量对应"""
    client = AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # 优先使用新版支持批量传入的 client.embed 接口
    if hasattr(client, "embed"):
        response = await client.embed(model=embed_model, input=texts)
        raw_embeddings = response.get("embeddings", [])

    # 如果是老版 client.embeddings 接口，则使用 asyncio.gather 并发单条处理
    elif hasattr(client, "embeddings"):
        async def get_single_embedding(text: str):
            res = await client.embeddings(model=embed_model, prompt=text, options={"num_ctx": 8192})
            return res.get("embedding", []) or res.get("embeddings", [])

        # 并发执行所有文本的 embedding 请求，确保顺序和数量完全一致
        raw_embeddings = await asyncio.gather(*(get_single_embedding(t) for t in texts))

    else:
        raise AttributeError("本地安装的 ollama 库接口不匹配。")

    return np.array(raw_embeddings)


# 用 LightRAG 期望的 EmbeddingFunc dataclass 容器包裹它
safe_ollama_embed = EmbeddingFunc(
    embedding_dim=768,  # 💡 严格对应本地 nomic-embed-text 模型的 768 维度
    max_token_size=8192,
    func=ollama_embed_core
)


# ==========================================
# 2. 🚀 初始化与缓存 LightRAG 客户端
# ==========================================
@st.cache_resource
def get_lightrag_client():
    """创建已初始化的本地 LightRAG 运行时，供 Streamlit 同步代码调用。"""
    return AsyncLightRAGClient(
        lambda: LightRAG(
            working_dir="medical_lightrag_db",
            embedding_func=safe_ollama_embed,
            llm_model_func=ollama_model_complete,
            **lightrag_config(),
        )
    )

def generate_prompt(query: str) -> Tuple[str, str, Dict[str, str]]:
    """从 LightRAG 召回上下文并组装回答提示。"""

    # 在缓存运行时的同一事件循环中完成初始化后的 aquery 调用。
    lightrag_context = get_lightrag_client().query_local_context(query)

    retrieval_status = {'检索状态': '已通过 LightRAG 召回本地医疗知识库'}

    # 组装最终喂给 Ollama 的系统指令
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>"
    prompt += "<指令>请你仅针对医疗类问题提供简洁和专业的回答。如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”</指令>"

    # 将 LightRAG 召回的混合图文本动态编入
    prompt += f"<提示>{lightrag_context}</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>请极其认真的整合提示里的知识，不可利用自身存在的旧知识进行回答，必须从提示中找到问题的答案！如果根据提示无法给出答案，你必须回答“根据已知信息无法回答该问题”。</注意>"

    logger.debug('LightRAG 拼装的最终 prompt: %s', prompt)
    return prompt, lightrag_context or '', retrieval_status


def main(is_admin: bool, usname: str) -> None:
    """Streamlit 主界面入口"""
    st.title(f"医疗智能问答机器人")

    with st.sidebar:
        col1, col2 = st.columns([0.6, 0.6])
        with col1:
            st.image(os.path.join("img", "logo.jpg"), use_container_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button('新建对话窗口'):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('请选择对话窗口:', window_options)
        active_window_index = int(selected_window.split()[1]) - 1

        st.caption(f"当前模型：{settings.OLLAMA_QWEN_MODEL}")

        show_retrieval = show_prompt = False
        if is_admin:
            show_retrieval = st.sidebar.checkbox("显示检索状态")
            show_prompt = st.sidebar.checkbox("显示查询的知识库信息")

        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    current_messages = st.session_state.messages[active_window_index]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if show_retrieval:
                    with st.expander("检索状态"):
                        st.write(message.get("retrieval", ""))
                if show_prompt:
                    with st.expander("点击显示知识库信息"):
                        st.write(message.get("prompt", ""))

    if query := st.chat_input("Ask me anything!", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        response_placeholder = st.empty()
        response_placeholder.text("正在进行图谱语义检索...")

        prompt, knowledge_context, retrieval_status = generate_prompt(query)

        last = ""
        for chunk in ollama.chat(
            model=settings.OLLAMA_QWEN_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        ):
            last += chunk['message']['content']
            response_placeholder.markdown(last)
        response_placeholder.markdown("")

        # 从生成的提示词中把用于回显的上下文剥离出来显示给前台
        zhishiku_content = f"图谱召回上下文:\n{knowledge_context}"

        with st.chat_message("assistant"):
            st.markdown(last)
            if show_retrieval:
                with st.expander("检索状态"):
                    st.write(retrieval_status)
            if show_prompt:
                with st.expander("点击显示知识库信息"):
                    st.write(zhishiku_content)

        current_messages.append({"role": "assistant", "content": last, "retrieval": retrieval_status, "prompt": zhishiku_content})

    st.session_state.messages[active_window_index] = current_messages
