import os
import dotenv

from typing import Iterator, Union
from rag.search import doc_rag_stream

import streamlit as st

dotenv.load_dotenv()


class StreamResponse:
    def __init__(self, chunks=None):
        if chunks is None:
            chunks = []
        self.chunks = chunks
        self.__whole_msg = ""

    def generate(
            self,
            *,
            prefix: Union[str, None] = None,
            suffix: Union[str, None] = None,
    ) -> Iterator[str]:
        if prefix:
            yield prefix
        for chunk in self.chunks:
            self.__whole_msg += chunk.content
            yield chunk.content
        if suffix:
            yield suffix

    def get_whole(self) -> str:
        return self.__whole_msg


lang = os.getenv("UI_LANG", "zh")
if lang not in ["zh", "en"]:
    lang = "zh"

st.set_page_config(
    page_title="💬 oi-wiki 智能问答助手",
    page_icon="demo/ob-icon.png",
    initial_sidebar_state="collapsed"
)
st.title("💬 oi-wiki 智能问答助手")
st.caption("🚀 使用 OceanBase 向量检索特性和大语言模型能力构建的智能问答机器人")
# st.logo("demo/logo.png")

env_table_name = os.getenv("TABLE_NAME", "OIRAG")
env_llm_base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

with st.sidebar:
    st.subheader("🔧 设置")
    st.text_input(
        "表名",
        value=env_table_name,
        disabled=True,
        help="用于存放文档及其向量数据的表名，用环境变量 TABLE_NAME 进行设置",
    )
    if env_llm_base_url == "https://open.bigmodel.cn/api/paas/v4/":
        llm_model = st.selectbox(
            "大语言模型",
            ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"],
            index=0,
            help="用于回答问题的大语言模型，用环境变量 LLM_MODEL 进行设置",
        )
    else:
        llm_model = st.text_input("大语言模型", value=os.getenv("LLM_MODEL", ""))
    history_len = st.slider(
        "聊天历史长度",
        min_value=0,
        max_value=25,
        value=3,
        help="用于上下文理解的聊天历史长度",
    )

search_docs = True
oceanbase_only = True
show_refs = True
rerank = False

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，请问有什么可以帮助您的吗？"}]

avatar_m = {
    "assistant": "🤖",
    "user": "🧑‍",
}

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=avatar_m[msg["role"]]).write(msg["content"])


def remove_refs(history: list[dict]) -> list[dict]:
    """
    Remove the references from the chat history.
    This prevents the model from generating its own reference list.
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"].split("根据向量相似性匹配检索到的相关文档如下:")[0],
        }
        for msg in history
    ]


response_placeholder = st.empty()
if prompt := st.chat_input("请输入您的问题..."):
    st.chat_message("user", avatar=avatar_m["user"]).write(prompt)

    history = st.session_state["messages"][-history_len:] if history_len > 0 else []

    it = doc_rag_stream(
        query=prompt,
        chat_history=remove_refs(history),
        llm_model=llm_model
    )

    with st.status("处理中...", expanded=True) as status:
        for msg in it:
            if not isinstance(msg, str):
                status.update(label="思考完成！")
                break
            st.write(msg)

    res = StreamResponse(it)

    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("assistant", avatar=avatar_m["assistant"]).write_stream(
        res.generate()
    )

    st.session_state.messages.append({"role": "assistant", "content": res.get_whole()})
