import re
import time

from typing import Iterator, Union
from langchain_core.messages import AIMessageChunk
from rag.embeddings import get_embedding
from rag.documents import Document, DocumentMeta, section_map
from utils.connect_oceanbse import connect_oceanbase
from agent.prompt import RAG_PROMPT, SECTION_PROMPT, INTENT_PROMPT
from agent.base_agent import Agent


def doc_search_by_vector(vector: list[float], partition_names=None, limit: int = 10,) -> list[Document]:
    oceanbase = connect_oceanbase()

    docs = oceanbase.similarity_search_by_vector(
        embedding=vector,
        k=limit,
        partition_names=partition_names,
    )
    return docs


def doc_rag_stream(
    query: str,
    chat_history: list[dict],
    llm_model: str,
    universal_rag: bool = False,
    search_docs: bool = True,
    **kwargs,
) -> Iterator[Union[str, AIMessageChunk]]:
    start_time = time.time()

    all_sections = section_map.keys()

    intent_agent = Agent(prompt=INTENT_PROMPT, llm_model=llm_model)
    rag_agent = Agent(prompt=RAG_PROMPT, llm_model=llm_model)
    section_agent = Agent(prompt=SECTION_PROMPT, llm_model=llm_model)
    embedding = get_embedding()

    query_with_history = "\n".join([msg["content"] for msg in chat_history if msg["role"] == "user"])
    query_with_history += "\n" + query

    def message_with_time(text):
        nonlocal start_time
        cur_time = time.time()
        elapsed_time = cur_time - start_time
        start_time = cur_time
        return text + "（耗时 {:.2f} 秒）".format(elapsed_time)

    if not search_docs:
        yield None
        yield from rag_agent.stream(query, chat_history, document_snippets="")
        return

    if universal_rag:
        yield message_with_time("正在使用深度学习模型将提问内容嵌入为向量...")
        query_embedded = embedding.embed_query(query)

        yield message_with_time("正在使用 OceanBase 检索相关文档...")
        docs = doc_search_by_vector(
            query_embedded,
            limit=10,
        )
    else:
        yield "正在分析问题的意图..."

        intent = intent_agent.invoke_json(query)
        intent_type = intent.get("type", "Algorithm")

        if intent_type == "Chat":
            yield message_with_time("没有算法相关内容")
            yield None
            yield from rag_agent.stream(query, chat_history, document_snippets="")
            return

        section = section_agent.invoke_json(query_with_history)
        sections: list[str] = section.get("components", ["Basic"])

        sections = list(set(sec for sec in sections if sec in all_sections))

        yield "列出相关板块" + ", ".join(sections)

        yield message_with_time("正在使用深度学习模型将提问内容嵌入为向量...")
        query_embedded = embedding.embed_query(query)

        total_docs = []
        for sec in sections:
            yield message_with_time(f"正在使用 OceanBase 检索 {sec} 的相关文档...")
            total_docs.extend(doc_search_by_vector(query_embedded, [sec]))

        docs = total_docs[:10]

    yield message_with_time("大语言模型正在思考...")

    docs_content = "\n=====\n".join(
        [f"文档片段:\n\n" + chunk.page_content for i, chunk in enumerate(docs)]
    )

    ans_itr = rag_agent.stream(query, chat_history, document_snippets=docs_content)

    visited = {}
    count = 0
    buffer: str = ""
    pruned_references = []
    get_first_token = False
    for chunk in ans_itr:
        buffer += chunk.content
        if "[" in buffer and len(buffer) < 128:
            matches = re.findall(r"(\[+\@(\d+)\]+)", buffer)
            # [('[@1]', '1'), ('[@23]', '23')]
            if matches:
                sorted(matches, key=lambda x: x[0], reverse=True)

                for m, order in matches:
                    doc = docs[int(order) - 1]
                    meta = DocumentMeta.model_validate(doc.metadata)

                    doc_name = meta.doc_name
                    doc_url = meta.doc_url.replace("doc\\docs", "https://oi-wiki.org/")\
                                          .replace("\\", "/")\
                                          .replace(".md", "")
                    idx = count + 1
                    if doc_url in visited:
                        idx = visited[doc_url]
                    else:
                        visited[doc_url] = idx
                        doc_text = f"{idx}. [{doc_name}]({doc_url})"
                        pruned_references.append(doc_text)
                        count += 1

                    ref_text = f"[[{idx}]]({doc_url})"
                    buffer = buffer.replace(m, ref_text)

        if not get_first_token:
            get_first_token = True
            yield None

        yield AIMessageChunk(content=buffer)
        buffer = ""

    if len(buffer) > 0:
        yield AIMessageChunk(content=buffer)

    ref_tip = "根据向量相似性匹配检索到的相关文档如下:"

    if len(pruned_references) > 0:
        yield AIMessageChunk(content="\n\n" + ref_tip)

        for ref in pruned_references:
            yield AIMessageChunk(content="\n" + ref)

    elif len(docs) > 0:
        yield AIMessageChunk(content="\n\n" + ref_tip)

        visited = {}
        for doc in docs:
            meta = DocumentMeta.model_validate(doc.metadata)
            doc_name = meta.doc_name
            doc_url = meta.doc_url.replace("doc\\docs", "https://oi-wiki.org/") \
                .replace("\\", "/") \
                .replace(".md", "")
            if doc_url in visited:
                continue
            visited[doc_url] = True
            count = len(visited)
            doc_text = f"{count}. [{doc_name}]({doc_url})"
            yield AIMessageChunk(content="\n" + doc_text)
