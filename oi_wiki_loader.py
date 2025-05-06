#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import uuid
import dotenv

from langchain_core.documents import Document
from rag.embeddings import get_embedding
from rag.documents import MarkdownDocumentsLoader, section_map
from utils.connect_oceanbse import connect_oceanbase

dotenv.load_dotenv()

embeddings = get_embedding(
    ollama_url=os.getenv("OLLAMA_URL") or None,
    ollama_token=os.getenv("OLLAMA_TOKEN") or None,
    base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL") or None,
    api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or None,
    model=os.getenv("OPENAI_EMBEDDING_MODEL") or None,
)

ob = connect_oceanbase()


def optimize_ob_args():
    vals = []
    params = ob.obvector.perform_raw_text_sql(
        "SHOW PARAMETERS LIKE '%ob_vector_memory_limit_percentage%'"
    )
    for row in params:
        val = int(row[6])
        vals.append(val)
    if len(vals) == 0:
        print("ob_vector_memory_limit_percentage not found in parameters.")
        exit(1)
    if any(val == 0 for val in vals):
        try:
            ob.obvector.perform_raw_text_sql(
                "ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30"
            )
        except Exception as e:
            print("Failed to set ob_vector_memory_limit_percentage to 30.")
            print("Error message:", e)
            exit(1)
    ob.obvector.perform_raw_text_sql("SET ob_query_timeout=100000000")


def insert_batch(docs: list[Document], section):
    code = section_map[section]
    if not code:
        raise ValueError(f"section {section} not found in section_map.")

    ob.add_documents(
        docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
        extras=[{"section_code": code} for _ in docs],
        partition_name=section,
    )


def insert_oi_wiki(file_dir: str, partition_name):
    batch_size = 4
    limit = 300

    batch = []
    loader = MarkdownDocumentsLoader(file_dir)
    for doc in loader.load(limit=limit):
        if len(batch) == batch_size:
            insert_batch(batch, partition_name)
            batch = []
        batch.append(doc)

    if len(batch) > 0:
        insert_batch(batch, partition_name)


if __name__ == '__main__':
    optimize_ob_args()
    # insert_oi_wiki("doc\\docs\\basic", "Basic")
    # insert_oi_wiki("doc\\docs\\dp", "DP")
    # insert_oi_wiki("doc\\docs\\ds", "DS")
    # insert_oi_wiki("doc\\docs\\geometry", "Geometry")
    # insert_oi_wiki("doc\\docs\\graph", "Graph")
    # insert_oi_wiki("doc\\docs\\math", "Math")
    # insert_oi_wiki("doc\\docs\\misc", "Misc")
    # insert_oi_wiki("doc\\docs\\search", "Search")
    # insert_oi_wiki("doc\\docs\\string", "String")
    # insert_oi_wiki("doc\\docs\\topic", "Topic")
    # insert_oi_wiki("doc\\docs\\contest", "Contest")
    # insert_oi_wiki("doc\\docs\\lang", "Lang")
    # insert_oi_wiki("doc\\docs\\tools", "Tools")
    ...
