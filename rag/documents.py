#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import tqdm
from pydantic import BaseModel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import Iterator, List
from pathlib import Path


class DocumentMeta(BaseModel):
    """
    Document metadata.
    """

    class Config:
        extra = "allow"

    doc_url: str
    doc_name: str
    chunk_title: str
    enhanced_title: str


section_map = {
    "Basic": 1,
    "DP": 2,
    "DS": 3,
    "Geometry": 4,
    "Graph": 5,
    "Math": 6,
    "Misc": 7,
    "Search": 8,
    "String": 9,
    "Topic": 10,
    "Contest": 11,
    "Tools": 12,
    "Lang": 13
}


headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
    ("#####", "Header5"),
    ("######", "Header6"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)


def parse_md(file_path: str, max_chunk_size: int = 4096) -> Iterator[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    chunks = splitter.split_text(file_content)
    filename = os.path.basename(file_path)  # todo 此处可以修改成标准链接

    for chunk in chunks:
        metadata_values = list(chunk.metadata.values())
        default_title = metadata_values[-1] if metadata_values else filename

        meta = DocumentMeta(
            doc_url=file_path,
            chunk_title=default_title,
            enhanced_title=" -> ".join(metadata_values) if metadata_values else default_title,
            doc_name=chunk.metadata.get("Header1", default_title),
        )

        content = chunk.page_content
        if len(content) <= max_chunk_size:
            chunk.metadata = meta.model_dump()
            yield chunk
        else:
            for i in range(0, len(content), max_chunk_size):
                sub_content = content[i: i + max_chunk_size]
                sub_chunk = Document(sub_content, metadata=meta.model_dump())
                yield sub_chunk


class MarkdownDocumentsLoader:
    """
    Markdown Documents Loader.
    """

    def __init__(self, doc_base: str, skip_patterns: List[str] = None):
        self.doc_base = Path(doc_base)
        self.skip_patterns = [re.compile(p) for p in (skip_patterns or [])]

    def load(self, show_progress: bool = True, limit: int = 0, max_chunk_size: int = 4096) -> Iterator[Document]:
        files_to_process = [
            file_path
            for file_path in self.doc_base.rglob("*")
            if file_path.suffix in {".md", ".mdx"}
            and not any(p.search(str(file_path)) for p in self.skip_patterns)
        ]

        for count, file_path in enumerate(
            tqdm.tqdm(files_to_process, disable=not show_progress), 1
        ):
            for chunk in parse_md(str(file_path), max_chunk_size=max_chunk_size):
                yield chunk
            if 0 < limit <= count:
                print(f"Limit reached: {limit}, exiting early.")
                exit(0)
