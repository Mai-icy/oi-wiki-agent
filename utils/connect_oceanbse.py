import os
import dotenv
from rag.embeddings import get_embedding
from sqlalchemy import Column, Integer
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from rag.documents import MarkdownDocumentsLoader, section_map as cm
from pyobvector import ObListPartition, RangeListPartInfo

dotenv.load_dotenv()

connection_args = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD").replace("@", "%40") if os.getenv("DB_PASSWORD") else "",
    "db_name": os.getenv("DB_NAME"),
}

instance = None


def connect_oceanbase() -> OceanbaseVectorStore:
    global instance
    if instance is None:
        instance = OceanbaseVectorStore(
            embedding_function=get_embedding(),
            table_name=os.getenv("TABLE_NAME", "corpus"),
            connection_args=connection_args,
            metadata_field="metadata",
            extra_columns=[Column("section_code", Integer, primary_key=True)],
            partitions=ObListPartition(
                is_list_columns=False,
                list_part_infos=[RangeListPartInfo(k, v) for k, v in cm.items()]
                                + [RangeListPartInfo("p10", "DEFAULT")],
                list_expr="section_code",
            )
        )
    return instance


