import hashlib
import os
import shutil
import uuid
from typing import List, Dict, Any, Callable
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ai_services.embedding import OpenAIEmbeddingService


class VectorStore:
    def __init__(
            self,
            collection_name: str,
    ):
        working_dir = os.environ.get("WORKING_DIR")
        self._collection_name = collection_name
        self._path = os.path.join(working_dir, "qdrant_data")
        self._client = QdrantClient(path=self._path)
        embedding_model = os.environ.get("EMBEDDING_MODEL")
        if not embedding_model:
            raise ValueError("请提供嵌入模型")
        if os.environ.get("EMBEDDING_DIMENSION"):
            self._vector_size = int(os.environ.get("EMBEDDING_DIMENSION"))
        self._embedding_function = OpenAIEmbeddingService(
            model=embedding_model,
            dimensions=self._vector_size
        ).embed

        # 确保集合存在
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self._client.get_collections().collections
        if not any(collection.name == self._collection_name for collection in collections):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size, distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # 即时索引
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
                wal_config=models.WalConfigDiff(
                    wal_capacity_mb=32,
                ),
                shard_number=1,
                on_disk_payload=True,
            )

    def set_embedding_function(self, embedding_function: Callable):
        self._embedding_function = embedding_function

    @staticmethod
    def _generate_unique_id(input_string: str) -> str:
        # Create a hash of the input string
        hash_object = hashlib.md5(input_string.encode())
        hash_hex = hash_object.hexdigest()

        # Create a UUID using the hash
        namespace = uuid.NAMESPACE_DNS
        generated_uuid = uuid.uuid5(namespace, hash_hex)

        return str(generated_uuid)

    def add_documents(self, documents: List[Dict[str, Any]]):
        try:
            self._add_documents(documents)
        except Exception as e:
            print(f"清空当前库并重新添加文档: {e}")
            # force delete self._path
            shutil.rmtree(self._path)
            self._client = QdrantClient(path=self._path)
            self._create_collection_if_not_exists()
            self._add_documents(documents)

    def _add_documents(self, documents: List[Dict[str, Any]]):
        if not self._embedding_function:
            raise ValueError("请先设置嵌入函数")
        points = []
        texts = [doc['text'] for doc in documents]
        vectors = self._embedding_function(texts)
        for i, doc in enumerate(documents):
            if "checksum" in doc['metadata']:
                _uuid = self._generate_unique_id(doc['metadata']['checksum'])
            else:
                _uuid = self._generate_unique_id(doc['text'])
            point = models.PointStruct(
                id=_uuid,
                vector=vectors[i],
                payload={
                    'text': doc['text'],
                    **doc['metadata']
                }
            )
            points.append(point)

        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query: str, top_k: int = 5):
        if not self._embedding_function:
            raise ValueError("请先设置嵌入函数")

        query_vector = self._embedding_function(query)
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector[0],
            limit=top_k
        )

        return [
            {
                'text': hit.payload['text'],
                'metadata': {
                    k: v for k, v in hit.payload.items() if k != 'text'
                },
                'score': hit.score
            } for hit in results
        ]

    def persist(self):
        # Qdrant已经自动持久化到磁盘,所以这里不需要额外操作
        pass


# 使用示例
if __name__ == "__main__":
    from ai_services.embedding import OpenAIEmbeddingService
    from data_utils.file import generate_checksum
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    # 初始化 VectorStore
    vector_store = VectorStore("my_collection")

    service = OpenAIEmbeddingService(
        model="text-embedding-3-large",
        dimensions=256,
    )
    vector_store.set_embedding_function(service.embed)

    # 添加文档
    documents = [
        {
            'text': '这是第一个文档',
            'metadata': {
                'source_file': 'C:\\Users\\Admin\\Desktop\\Demos\\data\\videos\\一元一次方程.mov',
                'title': '文档1',
                'display_text': '文档1的摘要',
                'checksum': generate_checksum('C:\\Users\\Admin\\Desktop\\Demos\\data\\videos\\一元一次方程.mov')
            }
        },
        {
            'text': '这是第二个文档',
            'metadata': {
                'source_file': 'C:\\Users\\Admin\\Desktop\\Demos\\data\\videos\\一元一次方程 与工程有关的一元一次方程.mov',
                'title': '文档2',
                'display_text': '文档2的摘要',
                'checksum': generate_checksum(
                    'C:\\Users\\Admin\\Desktop\\Demos\\data\\videos\\一元一次方程 与工程有关的一元一次方程.mov')
            }
        }
    ]
    vector_store.add_documents(documents)

    # 搜索
    results = vector_store.search("第一个文档", top_k=2)
    print(results)

    # 持久化 (自动完成)
    vector_store.persist()
