from typing import List, Union

from PIL.Image import Image

from ai_services.cache import MiscCache
from main.image_processor import ImageDescriber
import os

from vdb.vector_store import VectorStore

MAX_VIDEO_COUNT = int(os.environ.get("MAX_SEARCH_RESULTS", 10))

COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
vector_store = VectorStore(COLLECTION_NAME)

cache = MiscCache()


def filter_results(results, threshold) -> List[dict]:
    return [result for result in results if result['score'] >= threshold]


def get_keywords_from_image(image: Union[str, Image]) -> List[str]:
    return ImageDescriber().invoke(image)


def search_videos_by_keywords(keywords: List[str], threshold: float) -> List[dict]:
    if not keywords:
        return []
    query = ", ".join(keywords)
    if (results := cache.get(query)) is not None:
        pass
    else:
        results = vector_store.search(query, top_k=MAX_VIDEO_COUNT)
        cache.set(query, results)

    return filter_results(results, threshold)
