from typing import Any

from singleton import singleton


@singleton
class EmbeddingCache:
    def __init__(self):
        self._cache = {}

    def get(self, model: str, key: str) -> Any:
        return self._cache.get(f"{model}_{key}", None)

    def set(self, model: str, key: str, value: Any):
        self._cache[f"{model}_{key}"] = value

    def clear(self):
        self._cache.clear()


@singleton
class MiscCache:
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Any:
        return self._cache.get(key, None)

    def set(self, key: str, value: Any):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()
