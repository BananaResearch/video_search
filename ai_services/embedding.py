from typing import Union, List, Optional

from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, OpenAIError
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from ai_services.cache import EmbeddingCache


class OpenAIEmbeddingService:
    def __init__(
            self,
            model: str,
            dimensions: Optional[int] = None,
    ):
        self._client = OpenAI()
        self._model_name = model
        self._dimensions = dimensions

    def embed(self, inputs: Union[str, List[str]]) -> list:
        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(1),
            retry=retry_if_exception_type((
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )),
        )
        for attempt in retrying:
            with attempt:
                return self._embed(inputs)

    def _embed(self, inputs: Union[str, List[str]]) -> list:
        if isinstance(inputs, str):
            inputs = [inputs]
        if inputs in [None, []]:
            return []

        cached = [None for _ in range(len(inputs))]
        uncached = []
        for i, query in enumerate(inputs):
            vec = EmbeddingCache().get(self._model_name, query)
            if vec is not None:
                cached[i] = vec
            else:
                uncached.append(query)
        if len(uncached) == 0:
            return cached

        model_kwargs = {
            "dimensions": self._dimensions,
        } if self._dimensions else {}

        response = self._client.embeddings.create(
            input=uncached,
            model=self._model_name,
            **model_kwargs
        )
        vecs = [item.embedding for item in response.data]
        for i, query in enumerate(uncached):
            EmbeddingCache().set(self._model_name, query, vecs[i])

        for i, vec in enumerate(cached):
            if vec is None:
                cached[i] = vecs.pop(0)

        return cached


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    service = OpenAIEmbeddingService(
        model="text-embedding-3-large",
        dimensions=8,
    )

    inputs = ["Hello, world!", "This is a test."]
    embeddings = service.embed(inputs)
    print(embeddings)
    print("----")
    inputs = ["Hello, world!", "This is a test. dfdd"]
    embeddings = service.embed(inputs)
    print(embeddings)
    print("----")
    inputs = ["This is a test. dfdd"]
    embeddings = service.embed(inputs)
    print(embeddings)
