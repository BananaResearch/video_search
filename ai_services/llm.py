from typing import Union, Iterator, List

import PIL
from PIL import Image
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from data_utils.image import image_file_to_base64, image_data_to_base64


class OpenAILLMService:
    def __init__(self, model: str):
        self._client = OpenAI()
        self._model_name = model

    def invoke(
            self,
            inputs: Union[str, list],
            images: list = None,
            **kwargs
    ) -> str:
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
                return self._invoke(inputs, images, **kwargs)

    def stream(
            self,
            inputs: Union[str, list],
            images: list = None,
            **kwargs
    ) -> Iterator[str]:
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
                return self._stream(inputs, images, **kwargs)

    def _stream(
            self,
            inputs: Union[str, list],
            images: list = None,
            **kwargs
    ) -> Iterator[str]:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._format_messages(inputs, images),
            stream=True,
            timeout=10,
            stream_options={"include_usage": True},
            **kwargs,
        )

        cached = ""
        for chunk in completion:
            if len(chunk.choices) == 0:
                if chunk.usage is not None:
                    pass
                break
            elif chunk.choices[0].finish_reason is not None:
                pass
            else:
                cached += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content

    def _invoke(
            self,
            inputs: Union[str, list],
            images: list = None,
            **kwargs
    ) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._format_messages(inputs, images),
            **kwargs,
        )
        return completion.choices[0].message.content

    @staticmethod
    def _format_messages(
            inputs: Union[str, list],
            images: list = None,
    ) -> List[dict]:
        if isinstance(inputs, str):
            if not images:
                return [{"role": "user", "content": inputs}]
            content = [{"type": "text", "text": inputs}]
            for i, img in enumerate(images):
                if isinstance(img, str):
                    image_url = image_file_to_base64(img)
                elif isinstance(img, Image.Image):
                    image_url = image_data_to_base64(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                )
            return [{"role": "user", "content": content}]
        else:
            if not images:
                return inputs
            content = []
            for i, img in enumerate(images):
                if isinstance(img, str):
                    image_url = image_file_to_base64(img)
                elif isinstance(img, Image.Image):
                    image_url = image_data_to_base64(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                )
            return [*inputs, {"role": "user", "content": content}]
