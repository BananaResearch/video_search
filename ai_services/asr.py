from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from singleton import singleton


@singleton
class OpenAIASRService:
    def __init__(self, model_name: str):
        self._client = OpenAI()
        self._model_name = model_name

    def transcribe(self, audio_file: str, **kwargs) -> dict:
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
                return self._transcribe(audio_file, **kwargs)

    def _transcribe(self, audio_file: str, **kwargs) -> dict:
        with open(audio_file, 'rb') as audio_file:
            transcription = self._client.audio.transcriptions.create(
                model=self._model_name,
                file=audio_file,
                response_format="verbose_json",
                **kwargs
            )
        return transcription.dict()
