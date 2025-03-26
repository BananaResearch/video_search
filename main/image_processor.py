import json
import os
import re
from typing import Any, Union

from PIL.Image import Image

from ai_services.llm import OpenAILLMService
from prompt_template import PromptTemplate
from singleton import singleton


def parse_json_from_text(text) -> dict | list:
    # Regular expression to match code blocks with optional \n characters
    pattern = r'```(?:json)?\s*(.*?)\s*```'

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        obj_str = matches[-1].strip()
    else:
        obj_str = text.strip()

    return json.loads(obj_str)


@singleton
class ImageDescriber:

    def __init__(self):
        self._prompt_template = PromptTemplate.from_file("prompts/image_to_keyword.txt")
        llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
        self._llm = OpenAILLMService(llm_model)

    def invoke(self, image: Union[str, Image]) -> Any:
        prompt = self._prompt_template.invoke()
        response = self._llm.invoke(
            prompt,
            images=[image],
            temperature=0,
            seed=int(os.environ.get("SEED", 42))
        )
        # print(response)
        try:
            keywords = parse_json_from_text(response)
        except json.JSONDecodeError:
            keywords = []
        return keywords


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    image_path = "./data/images/383d75a3f2d5cd930a436aa20127d6c.jpg"
    describer = ImageDescriber()
    keywords = describer.invoke(image_path)
    print(keywords)
