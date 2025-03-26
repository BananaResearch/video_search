import re
from pathlib import Path
from typing import Union, List


class PromptTemplate:
    def __init__(self, template: str):
        self._template = template
        self._partial_variables = {}
        self._input_variables = self._get_input_variables()

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'PromptTemplate':
        try:
            with open(filename, "r", encoding="utf-8") as f:
                template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found.")
        return cls(template)

    def _get_input_variables(self) -> List[str]:
        pattern = r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})'
        triple_brace_pattern = r'\{\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}\}'
        slots = re.findall(pattern, self._template)
        triple_brace_slots = re.findall(triple_brace_pattern, self._template)
        return list(set(slots + triple_brace_slots))

    def invoke(self, **kwargs) -> str:
        if len(self._input_variables) == 0:
            return self._template
        inputs = {}
        for v in self._input_variables:
            if v in kwargs:
                inputs[v] = kwargs[v]
            elif v in self._partial_variables:
                inputs[v] = self._partial_variables[v]
            else:
                raise ValueError(f"Missing input variable {v}")
        return self._template.format(**inputs)

    def partial(self, **kwargs) -> 'PromptTemplate':
        for v in self._input_variables:
            if v in kwargs:
                self._partial_variables[v] = kwargs[v]
        return self

    def input_variables(self) -> List[str]:
        return self._input_variables

    def get_template(self) -> str:
        return self._template

    def get_partial_variables(self) -> dict:
        return self._partial_variables

