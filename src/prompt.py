from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


class Prompt:
    def __init__(self, messages: List[Dict[str, str]]) -> None:
        self.messages = deepcopy(messages)

    def to_chat_completion(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        rendered: List[Dict[str, str]] = []
        for message in self.messages:
            rendered.append(
                {
                    "role": message["role"],
                    "content": message["content"].format(**data),
                }
            )
        return rendered
