from __future__ import annotations
from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

