"""
app/providers/base.py — Abstract BaseProvider interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Concrete implementations must supply:
    - generate()       — plain-text generation
    - generate_json()  — JSON-mode generation
    - is_available()   — health check
    - model_name       — model identifier property
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a text response from *prompt*.

        Parameters
        ----------
        prompt:
            The user/input prompt.
        system:
            Optional system-level instruction prepended to the conversation.
        temperature:
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            Raw string response from the model.
        """

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system: str = "",
        schema_hint: str = "",
    ) -> dict:
        """
        Generate a JSON response and parse it into a dict.

        Parameters
        ----------
        prompt:
            The user/input prompt.
        system:
            Optional system-level instruction.
        schema_hint:
            Optional JSON schema description to help the model structure output.

        Returns
        -------
        dict
            Parsed JSON object.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether the provider endpoint is reachable and healthy.

        Returns
        -------
        bool
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g. ``"qwen2.5:7b"``)."""
