"""
app/providers/gemini_provider.py — Stub Gemini provider (disabled by default).

Set GEMINI_ENABLED=true and GEMINI_API_KEY in your .env file to enable.
"""

from __future__ import annotations

import json
from typing import Optional

import requests

from app.providers.base import BaseProvider
from app.utils.logger import get_logger

logger = get_logger(__name__)

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = 120


class GeminiProvider(BaseProvider):
    """
    Provider that calls the Google Gemini REST API.

    Disabled by default. Enable by setting ``GEMINI_ENABLED=true`` and
    ``GEMINI_API_KEY=<your_key>`` in ``.env``.

    Parameters
    ----------
    api_key:
        Gemini API key. Never logged.
    model:
        Model identifier (e.g. ``"gemini-1.5-flash"``).
    enabled:
        Whether the provider is active. If False, all calls raise RuntimeError.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        enabled: bool = False,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._enabled = enabled

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Return True only if Gemini is enabled and an API key is configured."""
        if not self._enabled:
            return False
        if not self._api_key:
            return False
        return True

    def _require_enabled(self) -> None:
        """Raise RuntimeError if Gemini is not properly configured."""
        if not self._enabled or not self._api_key:
            raise RuntimeError(
                "Gemini provider is disabled. "
                "Set GEMINI_ENABLED=true and GEMINI_API_KEY in .env to enable."
            )

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text using the Gemini REST API.

        Raises
        ------
        RuntimeError
            If Gemini is not enabled or API key is missing.
        """
        self._require_enabled()

        url = f"{_GEMINI_API_BASE}/{self._model}:generateContent"
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        # API key passed as query parameter — never in log output
        try:
            resp = requests.post(
                url,
                params={"key": self._api_key},
                json=payload,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Gemini API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Gemini request timed out after {_TIMEOUT}s")

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response structure: {data}") from exc

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        schema_hint: str = "",
    ) -> dict:
        """
        Generate a JSON response using Gemini.

        Raises
        ------
        RuntimeError
            If Gemini is not enabled or the response is not valid JSON.
        """
        self._require_enabled()

        json_prompt = (
            f"{prompt}\n\nReturn ONLY valid JSON. No markdown fences. No commentary."
        )
        if schema_hint:
            json_prompt += f"\n\nExpected schema:\n{schema_hint}"

        raw = self.generate(json_prompt, system=system, temperature=0.1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini response is not valid JSON: {exc}\nRaw:\n{raw[:500]}"
            ) from exc

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "GeminiProvider":
        """Construct a GeminiProvider from the global AppConfig singleton."""
        from app.config import get_config  # noqa: PLC0415

        cfg = get_config()
        return cls(
            api_key=cfg.gemini_api_key,
            enabled=cfg.gemini_enabled,
        )
