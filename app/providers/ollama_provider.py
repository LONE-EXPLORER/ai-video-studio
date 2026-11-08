"""
app/providers/ollama_provider.py — Ollama REST API provider.

Uses the requests library directly (no ollama Python SDK) to call:
  POST {host}/api/generate
  GET  {host}/api/tags
"""

from __future__ import annotations

import json
from typing import List, Optional

import requests

from app.providers.base import BaseProvider
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT = 360  # seconds (allows cold-loading large 24B models into VRAM)


class OllamaProvider(BaseProvider):
    """
    Provider that communicates with a local Ollama server via its REST API.

    Parameters
    ----------
    host:
        Base URL of the Ollama server (e.g. ``"http://localhost:11434"``).
    model:
        Model identifier to use for generation (e.g. ``"qwen2.5:7b"``).
    fallback_model:
        Model to try if *model* is not available.
    """

    def __init__(
        self,
        host: str,
        model: str,
        fallback_model: Optional[str] = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._fallback_model = fallback_model
        self._active_model: Optional[str] = None  # resolved after first availability check

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the currently active model identifier."""
        return self._active_model or self._model

    def is_available(self) -> bool:
        """
        Return True if the Ollama server responds to GET /api/tags.
        Also resolves which model to use (primary or fallback).
        """
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            logger.debug("Ollama available models: %s", available)

            if self._model in available:
                self._active_model = self._model
            elif self._fallback_model and self._fallback_model in available:
                logger.warning(
                    "Primary model '%s' not found, using fallback '%s'",
                    self._model,
                    self._fallback_model,
                )
                self._active_model = self._fallback_model
            elif available:
                self._active_model = available[0]
                logger.warning(
                    "Neither '%s' nor fallback found; using first available model '%s'",
                    self._model,
                    self._active_model,
                )
            else:
                logger.warning("Ollama is running but no models are installed.")
                self._active_model = self._model  # will fail on generate, surfaced then

            return True
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama server not reachable at %s", self._host)
            return False
        except requests.exceptions.Timeout:
            logger.warning("Ollama health check timed out")
            return False
        except Exception as exc:
            logger.warning("Ollama availability check failed: %s", exc)
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Send a generation request to Ollama and return the response text.

        Parameters
        ----------
        prompt:
            User prompt.
        system:
            System prompt prepended to the request.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            Generated text.

        Raises
        ------
        RuntimeError
            On connection failure, non-200 HTTP status, or timeout.
        """
        model = self._active_model or self._model
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        }
        logger.debug("Ollama generate request: model=%s prompt_len=%d", model, len(prompt))

        try:
            resp = requests.post(
                f"{self._host}/api/generate",
                json=payload,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Ollama server not reachable at {self._host}: {exc}") from exc
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama generate request timed out after {_TIMEOUT}s"
            )
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc

        data = resp.json()
        response_text = data.get("response", "")
        logger.debug("Ollama response length: %d chars", len(response_text))
        return response_text

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        schema_hint: str = "",
    ) -> dict:
        """
        Generate a JSON response from Ollama using format: "json" mode.

        Retries once if the first response cannot be parsed as valid JSON.

        Parameters
        ----------
        prompt:
            User prompt.
        system:
            System prompt.
        schema_hint:
            Optional schema hint appended to the system prompt.

        Returns
        -------
        dict
            Parsed JSON object.

        Raises
        ------
        RuntimeError
            If JSON cannot be parsed after one retry.
        """
        model = self._active_model or self._model
        effective_system = system
        if schema_hint:
            effective_system = f"{system}\n\nExpected JSON schema:\n{schema_hint}".strip()

        payload = {
            "model": model,
            "prompt": prompt,
            "system": effective_system,
            "stream": False,
            "format": "json",
            "keep_alive": "5m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        }
        logger.debug("Ollama generate_json request: model=%s", model)

        def _attempt(payload: dict) -> dict:
            try:
                resp = requests.post(
                    f"{self._host}/api/generate",
                    json=payload,
                    timeout=_TIMEOUT,
                )
                resp.raise_for_status()
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(f"Ollama not reachable: {exc}") from exc
            except requests.exceptions.Timeout:
                raise RuntimeError(f"Ollama timed out after {_TIMEOUT}s")
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(f"Ollama HTTP error: {exc.response.text}") from exc

            raw = resp.json().get("response", "")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON from model: {exc}\nRaw:\n{raw[:500]}") from exc

        try:
            return _attempt(payload)
        except ValueError as first_error:
            logger.warning("JSON parse failed on first attempt, retrying: %s", first_error)
            retry_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Your previous response failed JSON validation with error: "
                f"{first_error}\n"
                f"Return ONLY valid JSON — no markdown fences, no commentary."
            )
            payload["prompt"] = retry_prompt
            try:
                return _attempt(payload)
            except ValueError as second_error:
                raise RuntimeError(
                    f"Ollama failed to produce valid JSON after retry: {second_error}"
                ) from second_error

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def get_available_models(self) -> List[str]:
        """
        Return a list of model names currently installed in Ollama.

        Returns
        -------
        list[str]
        """
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("Could not fetch Ollama model list: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "OllamaProvider":
        """Construct an OllamaProvider from the global AppConfig singleton."""
        from app.config import get_config  # noqa: PLC0415

        cfg = get_config()
        return cls(
            host=cfg.ollama_host,
            model=cfg.ollama_model,
            fallback_model=cfg.ollama_fallback_model,
        )
