"""Tests for fetch_models and _parse_model (OpenAI, OpenRouter, etc.)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.models import Capability, ModelSpec

from axio_transport_openai import OPENAI_MODELS, OpenAITransport


# ---------------------------------------------------------------------------
# Fake /v1/models server
# ---------------------------------------------------------------------------


class FakeModelsServer:
    def __init__(self) -> None:
        self.models_response: dict[str, Any] = {"object": "list", "data": []}
        self.status_code: int = 200

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", self._handle)
        return app

    async def _handle(self, request: web.Request) -> web.Response:
        if self.status_code != 200:
            return web.Response(status=self.status_code, text="error")
        return web.json_response(self.models_response)


@pytest.fixture
async def models_server() -> AsyncIterator[tuple[FakeModelsServer, str]]:
    server = FakeModelsServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}/v1"
    yield server, base_url
    await runner.cleanup()


# ---------------------------------------------------------------------------
# _parse_model unit tests
# ---------------------------------------------------------------------------


def test_parse_model_minimal() -> None:
    """Bare OpenAI-style entry with only id."""
    entry = {"id": "whisper-1", "object": "model", "created": 123, "owned_by": "openai"}
    m = OpenAITransport._parse_model(entry)
    assert m.id == "whisper-1"
    assert m.capabilities == frozenset({Capability.text})
    assert m.context_window == 128_000
    assert m.max_output_tokens == 8192
    assert m.input_cost == 0.0
    assert m.output_cost == 0.0


def test_parse_model_openrouter_rich() -> None:
    """OpenRouter-style entry with full metadata."""
    entry = {
        "id": "openai/gpt-4.1",
        "context_length": 1_047_576,
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": "0.000002", "completion": "0.000008"},
        "top_provider": {"context_length": 1_047_576, "max_completion_tokens": 32768},
        "supported_parameters": [
            "max_tokens",
            "temperature",
            "tools",
            "tool_choice",
            "response_format",
            "structured_outputs",
        ],
    }
    m = OpenAITransport._parse_model(entry)
    assert m.id == "openai/gpt-4.1"
    assert m.context_window == 1_047_576
    assert m.max_output_tokens == 32768
    assert Capability.text in m.capabilities
    assert Capability.vision in m.capabilities
    assert Capability.tool_use in m.capabilities
    assert Capability.json_mode in m.capabilities
    assert Capability.structured_outputs in m.capabilities
    assert Capability.reasoning not in m.capabilities
    assert m.input_cost == pytest.approx(2.0)
    assert m.output_cost == pytest.approx(8.0)


def test_parse_model_openrouter_free_router() -> None:
    """openrouter/free: null top_provider limits, zero pricing."""
    entry = {
        "id": "openrouter/free",
        "context_length": 200_000,
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": "0", "completion": "0"},
        "top_provider": {"context_length": None, "max_completion_tokens": None},
        "supported_parameters": [
            "tools",
            "tool_choice",
            "reasoning",
            "include_reasoning",
            "response_format",
            "structured_outputs",
        ],
    }
    m = OpenAITransport._parse_model(entry)
    assert m.id == "openrouter/free"
    assert m.context_window == 200_000
    assert m.max_output_tokens == 8192
    assert Capability.vision in m.capabilities
    assert Capability.tool_use in m.capabilities
    assert Capability.reasoning in m.capabilities
    assert m.input_cost == 0.0
    assert m.output_cost == 0.0


def test_parse_model_negative_pricing() -> None:
    """openrouter/auto: negative pricing (dynamic) clamped to 0."""
    entry = {
        "id": "openrouter/auto",
        "context_length": 2_000_000,
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": "-1", "completion": "-1"},
        "top_provider": {"context_length": None, "max_completion_tokens": None},
        "supported_parameters": ["tools", "reasoning"],
    }
    m = OpenAITransport._parse_model(entry)
    assert m.input_cost == 0.0
    assert m.output_cost == 0.0
    assert m.context_window == 2_000_000


def test_parse_model_embedding() -> None:
    """Model with embedding output modality."""
    entry = {
        "id": "text-embedding-custom",
        "context_length": 8192,
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["embedding"],
        },
        "pricing": {"prompt": "0.00000002", "completion": "0"},
        "top_provider": {"context_length": 8192, "max_completion_tokens": 0},
        "supported_parameters": [],
    }
    m = OpenAITransport._parse_model(entry)
    assert Capability.embedding in m.capabilities
    assert Capability.vision not in m.capabilities


# ---------------------------------------------------------------------------
# fetch_models integration tests
# ---------------------------------------------------------------------------


async def test_fetch_models_known_model_uses_builtin(
    models_server: tuple[FakeModelsServer, str],
) -> None:
    """Known model IDs should use the built-in OPENAI_MODELS spec."""
    server, base_url = models_server
    server.models_response = {
        "data": [
            {"id": "gpt-4.1-mini", "object": "model", "created": 1, "owned_by": "openai"},
        ],
    }
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(base_url=base_url, api_key="test", model=OPENAI_MODELS["gpt-4.1-mini"], session=session)
        await t.fetch_models()
    assert "gpt-4.1-mini" in t.models
    assert t.models["gpt-4.1-mini"] is OPENAI_MODELS["gpt-4.1-mini"]


async def test_fetch_models_unknown_model_parsed(
    models_server: tuple[FakeModelsServer, str],
) -> None:
    """Unknown model IDs should be parsed via _parse_model."""
    server, base_url = models_server
    server.models_response = {
        "data": [
            {
                "id": "custom/my-model",
                "context_length": 65536,
                "top_provider": {"max_completion_tokens": 4096},
                "supported_parameters": ["tools"],
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "pricing": {"prompt": "0.000001", "completion": "0.000003"},
            },
        ],
    }
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(base_url=base_url, api_key="test", model=OPENAI_MODELS["gpt-4.1-mini"], session=session)
        await t.fetch_models()
    m = t.models["custom/my-model"]
    assert m.context_window == 65536
    assert m.max_output_tokens == 4096
    assert Capability.tool_use in m.capabilities
    assert m.input_cost == pytest.approx(1.0)
    assert m.output_cost == pytest.approx(3.0)


async def test_fetch_models_error_falls_back(
    models_server: tuple[FakeModelsServer, str],
) -> None:
    """Non-200 response should fall back to OPENAI_MODELS."""
    server, base_url = models_server
    server.status_code = 500
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(base_url=base_url, api_key="test", model=OPENAI_MODELS["gpt-4.1-mini"], session=session)
        await t.fetch_models()
    assert t.models is OPENAI_MODELS


async def test_fetch_models_skips_empty_ids(
    models_server: tuple[FakeModelsServer, str],
) -> None:
    """Entries without id should be skipped."""
    server, base_url = models_server
    server.models_response = {
        "data": [
            {"id": "", "object": "model"},
            {"object": "model"},
            {"id": "valid-model", "object": "model"},
        ],
    }
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(base_url=base_url, api_key="test", model=OPENAI_MODELS["gpt-4.1-mini"], session=session)
        await t.fetch_models()
    assert len(t.models) == 1
    assert "valid-model" in t.models
