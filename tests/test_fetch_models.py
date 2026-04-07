"""Tests for fetch_models and _parse_model."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.models import ModelSpec

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


def test_parse_model_bare() -> None:
    """Bare OpenAI-style entry — only id is extracted."""
    entry = {"id": "whisper-1", "object": "model", "created": 123, "owned_by": "openai"}
    m = OpenAITransport._parse_model(entry)
    assert m == ModelSpec(id="whisper-1")


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
            {"id": "custom/my-model", "object": "model"},
        ],
    }
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(base_url=base_url, api_key="test", model=OPENAI_MODELS["gpt-4.1-mini"], session=session)
        await t.fetch_models()
    assert t.models["custom/my-model"] == ModelSpec(id="custom/my-model")


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
