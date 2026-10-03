from __future__ import annotations

from collections import UserList
from collections.abc import Sequence

import httpx
import pytest

from langgraph_sdk.client import HttpClient, RunsClient, SyncHttpClient, SyncRunsClient
from langgraph_sdk.schema import StreamMode, StreamPart

STREAM_MODE_CASES = [
    pytest.param(["values", "updates"], '["values","updates"]', id="list"),
    pytest.param(("values", "updates"), '["values","updates"]', id="tuple"),
    pytest.param(
        UserList(["values", "updates"]), '["values","updates"]', id="sequence"
    ),
    pytest.param(["values"], '["values"]', id="singleton-list"),
    pytest.param(("values",), '["values"]', id="singleton-tuple"),
    pytest.param([], "[]", id="empty-list"),
    pytest.param((), "[]", id="empty-tuple"),
    pytest.param("values", "values", id="string"),
    pytest.param(None, "", id="none"),
]

STREAM_PARTS = [
    StreamPart(event="values", data={"step": 1}, id="event-2"),
    StreamPart(event="updates", data={"node": {"step": 2}}, id="event-3"),
]


@pytest.fixture
def mock_transport() -> tuple[list[httpx.Request], httpx.MockTransport]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                b'id: event-2\nevent: values\ndata: {"step": 1}\n\n'
                b'id: event-3\nevent: updates\ndata: {"node": {"step": 2}}\n\n'
            ),
        )

    return requests, httpx.MockTransport(handler)


def _assert_request(
    requests: list[httpx.Request],
    expected_stream_mode: str,
    *,
    cancel_on_disconnect: str = "true",
    last_event_id: str = "method-cursor",
) -> None:
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert request.url.path == "/threads/thread-id/runs/run-id/stream"
    assert request.content == b""
    # Inspect the actual URL after HTTPX encodes the SDK's parameters.
    assert request.url.params.get_list("stream_mode") == [expected_stream_mode]
    assert dict(request.url.params) == {
        "stream_mode": expected_stream_mode,
        "cancel_on_disconnect": cancel_on_disconnect,
        "channel": "a b",
        "count": "2",
    }
    assert request.headers["Last-Event-ID"] == last_event_id
    assert request.headers["X-Custom"] == "custom"
    assert request.headers["Accept"] == "text/event-stream"
    assert request.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(("stream_mode", "expected_stream_mode"), STREAM_MODE_CASES)
def test_sync_join_stream_preserves_stream_modes(
    mock_transport: tuple[list[httpx.Request], httpx.MockTransport],
    stream_mode: StreamMode | Sequence[StreamMode] | None,
    expected_stream_mode: str,
) -> None:
    requests, transport = mock_transport
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(SyncHttpClient(client))
        parts = list(
            runs.join_stream(
                "thread-id",
                "run-id",
                stream_mode=stream_mode,
                cancel_on_disconnect=True,
                params={"channel": "a b", "count": 2},
                last_event_id="method-cursor",
                headers={"X-Custom": "custom"},
            )
        )

    _assert_request(requests, expected_stream_mode)
    assert parts == STREAM_PARTS


@pytest.mark.parametrize(("stream_mode", "expected_stream_mode"), STREAM_MODE_CASES)
async def test_join_stream_preserves_stream_modes(
    mock_transport: tuple[list[httpx.Request], httpx.MockTransport],
    stream_mode: StreamMode | Sequence[StreamMode] | None,
    expected_stream_mode: str,
) -> None:
    requests, transport = mock_transport
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs = RunsClient(HttpClient(client))
        parts = [
            part
            async for part in runs.join_stream(
                "thread-id",
                "run-id",
                stream_mode=stream_mode,
                cancel_on_disconnect=True,
                params={"channel": "a b", "count": 2},
                last_event_id="method-cursor",
                headers={"X-Custom": "custom"},
            )
        ]

    _assert_request(requests, expected_stream_mode)
    assert parts == STREAM_PARTS


def test_sync_join_stream_preserves_custom_overrides(
    mock_transport: tuple[list[httpx.Request], httpx.MockTransport],
) -> None:
    requests, transport = mock_transport
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(SyncHttpClient(client))
        parts = list(
            runs.join_stream(
                "thread-id",
                "run-id",
                stream_mode=["values", "updates"],
                cancel_on_disconnect=True,
                params={
                    "stream_mode": '["custom","messages"]',
                    "cancel_on_disconnect": False,
                    "channel": "a b",
                    "count": 2,
                },
                last_event_id="method-cursor",
                headers={"Last-Event-ID": "header-cursor", "X-Custom": "custom"},
            )
        )

    _assert_request(
        requests,
        '["custom","messages"]',
        cancel_on_disconnect="false",
        last_event_id="header-cursor",
    )
    assert parts == STREAM_PARTS


async def test_join_stream_preserves_custom_overrides(
    mock_transport: tuple[list[httpx.Request], httpx.MockTransport],
) -> None:
    requests, transport = mock_transport
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs = RunsClient(HttpClient(client))
        parts = [
            part
            async for part in runs.join_stream(
                "thread-id",
                "run-id",
                stream_mode=["values", "updates"],
                cancel_on_disconnect=True,
                params={
                    "stream_mode": '["custom","messages"]',
                    "cancel_on_disconnect": False,
                    "channel": "a b",
                    "count": 2,
                },
                last_event_id="method-cursor",
                headers={"Last-Event-ID": "header-cursor", "X-Custom": "custom"},
            )
        ]

    _assert_request(
        requests,
        '["custom","messages"]',
        cancel_on_disconnect="false",
        last_event_id="header-cursor",
    )
    assert parts == STREAM_PARTS
