from __future__ import annotations

from typing import Any

import httpx
import orjson
import pytest

from langgraph_sdk.stream.sync_controller import SyncStreamController
from langgraph_sdk.stream.transport.sync_ws import SyncProtocolWebSocketTransport
from streaming._events import values_event
from streaming._sync_fake_server import SyncFakeServer


class _FakeSyncWebSocket:
    def __init__(
        self,
        frames: list[dict[str, Any]],
        *,
        fail_after: int | None = None,
    ) -> None:
        self.frames = list(frames)
        self.fail_after = fail_after
        self.sent: list[str] = []
        self.closed = False

    def __enter__(self) -> _FakeSyncWebSocket:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def send(self, data: str | bytes) -> None:
        self.sent.append(data.decode() if isinstance(data, bytes) else data)

    def __iter__(self):
        for index, frame in enumerate(self.frames, start=1):
            yield orjson.dumps(frame).decode()
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("scripted sync websocket failure")

    def close(self) -> None:
        self.closed = True


def test_sync_websocket_sends_subscribe_body_and_yields_events():
    event = values_event(seq=1, values={"counter": 1})
    socket = _FakeSyncWebSocket([event])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        handle = transport.open_event_stream(
            {"channels": ["values"], "namespaces": [[]], "since": 7}
        )
        received = list(handle.events)
        err = handle.error()
        handle.close()

    assert orjson.loads(socket.sent[0]) == {
        "id": 1,
        "method": "subscription.subscribe",
        "params": {
            "channels": ["values"],
            "namespaces": [[]],
            "since": 7,
        },
    }
    assert received == [event]
    assert err is None


def test_sync_websocket_records_post_ready_error():
    socket = _FakeSyncWebSocket([values_event(seq=1)], fail_after=1)

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        with pytest.raises(RuntimeError, match="scripted sync websocket failure"):
            list(handle.events)
        err = handle.error()
        handle.close()

    assert isinstance(err, RuntimeError)


def test_sync_websocket_send_command_uses_http_commands_endpoint():

    fake = SyncFakeServer()
    with httpx.Client(transport=fake.transport, base_url="http://test") as client:
        ws = SyncProtocolWebSocketTransport(client=client, thread_id="t-1")
        result = ws.send_command(
            {"id": 3, "method": "run.start", "params": {"input": {"x": 1}}}
        )

    assert result == {"type": "success", "id": 3, "result": {"run_id": "run-1"}}
    assert fake.received_commands[0]["method"] == "run.start"


def test_sync_websocket_open_event_stream_raises_when_closed():
    with httpx.Client(base_url="http://test") as client:
        ws = SyncProtocolWebSocketTransport(client=client, thread_id="t-1")
        ws.close()
        with pytest.raises(RuntimeError, match="closed"):
            ws.open_event_stream({"channels": ["values"]})


def test_sync_websocket_transport_feeds_sync_stream_controller():

    socket = _FakeSyncWebSocket(
        [
            values_event(seq=1, values={"counter": 1}),
            values_event(seq=2, values={"counter": 2}),
        ]
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})
        controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        first = sub.queue.get(timeout=1.0)
        second = sub.queue.get(timeout=1.0)
        end = sub.queue.get(timeout=1.0)
        controller.close()
        transport.close()

    assert first is not None
    assert second is not None
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None


def test_sync_websocket_controller_reconnects_with_since_after_drop():

    first_socket = _FakeSyncWebSocket(
        [values_event(seq=1, values={"counter": 1})],
        fail_after=1,
    )
    second_socket = _FakeSyncWebSocket([values_event(seq=2, values={"counter": 2})])
    sockets = [first_socket, second_socket]

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return sockets.pop(0)

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})
        controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        first = sub.queue.get(timeout=1.0)
        second = sub.queue.get(timeout=1.0)
        end = sub.queue.get(timeout=1.0)
        controller.close()
        transport.close()

    assert first is not None
    assert second is not None
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None
    assert orjson.loads(second_socket.sent[0])["params"]["since"] == 1


def test_sync_ws_transport_forwards_ping_kwargs():
    """ping_interval and ping_timeout are stored and forwarded to the connect callable."""
    captured_kwargs: list[dict] = []
    socket = _FakeSyncWebSocket([values_event(seq=1)])

    def connect(
        url: str,
        additional_headers: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = (url, additional_headers)
        captured_kwargs.append(kwargs)
        return socket

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
            ping_interval=15.0,
            ping_timeout=20.0,
        )
        assert transport._ping_interval == 15.0
        assert transport._ping_timeout == 20.0
        handle = transport.open_event_stream({"channels": ["values"]})
        list(handle.events)
        handle.close()

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("ping_interval") == 15.0
    assert captured_kwargs[0].get("ping_timeout") == 20.0


@pytest.mark.parametrize("cookie_header", [None, "client", "stream"])
def test_sync_ws_handshake_forwards_httpx_client_cookies(cookie_header: str | None):
    """Cookies on the httpx.Client are forwarded to the WS handshake."""
    captured_headers: list[list[tuple[str, str]]] = []
    socket = _FakeSyncWebSocket([values_event(seq=1)])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = url
        captured_headers.append(list(additional_headers or []))
        return socket

    with httpx.Client(base_url="http://test") as client:
        client.cookies.set("session", "abc123")
        if cookie_header is not None:
            client.headers["cOoKiE"] = "session=client"
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            headers={"COOKIE": "session=stream"} if cookie_header == "stream" else None,
            connect=connect,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        list(handle.events)
        handle.close()

    assert len(captured_headers) == 1
    cookies = httpx.Headers(captured_headers[0]).get_list("cookie")
    assert cookies == [f"session={cookie_header or 'abc123'}"]


@pytest.mark.parametrize("override", [False, True])
def test_sync_ws_handshake_forwards_current_client_headers(override: bool):
    captured: list[httpx.Headers] = []

    def connect(url: str, additional_headers=None, **_kw: Any):
        _ = url
        captured.append(httpx.Headers(additional_headers))
        return _FakeSyncWebSocket([])

    with httpx.Client(
        base_url="http://test",
        headers={
            "x-api-key": "initial-key",
            "Authorization": "Bearer client-token",
            "X-Tenant": "client-tenant",
        },
    ) as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            headers=(
                {"authorization": "Bearer stream-token", "X-TENANT": "stream-tenant"}
                if override
                else None
            ),
            connect=connect,
        )
        for api_key in ("first-key", "refreshed-key"):
            client.headers["X-Api-Key"] = api_key
            handle = transport.open_event_stream({"channels": ["values"]})
            list(handle.events)
            handle.close()

    assert [headers.get_list("x-api-key") for headers in captured] == [
        ["first-key"],
        ["refreshed-key"],
    ]
    for headers in captured:
        assert headers.get_list("authorization") == [
            "Bearer stream-token" if override else "Bearer client-token"
        ]
        assert headers.get_list("x-tenant") == [
            "stream-tenant" if override else "client-tenant"
        ]


def test_sync_ws_handshake_does_not_inherit_http_protocol_headers():
    captured: list[httpx.Headers] = []
    protocol_headers = {
        "Host": "http-host",
        "Connection": "keep-alive",
        "Upgrade": "h2c",
        "Content-Length": "123",
        "Content-Type": "application/json",
        "Transfer-Encoding": "chunked",
        "Sec-WebSocket-Key": "stale-key",
        "Sec-WebSocket-Version": "12",
        "Sec-WebSocket-Extensions": "stale-extension",
        "Sec-WebSocket-Protocol": "stale-protocol",
    }

    def connect(url: str, additional_headers=None, **_kw: Any):
        _ = url
        captured.append(httpx.Headers(additional_headers))
        return _FakeSyncWebSocket([])

    with httpx.Client(
        base_url="http://test", headers={**protocol_headers, "X-Api-Key": "client-key"}
    ) as client:
        transport = SyncProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            headers={"Origin": "https://app.example"},
            connect=connect,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        list(handle.events)
        handle.close()

    assert len(captured) == 1
    assert captured[0]["x-api-key"] == "client-key"
    assert captured[0]["origin"] == "https://app.example"
    assert all(header not in captured[0] for header in protocol_headers)


def test_sync_close_before_iteration_closes_socket():
    """Calling `handle.close()` before iterating events must close the socket."""
    connect_calls: list[str] = []
    socket = _FakeSyncWebSocket([values_event(seq=1)])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        connect_calls.append(url)
        return socket

    with httpx.Client(base_url="http://test") as client:
        transport = SyncProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        # Close without consuming any events.
        handle.close()

    assert socket.closed
