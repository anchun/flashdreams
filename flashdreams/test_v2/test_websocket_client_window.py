# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the TCP-only browser client window."""

import asyncio
import json

import av
import pytest
import torch
from aiohttp import ClientSession, WSMsgType

from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.websocket_client_window import WebSocketClientWindow

pytestmark = pytest.mark.ci_cpu


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )


async def _next_message(socket, kind: WSMsgType):
    for _ in range(10):
        message = await asyncio.wait_for(socket.receive(), timeout=2)
        if message.type is kind:
            return message
    raise AssertionError(f"WebSocket did not receive {kind.name}")


@pytest.mark.asyncio
async def test_websocket_page_health_frame_and_keyboard_input() -> None:
    window = WebSocketClientWindow()
    window.open(_session_desc())
    try:
        async with ClientSession() as client:
            async with client.get(window.server.url) as response:
                page = await response.text()
                assert response.status == 200
                assert "<title>FlashDreams WebSocket</title>" in page
                assert 'data-transport="websocket"' in page
                assert '<img id="video"' in page

            async with client.ws_connect(
                f"{window.server.url}api/websocket?scene=rain"
            ) as socket:
                cursor = await _next_message(socket, WSMsgType.TEXT)
                assert json.loads(cursor.data)["type"] == "cursor_options"

                window.write(
                    StepResult(
                        step_index=0,
                        output=torch.full((1, 3, 16, 16), 37, dtype=torch.uint8),
                        frame_count=1,
                        output_layout=VideoTensorLayout.tchw,
                    )
                )
                frame_message = await _next_message(socket, WSMsgType.BINARY)
                packet = av.Packet(frame_message.data)
                decoder = av.CodecContext.create("mjpeg", "r")
                [frame] = decoder.decode(packet)
                assert frame.width == 16
                assert frame.height == 16
                assert float(frame.to_ndarray(format="rgb24").mean()) == pytest.approx(
                    37, abs=2
                )

                await socket.send_json(
                    {"type": "keyboard", "key": "w", "pressed": True}
                )
                for _ in range(100):
                    events = window.get_user_input_events()
                    keyboard = [
                        event
                        for event in events.get_events()
                        if isinstance(event, KeyboardUserInputEvent)
                    ]
                    if keyboard:
                        break
                    await asyncio.sleep(0.01)
                assert len(keyboard) == 1
                assert keyboard[0].key == "w"
                assert keyboard[0].state is KeyboardInputState.PRESSED

                query_events = window.get_user_input_events()
                assert query_events.get_events() == []

                await socket.send_str("{")
                error = await _next_message(socket, WSMsgType.TEXT)
                assert json.loads(error.data)["type"] == "error"

                async with client.get(f"{window.server.url}healthz") as response:
                    health = await response.json()
                assert health["transport"] == "websocket"
                assert health["client_connected"] is True

            for _ in range(100):
                async with client.get(f"{window.server.url}healthz") as response:
                    health = await response.json()
                if not health["client_connected"]:
                    break
                await asyncio.sleep(0.01)
            assert health["client_connected"] is False
    finally:
        window.close()


@pytest.mark.asyncio
async def test_new_websocket_replaces_the_existing_viewer() -> None:
    window = WebSocketClientWindow()
    window.open(_session_desc())
    try:
        async with ClientSession() as client:
            first = await client.ws_connect(f"{window.server.url}api/websocket")
            await _next_message(first, WSMsgType.TEXT)
            second = await client.ws_connect(f"{window.server.url}api/websocket")
            try:
                await _next_message(second, WSMsgType.TEXT)
                first_closed = await asyncio.wait_for(first.receive(), timeout=2)
                assert first_closed.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            finally:
                await first.close()
                await second.close()
    finally:
        window.close()


def test_websocket_mode_rejects_an_invalid_port() -> None:
    with pytest.raises(ValueError, match="between 0 and 65535"):
        WebSocketClientWindow(port=65536)
