# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TCP-only browser client window for the v2 runtime."""

from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


class WebSocketClientWindow(WebRTCClientWindow):
    """Stream JPEG frames and browser controls over one WebSocket."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        """Create a TCP-only browser backend."""
        super().__init__(
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
            _transport="websocket",
        )
