"""
app/ai/segmenter.py

PLACEHOLDER — not yet wired into any endpoint.

This class exists for the future WebSocket-based continuous
streaming flow (FR-18 through FR-21 in the project's functional
requirements): a client would stream landmark frames continuously
over a WebSocket, and this segmenter would buffer them and detect
pauses to decide where one signed word ends and the next begins.

The current /api/recognition/classify endpoint is a single-shot REST
call (one frame in, one prediction out) — it does NOT use this class.
This file is here so the app/ai/ folder structure matches the FYP
plan, ready to be wired in once Flask-SocketIO streaming is built.
"""


class SlidingWindowSegmenter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = []

    def add_frame(self, landmarks):
        """Add one frame's landmarks to the rolling buffer."""
        self.buffer.append(landmarks)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_pause_detected(self):
        """
        DUMMY: always returns False for now. Real implementation
        would compare recent frames for near-zero movement to infer
        the signer paused between words.
        """
        return False

    def reset(self):
        self.buffer = []