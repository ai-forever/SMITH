"""Prompt policy for the backend-neutral retrieval workflow."""

from __future__ import annotations

from ..config import HarnessConfig
from harness.common import Message
from ..state import SessionState
from .system_prompt import render_system_prompt


class PromptPolicy:
    """Seed the system/user turns once; the loop appends turns to ``state.messages``."""

    def render(self, state: SessionState, config: HarnessConfig) -> list[Message]:
        if not state.messages:
            state.messages.append(Message(role="system", content=render_system_prompt(config)))
            state.messages.append(Message(role="user", content=state.query))
        return list(state.messages)
