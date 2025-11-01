"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    email: Optional[str] = None


@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    image_count: int = 0