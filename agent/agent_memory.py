"""Session-scoped context for the Revenue Agent."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict


class AgentMemory:
    """Keeps the active analysis and conversational context per demo session."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start(self, session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        record = self._sessions.setdefault(session_id, {})
        record.update({
            "state": state,
            "focus": {},
            "active_policy": None,
            "deployed_policy": None,
            "last_simulation": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return record

    def get(self, session_id: str) -> Dict[str, Any]:
        return self._sessions.setdefault(session_id, {"focus": {}})

    def set_focus(self, session_id: str, dimension: str, value: str) -> None:
        record = self.get(session_id)
        record["focus"] = {"dimension": dimension, "value": value}
        record["updated_at"] = datetime.now(timezone.utc).isoformat()

    def save_simulation(self, session_id: str, simulation: Dict[str, Any]) -> None:
        record = self.get(session_id)
        record["last_simulation"] = simulation
        record["updated_at"] = datetime.now(timezone.utc).isoformat()

    def deploy(self, session_id: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        record = self.get(session_id)
        deployed = deepcopy(policy)
        deployed["deployed_at"] = datetime.now(timezone.utc).isoformat()
        record["deployed_policy"] = deployed
        record["active_policy"] = deployed
        record["updated_at"] = deployed["deployed_at"]
        return deployed
