from typing import Any, Dict


class ReachyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement reachy-bridge API call.
        raise NotImplementedError("Reachy client not implemented yet")
