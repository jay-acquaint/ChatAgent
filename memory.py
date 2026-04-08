from typing import Dict, List


# session_id → messages
memory_store: Dict[str, List[str]] = {}


def get_history(session_id: str) -> List[str]:
    return memory_store.get(session_id, [])


def save_message(session_id: str, message: str):
    if session_id not in memory_store:
        memory_store[session_id] = []

    memory_store[session_id].append(message)
    memory_store[session_id] = memory_store[session_id][-10:]