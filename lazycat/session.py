from typing import Any

class ConversationSession:
    """Manages conversation history and basic session state for an agent."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        
    def add_assistant_message(self, content: str = "", tool_calls: list[dict] | None = None):
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        
    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        })
        
    def get_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)
        
    def clear(self):
        self.messages.clear()
        self.state.clear()
