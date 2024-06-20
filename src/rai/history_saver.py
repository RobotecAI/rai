import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import markdown
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class HistoryMessage(TypedDict):
    type: str
    text: str
    images: Optional[List[str]]


class HistorySaver:
    def __init__(self, history: Sequence[BaseMessage], logs_dir: str = "logs"):
        self._history: Sequence[BaseMessage] = history
        self.logs_dir = logs_dir

    def build_html(self):
        user_to_color = {
            "system": "#0c8571",
            "human": "#007bff",
            "ai": "#28a745",
            "tool": "#6c757d",
        }

        html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Chat History</title>\n</head>\n<body style="background-color:#f4f4f4; padding:20px; font-family:Arial, sans-serif;">\n'
        for msg in self._history:
            role_style = f"color:white; background-color:{user_to_color[msg.type]}; display:inline-block; border-radius:10px; padding:10px;"
            html += f'<h3 style="{role_style}">{msg.__class__.__name__}</h3>\n'
            history_message = self._handle_message(msg)
            if isinstance(history_message["images"], list):
                for image_url in history_message["images"]:
                    html += f'<div align="center">\n<img src="{image_url}" alt="image" style="width:600px; margin-top:20px;">\n</div>\n'
            message_html = markdown.markdown(history_message["text"])
            if msg.type == "human":
                message_html = message_html.replace("\n", "<br>")
            html += f'<div style="color:black; background-color:white; border-radius:15px; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-top:10px;">\n{message_html}\n</div>\n'

        html += "</body>\n</html>"
        return html

    def _handle_message(self, msg: BaseMessage) -> HistoryMessage:
        if isinstance(msg, SystemMessage):
            return self._handle_system_message(msg)
        elif isinstance(msg, HumanMessage):
            return self._handle_human_message(msg)
        elif isinstance(msg, AIMessage):
            return self._handle_assistant_message(msg)
        elif isinstance(msg, ToolMessage):
            return self._handle_tool_message(msg)
        else:
            raise ValueError(f"Unexpected message type: {msg.__class__.__name__}")

    def _handle_list_content(
        self, content: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        images: List[str] = [
            content["image_url"]["url"]
            for content in content
            if content["type"] == "image_url"
        ]
        text: List[str] = [
            content["text"] for content in content if content["type"] == "text"
        ]
        return text, images

    def _handle_tool_message(self, msg: ToolMessage) -> HistoryMessage:
        text: str = f"Function call {msg.tool_call_id} output: "
        images = None
        if isinstance(msg.content, list):
            texts, images = self._handle_list_content(msg.content)
            text += ", ".join(texts)
        else:
            text += msg.content
        return HistoryMessage(type=msg.type, text=text, images=images)

    def _handle_human_message(self, msg: HumanMessage) -> HistoryMessage:
        text = ""
        images = None
        if isinstance(msg.content, list):
            texts, images = self._handle_list_content(msg.content)
            text += ", ".join(texts)
        else:
            text = msg.content
        return HistoryMessage(type=msg.type, text=text, images=images)

    def _handle_assistant_message(self, msg: AIMessage) -> HistoryMessage:
        text = ""
        images = None
        if isinstance(msg.content, list):
            texts, images = self._handle_list_content(msg.content)
            text += ", ".join(texts)
        else:
            text = msg.content

        if len(msg.tool_calls) > 0:
            text += "\n\n**Requested function calls:**\n"
            for tool_call in msg.tool_calls:
                text += f"{tool_call['name']}({', '.join([f'{k}={v}' for k, v in tool_call['args'].items()])})\n"

        return HistoryMessage(type=msg.type, text=text, images=images)

    def _handle_system_message(self, msg: SystemMessage) -> HistoryMessage:
        text = ""
        images = None
        if isinstance(msg.content, list):
            texts, images = self._handle_list_content(msg.content)
            text += ", ".join(texts)
        else:
            text = msg.content
        return HistoryMessage(type=msg.type, text=text, images=images)

    def save_to_html(self, folder: str = "") -> str:
        html = self.build_html()
        logs_dir = self.logs_dir
        log_dir = logs_dir.split(os.sep)[0]
        rest_dir = os.sep.join(logs_dir.split(os.sep)[1:])
        final_dir = os.path.join(log_dir, folder, rest_dir)
        os.makedirs(final_dir, exist_ok=True)
        out_file = os.path.join(final_dir, "history.html")
        with open(out_file, "w") as f:
            f.write(html)
        return out_file
