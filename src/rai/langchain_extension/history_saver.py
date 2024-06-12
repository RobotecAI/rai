import os
from typing import Sequence

import markdown
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


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
            text = msg.content
            if isinstance(msg.content, list):
                text = ""
                for content in msg.content:
                    if content["type"] == "text":
                        text += content["text"]
                    elif content["type"] == "image_url":
                        html += f'<div align="center">\n<img src="{content["image_url"]["url"]}" alt="image" style="width:600px; margin-top:20px;">\n</div>\n'

            if isinstance(msg, AIMessage):
                text += "\n\n**Function calls:**\n "
                for tool_call in msg.tool_calls:
                    text += f"{tool_call['name']}({', '.join([f'{k}={v}' for k, v in tool_call['args'].items()])})\n"

            if isinstance(msg, ToolMessage):
                text = f"**Tool call output:**\n{msg.content}\n\n**Tool call id:**\n{msg.tool_call_id}\n"
            message_html = markdown.markdown(text)
            if msg.type == "human":
                message_html = message_html.replace("\n", "<br>")
            html += f'<div style="color:black; background-color:white; border-radius:15px; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-top:10px;">\n{message_html}\n</div>\n'

        html += "</body>\n</html>"
        return html

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
