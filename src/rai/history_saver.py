import base64
import datetime
import json
import os
import pickle
from typing import Any, Dict, List

import markdown

from rai.message import Message
from rai.vendors.vendors import AiVendor


class HistorySaver:
    def __init__(
        self, ai_vendor: AiVendor, history: List[Message], logs_dir: str = "logs"
    ):
        self.ai_vendor = ai_vendor
        self._history: List[Message] = history
        self.logs_dir = logs_dir

    def build_html(self):
        user_to_color = {
            "system": "#0c8571",
            "user": "#007bff",
            "assistant": "#28a745",
        }

        html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Chat History</title>\n</head>\n<body style="background-color:#f4f4f4; padding:20px; font-family:Arial, sans-serif;">\n'
        for msg in self._history:
            role_style = f"color:white; background-color:{user_to_color[msg.role]}; display:inline-block; border-radius:10px; padding:10px;"
            html += f'<h3 style="{role_style}">{msg.role}</h3>\n'
            for image in msg.images:
                html += f'<div align="center">\n<img src="data:image/png;base64, {image}" alt="image" style="width:600px; margin-top:20px;">\n</div>\n'

            message_html = markdown.markdown(msg.content)
            if msg.role == "user":
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

    def save_to_pkl(self, folder: str = ""):
        logs_dir = self.logs_dir
        log_dir = logs_dir.split(os.sep)[0]
        rest_dir = os.sep.join(logs_dir.split(os.sep)[1:])
        final_dir = os.path.join(log_dir, folder, rest_dir)
        os.makedirs(final_dir, exist_ok=True)
        pickle.dump(self._history, open(os.path.join(final_dir, "history.pkl"), "wb"))

    def save_to_json(self, folder: str = ""):
        logs_dir = self.logs_dir
        log_dir = logs_dir.split(os.sep)[0]
        rest_dir = os.sep.join(logs_dir.split(os.sep)[1:])
        final_dir = os.path.join(log_dir, folder, rest_dir)
        os.makedirs(final_dir, exist_ok=True)
        history_without_images: List[Dict[str, Any]] = []
        for msg in self._history:
            new_msg = msg.copy()
            if "images" in new_msg:
                new_msg.pop("images")
            history_without_images.append(new_msg)

        with open(os.path.join(final_dir, "history.json"), "w") as f:
            json.dump(history_without_images, f, indent=4)

    def get_html(self):
        html = self.build_html()
        return html

    def save_to_markdown(self):
        user_to_color = {
            "system": "#0c8571",
            "user": "#007bff",
            "assistant": "#28a745",
        }
        logs_dir = os.path.join(
            "logs",
            self.ai_vendor.__class__.__name__
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        os.makedirs(logs_dir, exist_ok=True)

        md = '<div style="background-color:#f4f4f4; padding:20px; font-family:Arial, sans-serif;">\n\n'
        img_num = 0
        for msg in self._history:
            md += f'### <div style="color:white; background-color:{user_to_color[msg.role]}; display:inline-block; border-radius:10px; padding:10px;">{msg.role}</div>\n'
            for image in msg.images:
                image_data = image
                image_path = os.path.join(
                    logs_dir, f"image_{str(img_num).zfill(3)}.png"
                )
                img_num += 1
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                md += f'<div align="center">\n<img src="{os.path.basename(image_path)}" alt="image" style="width:600px; margin-top:20px;">\n</div>\n'

            md += f'<div style="color:black; background-color:white; border-radius:15px; padding:15px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-top:10px;">\n{msg.content}\n</div>\n\n'

        with open(os.path.join(logs_dir, "history.md"), "w") as f:
            f.write(md)
