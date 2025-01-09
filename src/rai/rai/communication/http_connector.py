# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import json
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Callable, Dict, List

from rai.communication.base_connector import BaseConnector, BaseMessage


class MessageHandler(SimpleHTTPRequestHandler):
    """Handler for HTTP requests serving a simple message viewer."""

    def do_GET(self):
        """Serve either the main HTML page or message data."""
        if self.path == "/":
            self._serve_html()
        elif self.path == "/messages":
            self._serve_messages()

    def _serve_html(self):
        """Serve the main HTML interface."""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(self.server.get_html_content().encode())

    def _serve_messages(self):
        """Serve message data as JSON."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.server.messages).encode())


class SimpleHTTPServer(HTTPServer):
    """HTTP server that maintains a list of messages."""

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, message: str):
        """Add a new message with timestamp."""
        self.messages.append(
            {"timestamp": datetime.now().isoformat(), "content": message}
        )

    def get_html_content(self) -> str:
        """Return the HTML content for the web interface."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Message Viewer</title>
            <style>
                body { max-width: 800px; margin: 0 auto; padding: 20px; }
                .message { border: 1px solid #ddd; margin: 10px 0; padding: 10px; }
                .timestamp { color: #666; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <h1>Messages</h1>
            <div id="messages"></div>
            <script>
                function updateMessages() {
                    fetch('/messages')
                        .then(response => response.json())
                        .then(messages => {
                            document.getElementById('messages').innerHTML =
                                messages.slice().reverse().map(msg => `
                                    <div class="message">
                                        <div class="timestamp">${msg.timestamp}</div>
                                        <div>${msg.content}</div>
                                    </div>
                                `).join('');
                        });
                }
                setInterval(updateMessages, 1000);
                updateMessages();
            </script>
        </body>
        </html>
        """


class HTTPConnector(BaseConnector):
    """Connector that displays messages via a web interface."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port

        self.server = SimpleHTTPServer((self.host, self.port), MessageHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        atexit.register(self.cleanup)
        print(f"Server started at http://{self.host}:{self.port}")

    def send_message(self, msg: BaseMessage, target: str) -> None:
        """Add message to the web interface."""
        self.server.add_message(str(msg.content))

    def receive_message(self, source: str) -> BaseMessage:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support receiving messages"
        )

    def start_action(
        self, target: str, on_feedback: Callable, on_finish: Callable = lambda _: None  # type: ignore
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport starting actions"
        )

    def terminate_action(self, action_handle: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport terminating actions"
        )

    def send_and_wait(self, target: str) -> BaseMessage:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport sending messages"
        )

    def cleanup(self):
        """Clean up server resources."""
        self.server.shutdown()
        self.server.server_close()
