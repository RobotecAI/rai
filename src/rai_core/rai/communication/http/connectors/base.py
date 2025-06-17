from typing import TypeVar

from rai.communication.base_connector import BaseConnector
from rai.communication.http.messages import HTTPMessage

T = TypeVar("T", bound=HTTPMessage)


class HTTPBaseConnector(BaseConnector[T]): ...
