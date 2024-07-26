import time

from langchain.tools import tool


@tool
def sleep(n: int):
    """Wait n seconds"""

    time.sleep(n)
