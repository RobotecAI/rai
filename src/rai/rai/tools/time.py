import time

from langchain.tools import tool


@tool
def sleep_max_5s(n: int):
    """Wait n seconds, max 5s"""
    if n > 5:
        n = 5

    time.sleep(n)
