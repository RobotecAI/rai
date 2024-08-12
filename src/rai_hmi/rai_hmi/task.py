from enum import Enum

from langchain_core.pydantic_v1 import BaseModel


class Priority(str, Enum):
    highest = "highest"
    high = "high"
    medium = "medium"
    low = "low"
    lowest = "lowest"


class Task(BaseModel):
    name: str
    description: str
    priority: Priority
