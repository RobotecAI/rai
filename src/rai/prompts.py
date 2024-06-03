from typing import List

from rai.message import ConstantMessage


def enumerate_actions(actions: List[str]) -> str:
    return "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])


def list_actions(actions: List[str]) -> str:
    return "\n\n".join([f"{action}" for action in actions]) + "\n"


class EnumeratedActionPropmt(ConstantMessage):
    QUESTION: str = "Please respond with only 1 number, without additional characters."

    def __init__(self, actions: List[str]) -> None:
        enumerated = enumerate_actions(actions)
        content = f"{enumerated}\n{self.QUESTION}"
        super().__init__(role="user", content=content)


class SimpleActionPrompt(ConstantMessage):
    def __init__(self, actions: List[str]) -> None:
        content = (
            "Please select one of the following actions:\n"
            + "\n".join(actions)
            + "\nRespond with only the action's name. Do not add any extra characters."
        )
        super().__init__(role="user", content=content)
