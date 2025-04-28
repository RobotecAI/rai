from pydantic import BaseModel, ConfigDict


class RaiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ros2BaseModel(RaiBaseModel):
    _prefix: str

    def get_msg_name(self) -> str:
        return f"{self._prefix}/{self.__class__.__name__}"
