from pydantic import BaseModel, ConfigDict


class RaiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
