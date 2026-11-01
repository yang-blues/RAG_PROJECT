from typing import Literal

from pydantic import BaseModel


class ApprovalRequest(BaseModel):
    thread_id: str
    user_id: str
    decision:Literal["yes","no"]