

from pydantic import BaseModel
from knowledge.schemas import EntityType


class SearchResult(BaseModel):
    id: int
    type: EntityType
    name: str
    score: float         
    source: str            