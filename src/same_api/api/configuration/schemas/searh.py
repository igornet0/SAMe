from typing import Optional
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SearchRequest(BaseModel):
    queries: List[str]
    method: Optional[str] = "hybrid"
    similarity_threshold: Optional[float] = 0.6
    max_results: Optional[int] = 10

class SearchResponse(BaseModel):
    results: Dict[str, List[Dict[str, Any]]]
    statistics: Dict[str, Any]
    processing_time: float

class InitializeRequest(BaseModel):
    catalog_file_path: Optional[str] = None
    search_method: Optional[str] = "hybrid"
    similarity_threshold: Optional[float] = 0.6