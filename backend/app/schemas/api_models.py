from pydantic import BaseModel
from typing import List, Optional, Any

# --- 请求模型 (Request) ---

class TextAnalysisRequest(BaseModel):
    session_id: str
    text_prompt: str
    chat_history: Optional[List[Any]] = None

class InteractionPoint(BaseModel):
    x: float
    y: float
    label: int  # 1=正样本, 0=负样本

class InteractionRequest(BaseModel):
    session_id: str
    interaction_type: str # 'point_click' or 'box'
    points: List[InteractionPoint]
    previous_mask_id: Optional[str] = None

# --- 响应模型 (Response) ---

class SessionInitResponse(BaseModel):
    session_id: str
    image_url: str
    image_dims: List[int]

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    mask_url: Optional[str] = None
    stats: Optional[dict] = None