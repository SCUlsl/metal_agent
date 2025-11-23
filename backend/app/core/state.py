# backend/app/core/state.py

from app.services.sam_engine import SAMEngine
from app.services.llm_agent import LLMAgent
# 移除 VLMAgent 的直接引用，因为 LLM 会持有它，或者在这里保留也可以
# from app.services.vlm_agent import VLMAgent 
from app.core.memory import SessionMemory # <--- 新增
from typing import Dict

class GlobalState:
    def __init__(self):
        self.sam_engine = SAMEngine()
        self.llm_agent = LLMAgent()
        
        # 替代旧的 image_paths 字典，使用功能更强大的 Memory 字典
        # {session_id: SessionMemory}
        self.sessions: Dict[str, SessionMemory] = {} 

    def get_session(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            # 如果不存在，创建一个空的 (通常在 init 接口创建，这里防守性编程)
            self.sessions[session_id] = SessionMemory(session_id)
        return self.sessions[session_id]

global_state = GlobalState()