from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time

class TaskStep(BaseModel):
    """任务链中的单个步骤"""
    step_id: str
    description: str        
    tool: str               # e.g. "sam3", "vlm", "finish"
    params: Dict[str, Any]  
    status: str = "pending" # pending, running, success, failed
    result: Optional[Any] = None
    error_msg: Optional[str] = None

class SessionMemory:
    def __init__(self, session_id: str, image_path: Optional[str] = None):
        self.session_id = session_id
        self.image_path = image_path
        
        # 1. 对话记忆
        self.chat_history: List[Dict[str, str]] = []
        
        # 2. 任务链记忆
        self.task_chain: List[TaskStep] = []
        self.current_step_index: int = 0
        
    def add_user_message(self, text: str):
        self.chat_history.append({"role": "user", "content": text})
        
    def add_ai_message(self, text: str):
        self.chat_history.append({"role": "assistant", "content": text})

    def get_plan_summary(self) -> str:
        """生成给 LLM 看的当前计划状态摘要"""
        # === 【关键修改】 ===
        # 在摘要开头明确标注图片状态
        img_status = f"✅ Image Loaded: {self.image_path}" if self.image_path else "❌ No Image Loaded"
        summary = f"Session Context:\n- {img_status}\n\n"
        # ===================

        if not self.task_chain:
            summary += "Current Task Chain: [Empty (Waiting for plan)]\n"
        else:
            summary += "Current Task Chain:\n"
            for i, step in enumerate(self.task_chain):
                cursor = "->" if i == self.current_step_index else "  "
                status_info = f"(Status: {step.status})"
                if step.result:
                    # 简化结果显示，防止 Token 溢出
                    res_str = str(step.result)[:200] + "..." if len(str(step.result)) > 200 else str(step.result)
                    status_info += f" Result: {res_str}"
                    
                summary += f"{cursor} Step {i+1}: {step.description} | Tool: {step.tool} {status_info}\n"
        
        return summary

    def update_task_result(self, step_index: int, status: str, result: Any = None, error: str = None):
        if 0 <= step_index < len(self.task_chain):
            self.task_chain[step_index].status = status
            self.task_chain[step_index].result = result
            self.task_chain[step_index].error_msg = error