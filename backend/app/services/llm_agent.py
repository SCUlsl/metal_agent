import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from app.core.memory import SessionMemory, TaskStep

# Qwen OpenAI 兼容地址
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 

class LLMAgent:
    def __init__(self):
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        self.client = OpenAI(api_key=self.api_key, base_url=QWEN_BASE_URL)
        self.llm_model = "qwen-max"
        
        # 延迟导入 VLMAgent 以避免循环依赖
        from app.services.vlm_agent import VLMAgent
        self.vlm_agent = VLMAgent()
        
        # --- Planner Prompt ---
        self.planner_prompt = """
        你是一个材料科学AI专家 (Agent 1)，负责图像分析任务的规划。
        
        **当前环境**:
        - 检查 [Current Context] 中的 "Image Loaded" 状态。
        - 如果显示 "✅ Image Loaded"，说明系统已持有图像，你可以直接使用工具，**无需**请求用户上传。
        
        **可用工具**:
        1. `sam3`: 图像分割。参数: {"prompts": ["string list"]}。用于识别和分割特定目标（如 "martensite", "black particles"）。
        2. `vlm`: 视觉理解。参数: {"query": "string"}。
           - **强烈建议**: 在进行分割前，先调用此工具询问 "这张图里有哪些主要特征？"，以便为 sam3 提供更准确的 prompt。
           - 当 sam3 分割失败时，也应调用此工具进行反思和修正。
        3. `finish`: 任务结束。参数: {"response": "给用户的最终回复"}。
        
        **输出格式 (必须是 JSON)**:
        {
            "thought": "分析当前状态（是否有图？上一步结果如何？），解释下一步计划...",
            "update_plan": [ ... ], // (可选) 定义或追加后续步骤列表，例如 [{"desc": "分析图像特征", "tool": "vlm", "params": {"query": "..."}}]
            "action": {             // 当前立即执行的动作
                "tool": "sam3" | "vlm" | "finish",
                "params": { ... }
            }
        }
        """

    def plan_and_execute(self, text: str, session_memory: SessionMemory) -> Dict[str, Any]:
        """
        Agent 1 的核心大脑：Planning Loop
        """
        # 1. 记录用户输入 (只在第一步记录，防止循环中重复记录)
        # 注意：endpoints.py 中如果是自动循环的后续步骤，传入的 text 是 "Continue..."
        # 这里我们可以简单判断一下，避免污染聊天记录
        if "Continue processing" not in text:
            session_memory.add_user_message(text)
        
        # 2. 构造 System Prompt (包含状态信息)
        plan_summary = session_memory.get_plan_summary()
        
        # 将最新的对话历史也加进去，让 Agent 1 知道用户到底想要什么
        chat_context = json.dumps(session_memory.chat_history[-3:], ensure_ascii=False)
        
        full_prompt = f"""{self.planner_prompt}

[Current Context]
{plan_summary}

[Recent Chat]
{chat_context}

[System Trigger]
{text}
"""
        
        try:
            # 3. 调用 LLM 获取决策
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": full_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            print(f"Agent 1 Raw Output: {content[:100]}...") # Debug log
            
            decision = json.loads(content)
            print(f"Agent 1 Decision: {decision.get('thought')}")
            print(f"Agent 1 Action: {decision.get('action', {}).get('tool')}")
            
            # 4. 如果 LLM 决定修改计划 (Refanning / Rollback)
            if "update_plan" in decision and decision["update_plan"]:
                new_steps = []
                for i, s in enumerate(decision['update_plan']):
                    # 容错处理
                    new_steps.append(TaskStep(
                        step_id=f"plan_{int(session_memory.current_step_index) + i + 1}", 
                        description=s.get('desc', 'No description'), 
                        tool=s.get('tool'), 
                        params=s.get('params', {})
                    ))
                
                # 动态追加任务
                # 简单策略：保留已执行的，丢弃未执行的旧计划，追加新计划
                executed_steps = session_memory.task_chain[:session_memory.current_step_index]
                session_memory.task_chain = executed_steps + new_steps

            return {
                "success": True,
                "thought": decision.get("thought", ""),
                "action": decision.get("action")
            }

        except Exception as e:
            print(f"Planning Error: {e}")
            return {
                "success": False, 
                "message": str(e), 
                "action": {"tool": "finish", "params": {"response": "系统规划出错，请检查后端日志。"}}
            }