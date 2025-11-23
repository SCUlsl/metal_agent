import os
import base64
from io import BytesIO
from typing import Dict, Any
from openai import OpenAI
from PIL import Image

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 

class VLMAgent:
    """
    Agent 2 (Vision Understanding): 
    只负责视觉理解。接收图片和文本问题，返回对图片内容的文本描述。
    不负责检测坐标，也不负责分割。
    """
    def __init__(self):
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        self.client = OpenAI(api_key=self.api_key, base_url=QWEN_BASE_URL)
        self.vlm_model = "qwen-vl-max"

    def _local_image_to_base64(self, image_path: str) -> str:
        try:
            img = Image.open(image_path).convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            raise Exception(f"图片处理失败: {e}")

    def answer_visual_question(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        视觉问答 (VQA) 接口。
        question: 由 Agent 1 生成的针对图片的具体问题。
        """
        if not self.client:
            return {"success": False, "message": "VLM 客户端未初始化"}
        
        try:
            base64_image = self._local_image_to_base64(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            # 调用 VLM
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                temperature=0.2 # 稍微增加一点创造性用于描述
            )
            
            description = response.choices[0].message.content
            return {"success": True, "answer": description}

        except Exception as e:
            print(f"Agent 2 VQA Error: {e}")
            return {"success": False, "message": str(e)}