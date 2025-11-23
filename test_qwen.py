# test_qwen_openai.py

import os
import sys
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# --- 配置 ---
# Qwen 的 OpenAI 兼容模式 API 地址
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 

# 🚨 请务必替换为你项目目录下实际存在的图片路径 (本地路径)
TEST_IMAGE_PATH: str = "./test_image.png" 

TEXT_MODEL = "qwen-max"
VLM_MODEL = "qwen-vl-max"

TEXT_PROMPT = "请用一句话告诉我铁碳合金的金相组织中奥氏体的最高含碳量是多少？"
VLM_PROMPT = "请描述一下这张图片的主体内容是什么？"
# -----------


def local_image_to_base64(image_path: str) -> str:
    """将本地图片文件转换为 Base64 编码的字符串"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件未找到: {image_path}")
        
    # 使用 Pillow 确保图像格式一致性 (例如 PNG 或 JPEG)
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    # 转换为 JPEG 格式进行传输
    img.save(buffer, format="JPEG")
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_openai_client() -> OpenAI:
    """创建并配置 OpenAI 兼容客户端"""
    # 检查 API Key
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置！")
        
    return OpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL
    )


def test_qwen_text_api(client: OpenAI, model: str, prompt: str) -> bool:
    """测试 Qwen 文本模型 (qwen-max)"""
    print(f"--- 1. 启动 Qwen 文本 API 测试 ({model}) ---")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"✅ 文本模型调用成功，API Key 有效。")
        print(f"🤖 回复: {content[:100]}...")
        return True
            
    except Exception as e:
        print(f"❌ 文本模型调用发生异常: {e}")
        return False


def test_qwen_vlm_api(client: OpenAI, model: str, image_path: str, prompt: str) -> bool:
    """测试 Qwen VLM 模型 (qwen-vl-max) - 使用 Base64 编码"""
    print(f"\n--- 2. 启动 Qwen VLM API 测试 ({model}) ---")

    try:
        # 1. 编码图片
        base64_image = local_image_to_base64(image_path)
        
        # 2. 构建兼容 OpenAI 的多模态内容列表
        messages = [
            {
                "role": "user",
                "content": [
                    # Base64 图片部分
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    # 文本部分
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # 3. 调用 API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"✅ VLM 模型调用成功，VLM 权限和网络连接有效。")
        print(f"🤖 回复: {content[:100]}...")
        return True

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return False
    except Exception as e:
        print(f"❌ VLM 模型调用发生异常: {e}")
        return False


if __name__ == "__main__":
    
    # 1. 从 .env 文件加载环境变量
    load_dotenv()
    
    # 2. 创建客户端
    try:
        client = create_openai_client()
    except ValueError as e:
        print(f"🚨 致命错误: {e}")
        print("请确认您的 .env 文件在当前目录下，并且格式为 DASHSCOPE_API_KEY=\"您的密钥\"")
        sys.exit(1)

    print("--- 启动测试 (Qwen 兼容 OpenAI) ---")
    print(f"Base URL: {QWEN_BASE_URL}")
    print("--------------------------------------------------")

    # 3. 运行测试
    text_ok = test_qwen_text_api(client, TEXT_MODEL, TEXT_PROMPT)
    
    if text_ok:
        print("\n==================================================")
        test_qwen_vlm_api(client, VLM_MODEL, TEST_IMAGE_PATH, VLM_PROMPT)
    else:
        print("\n文本API测试失败，VLM测试已跳过。请检查您的API Key是否正确。")