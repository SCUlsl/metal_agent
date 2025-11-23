from dotenv import load_dotenv
# 1. 优先加载环境变量
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router

# 注意：不再需要在 main.py 中手动连接 agent，因为 LLMAgent 现在会自动初始化 VLMAgent

app = FastAPI(title="MatSeg Backend")

# 2. 配置 CORS (允许前端访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # 允许前端的地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 挂载静态文件目录 (让前端能访问到 uploads 和 masks 图片)
# 确保目录存在，防止报错
import os
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/masks", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. 注册路由
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"Hello": "MatSeg Backend is running!", "version": "v2.0-Memory-Enabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)