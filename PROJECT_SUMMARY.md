# MatSeg 项目速览

本项目提供一个端到端的材料显微组织分割助手，后端由 FastAPI + 多智能体 (LLM + VLM + SAM3) 负责规划和执行任务，前端由 React/Vite 构建交互式可视化界面，便于与其他大模型或人员共享上下文。

## 🔧 技术栈与核心能力
- **后端**：FastAPI、Pydantic、OpenAI 兼容客户端 (Qwen Max/VL)；封装了 LLM/VLM 调度和 SAM3 分割引擎。
- **前端**：React 18、TypeScript、Vite、Tailwind、Konva (画布交互)；支持图像上传、聊天提示和点击式提示。
- **多模态协作**：LLMAgent 规划、VLMAgent 做视觉理解、SAMEngine 负责 SAM3 分割，配合会话内存实现自动多步推理。

## 📁 目录结构 (精简)
```
metal_agent/
├── backend/                 # FastAPI 服务根目录
│   ├── main.py              # 应用入口，配置 CORS/静态资源/路由
│   ├── requirements.txt     # Python 依赖
│   ├── app/
│   │   ├── api/endpoints.py # REST API：会话初始化、文本分析、交互分割
│   │   ├── core/
│   │   │   ├── state.py     # 全局状态：SAMEngine + LLMAgent + 会话字典
│   │   │   └── memory.py    # SessionMemory/TaskStep，记录对话与计划链
│   │   ├── schemas/api_models.py # Pydantic 请求/响应模型
│   │   └── services/
│   │       ├── llm_agent.py # 调用 Qwen-Max，规划动作并驱动子代理
│   │       ├── vlm_agent.py # 调用 Qwen-VL-Max 进行视觉问答
│   │       └── sam_engine.py# SAM3 推理封装，保存 mask/统计
│   └── static/              # 上传图片与 mask 缓存
├── matseg-ui/               # Vite React 前端
│   ├── src/App.tsx          # 单页应用：上传/聊天/Canvas/统计
│   ├── public/              # 静态占位资源
│   └── package.json         # 前端依赖
├── test_qwen.py             # Qwen API 连通性与多模态测试脚本
└── test_image.png           # 样例图片 (供测试脚本使用)
```

## 🧩 后端模块说明
- `backend/main.py`
  - 在启动时 `load_dotenv()` 读取环境变量 (如 `DASHSCOPE_API_KEY`)。
  - 配置 CORS 允许 `http://localhost:5173` 前端访问，挂载 `static/` 目录便于图片/掩膜访问。
  - 注册 API 路由并提供健康检查根路径。

- `app/api/endpoints.py`
  - `/session/init`：接受图片 `UploadFile`，持久化到 `static/uploads/`，创建 `SessionMemory`，调用 `SAMEngine.set_image()` 预编码图像，返回 `session_id` 与可访问的 `image_url`。
  - `/analyze/text`：接收文本提示并驱动“自动任务循环”。LLM 每轮规划 -> 选择工具 (`sam3`/`vlm`/`finish`) -> 记录 `TaskStep` 状态；循环最多 5 步，可自动串联视觉理解和分割并汇报最终消息/最新 mask/stats。
  - `/analyze/interact`：处理 HITL 点选 (正/负样本) 请求，调用 `predict_click`（占位）更新 mask。

- `app/core/memory.py`
  - `SessionMemory` 保存 `image_path`、最近聊天、任务链、当前指针。`get_plan_summary()` 会生成包含“是否已加载图像”的摘要作为 LLM 上下文，`update_task_result()` 用于回写状态和结果。

- `app/services/llm_agent.py`
  - 通过 `OpenAI(api_key, base_url=QWEN_BASE_URL)` 调用通义千问兼容接口。
  - `planner_prompt` 约束 LLM 作为“Agent 1”规划者，定义可用工具与 JSON 输出协议，并强调在分割前先调用 `vlm`。
  - `plan_and_execute()` 会：
    1. 将用户输入写入 `SessionMemory`。
    2. 拼接 `[Current Context]` + 最近聊天成 System Prompt。
    3. 解析 LLM JSON 响应，必要时用 `update_plan` 重写任务链。
    4. 返回当前要执行的工具和参数供 API 循环调用。

- `app/services/vlm_agent.py`
  - 将本地图像转 Base64 嵌入 `image_url` 内容，调用 `qwen-vl-max` 完成视觉问答 (如“图像包含哪些特征”)；错误时返回描述。

- `app/services/sam_engine.py`
  - 懒加载 `SAM3` 模型 (若 `sam3` 库或 checkpoint 缺失则降级为 mock)。
  - `set_image()` 负责读取图片、转换 RGB、编码到 predictor 并缓存原始尺寸/路径。
  - `predict_by_text()`：根据 LLM 传入的文本 prompts 运行 SAM3，合并多掩膜、计算体积分数、落盘 mask (`static/masks`) 并返回 URL + 统计；若模型未加载或 Session 未预热会返回错误。
  - `predict_click()` 预留：用于将交互点转换为 SAM 输入（尚未实现）。

- `app/schemas/api_models.py`
  - 定义 `SessionInitResponse`、`AnalysisResponse` 以及 `TextAnalysisRequest` / `InteractionRequest` + `InteractionPoint`，保持请求/响应结构清晰。

- `backend/requirements.txt`
  - 最小依赖集合 (FastAPI、Uvicorn、Pydantic、Numpy、OpenCV 等)，Torch/SAM3 需按 CUDA 环境单独安装。

## 💻 前端模块说明 (`matseg-ui/`)
- `src/App.tsx`
  - **上传区**：`handleImageUpload()` 发送 `FormData` 到 `/session/init`，成功后缓存 `session_id` 与远程图片 URL，并将图像加载进 `Konva` 画布。
  - **聊天区**：`handleSendMessage()` 将用户输入 + `session_id` POST 到 `/analyze/text`，展示 Agent 返回的 `AnalysisResponse.message`；若无 session 则提示处于纯文本模式。
  - **画布区**：使用 `Stage/Layer` 绘制原图、半透明 mask、用户点击点；`handleCanvasClick()` 根据鼠标左右键追加 `Point` 并调用 `refineSegmentation()` (占位)。
  - **统计侧栏**：目前为静态示例卡片，可后续绑定 `AnalysisResponse.stats`。
- 其他 `src/*.tsx|css` 文件维持 Vite 模板默认结构；`public/placeholder_microstructure.png` 作为示例图。

## 🧪 辅助脚本
- `test_qwen.py`
  - 通过 `dotenv` 读取 `DASHSCOPE_API_KEY`，使用相同的 OpenAI 兼容客户端依次测试 `qwen-max` (文本) 与 `qwen-vl-max` (多模态) API；`TEST_IMAGE_PATH` 默认指向仓库根的 `test_image.png`。
  - 提供本地自检手段，确保调用凭证、网络与图像编码链路正常。

## 🔄 典型工作流
1. **上传图像**：前端将文件 POST 至 `/session/init`，后端保存图像、初始化 `SessionMemory` 并预热 SAM3。
2. **用户提问/指令**：文本提示发送到 `/analyze/text`。
3. **Agent 自动循环**：
   - LLMAgent 阅读 `SessionMemory` + 最近对话，输出 JSON 决策。
   - 若选择 `vlm`：VLMAgent 读取本地图片并调用 `qwen-vl-max` 获取描述，结果写回任务链。
   - 若选择 `sam3`：SAMEngine 以文本 prompts 执行 SAM3 分割，生成 mask 图/统计。
   - LLM 根据历史结果决定下一动作或 `finish`，循环最多 5 次。
4. **结果返回**：API 将最终消息、最新 `mask_url` 与统计数据回传前端；前端可叠加 mask、展示指标或导出结果。
5. **交互细化 (可选)**：用户在画布上点选区域，未来将通过 `/analyze/interact` 触发 `predict_click()` 做 HITL 修正。

## 📌 共享给其他大模型的要点
- 运行需 `DASHSCOPE_API_KEY`，且 `sam3` checkpoint 路径为 `backend/app/models/sam3_checkpoint.pth` (需自行准备)。
- 所有静态资源通过 `http://<backend>/static/...` 暴露，前端默认访问 `localhost:8000`。
- Session 状态由服务端 `SessionMemory` 管理，只需在所有请求中附带 `session_id`。上传图像前的文本对话亦被支持，只是无视觉工具可用。
- Agent 执行链完全可复用：遵循 JSON 规划协议 (`sam3`/`vlm`/`finish`) 即可模拟当前逻辑。

> 以上内容可直接提供给其他大模型或协作者，使其快速了解 MatSeg 项目的能力边界与接口契约。
