from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.api_models import SessionInitResponse, TextAnalysisRequest, InteractionRequest, AnalysisResponse
from app.core.state import global_state
from app.core.memory import SessionMemory, TaskStep
import uuid
import shutil
import os

router = APIRouter()

# 确保静态目录存在
UPLOAD_DIR = "static/uploads"
MASK_DIR = "static/masks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

@router.post("/session/init", response_model=SessionInitResponse)
async def init_session(file: UploadFile = File(...)):
    """初始化会话，保存图片，预热 SAM"""
    session_id = str(uuid.uuid4())
    
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{session_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # === 【关键修改开始】 ===
    # 获取文件的绝对路径。这能解决 WSL/Docker 环境下 Agent 找不到文件的问题。
    abs_file_path = os.path.abspath(file_path)
    print(f"[Init] Image saved to absolute path: {abs_file_path}")

    # 初始化记忆 (存储绝对路径)
    new_session = SessionMemory(session_id=session_id, image_path=abs_file_path)
    global_state.sessions[session_id] = new_session

    # 预热 SAM (使用绝对路径)
    try:
        print(f"正在为会话 {session_id} 预计算 SAM 特征...")
        global_state.sam_engine.set_image(session_id, abs_file_path)
    except Exception as e:
        print(f"SAM 预热警告: {e}")
    # === 【关键修改结束】 ===
    
    return {
        "session_id": session_id,
        "image_url": f"/static/uploads/{safe_filename}", 
        "image_dims": [1024, 1024]
    }

@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    session_id = request.session_id
    # 获取会话记忆，如果不存在(纯文本)则创建一个
    session_memory = global_state.get_session(session_id)
    
    # === 自动任务执行循环 (Auto-Loop) ===
    MAX_STEPS = 5
    step_count = 0
    final_response_text = ""
    last_mask_url = None
    last_stats = None

    while step_count < MAX_STEPS:
        step_count += 1
        print(f"\n--- Step {step_count} Start (Session: {session_id}) ---")
        
        # 1. Agent 1 规划与决策
        current_prompt = request.text_prompt if step_count == 1 else "Continue processing based on the previous step result."
        
        plan_result = global_state.llm_agent.plan_and_execute(
            text=current_prompt,
            session_memory=session_memory
        )
        
        if not plan_result["success"]:
            return AnalysisResponse(success=False, message=f"Planning Failed: {plan_result['message']}")

        action = plan_result["action"]
        tool = action["tool"]
        params = action["params"]
        thought = plan_result.get("thought", "")

        # 补全任务链记录
        if session_memory.current_step_index >= len(session_memory.task_chain):
            new_step = TaskStep(
                step_id=f"auto_{step_count}",
                description=f"Auto-generated step for {tool}",
                tool=tool,
                params=params,
                status="running"
            )
            session_memory.task_chain.append(new_step)

        # 2. 执行工具
        if tool == "finish":
            print("--> Agent decides to FINISH.")
            final_response_text = params.get("response", "")
            session_memory.update_task_result(session_memory.current_step_index, "success", "Finished")
            break 
            
        elif tool == "sam3":
            prompts = params.get("prompts", [])
            print(f"--> Executing SAM 3 with prompts: {prompts}")
            
            sam_result = global_state.sam_engine.predict_by_text(session_id, prompts)
            
            status = "success" if sam_result["success"] and sam_result.get("found") else "failed"
            session_memory.update_task_result(
                session_memory.current_step_index, 
                status=status,
                result=sam_result,
                error=sam_result.get("message")
            )
            
            if sam_result.get("found"):
                last_mask_url = sam_result["mask_url"]
                last_stats = sam_result["stats"]
                
        elif tool == "vlm":
            query = params.get("query", "")
            print(f"--> Executing VLM Refinement: {query}")
            
            # 使用记忆中的绝对路径
            current_abs_path = session_memory.image_path
            
            if not current_abs_path or not os.path.exists(current_abs_path):
                vlm_res = {"success": False, "message": f"Image file not found at: {current_abs_path}"}
            else:
                vlm_res = global_state.llm_agent.vlm_agent.answer_visual_question(
                    current_abs_path, 
                    query
                )
            
            print(f"    VLM Result: {vlm_res.get('answer', 'No answer')[:50]}...")
            
            session_memory.update_task_result(
                session_memory.current_step_index, 
                status="success" if vlm_res["success"] else "failed",
                result=vlm_res.get("answer", vlm_res.get("message"))
            )

        # 3. 移动指针到下一步
        session_memory.current_step_index += 1
    
    if not final_response_text:
        final_response_text = "Task loop finished (max steps reached)."

    return AnalysisResponse(
        success=True,
        message=f"{final_response_text}\n\n(Thinking: {thought})",
        mask_url=last_mask_url,
        stats=last_stats
    )

@router.post("/analyze/interact", response_model=AnalysisResponse)
async def interact(request: InteractionRequest):
    """HITL 点击交互"""
    if not request.points:
        raise HTTPException(status_code=400, detail="未提供交互点。")

    result = global_state.sam_engine.predict_click(
        session_id=request.session_id,
        points=request.points
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])

    return AnalysisResponse(
        success=True,
        message="分割结果已根据您的点击更新。",
        mask_url=result['mask_url'],
        stats=result['stats']
    )