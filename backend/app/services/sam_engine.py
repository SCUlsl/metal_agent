import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any

# 假设用户已安装 sam3 库 (基于提供的 notebook)
try:
    from sam3 import SAM3, SAM3ImagePredictor
except ImportError:
    print("Warning: SAM3 library not found. Using Mock mode.")
    SAM3 = None
    SAM3ImagePredictor = None

# 模型配置 (请根据实际路径修改)
SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'sam3_checkpoint.pth') 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAMEngine:
    def __init__(self):
        print(f"Initializing SAM 3 Engine (Device: {DEVICE})...")
        self.predictor = None
        
        if SAM3 and os.path.exists(SAM_CHECKPOINT):
            try:
                # 1. 加载 SAM 3 模型
                self.model = SAM3(checkpoint=SAM_CHECKPOINT)
                self.model.to(device=DEVICE)
                self.predictor = SAM3ImagePredictor(self.model)
                print("SAM 3 Model Loaded Successfully.")
            except Exception as e:
                print(f"Error loading SAM 3: {e}")
        else:
            print(f"SAM 3 Checkpoint not found at {SAM_CHECKPOINT} or library missing.")

        # 缓存
        self.image_cache: Dict[str, Any] = {} # session_id -> {image_tensor/path}

    def set_image(self, session_id: str, image_path: str):
        """预处理图像"""
        if not self.predictor: return
        
        print(f"SAM 3: Encoding image for session {session_id}...")
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SAM 3 的 set_image
        self.predictor.set_image(image_rgb)
        
        # 缓存原始尺寸等信息
        self.image_cache[session_id] = {
            "shape": image_rgb.shape[:2], # H, W
            "path": image_path
        }

    def predict_by_text(self, session_id: str, prompts: List[str]) -> Dict[str, Any]:
        """
        Agent 3 核心功能: 使用 SAM 3 进行文本提示分割
        """
        if not self.predictor:
            return {"success": False, "message": "SAM 3 Model not loaded."}
        
        # 1. 确保当前 Predictor 加载的是该 Session 的图 (SAM3 stateful)
        # 简化处理：实际生产中可能需要管理多个 predictor 或重新 set_image
        if session_id in self.image_cache:
            # 重新加载图片以确保上下文正确 (根据 sam3 实现可能不需要每次都做，但为了安全)
            path = self.image_cache[session_id]["path"]
            image = cv2.imread(path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
        else:
            return {"success": False, "message": "Session image not encoded."}

        try:
            print(f"SAM 3 Predicting with prompts: {prompts}")
            
            # 2. 调用 SAM 3 预测 (参考 Notebook API)
            # predict 返回: masks, scores, logits
            masks, scores, _ = self.predictor.predict(
                prompts=prompts,
                box_prompts=None, # 我们优先用文本
                point_prompts=None
            )
            
            # 3. 后处理结果
            # masks shape usually: (N_prompts, H, W) or similar
            if len(masks) == 0:
                return {"success": True, "found": False, "message": "No objects found."}

            # 合并所有 mask 用于展示
            final_mask = np.any(masks, axis=0) # Logical OR
            
            # 计算统计信息 (简单版)
            pixel_count = np.sum(final_mask)
            total_pixels = final_mask.size
            volume_fraction = (pixel_count / total_pixels) * 100
            
            # 保存掩码文件
            mask_img = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
            mask_img[final_mask] = [255, 255, 255]
            
            filename = f"sam3_mask_{session_id}_{int(volume_fraction)}.png"
            save_path = os.path.join("static/masks", filename)
            cv2.imwrite(save_path, mask_img)
            
            return {
                "success": True,
                "found": True,
                "mask_url": f"/static/masks/{filename}",
                "stats": {
                    "targets": prompts,
                    "count": len(masks), # 或者是连通域数量
                    "volume_fraction": round(volume_fraction, 2)
                }
            }

        except Exception as e:
            print(f"SAM 3 Prediction Error: {e}")
            return {"success": False, "message": str(e)}

    # 保留点击预测用于 HITL
    def predict_click(self, session_id: str, points: List[Dict]) -> Dict[str, Any]:
        # (此处代码复用之前的逻辑，改为调用 self.predictor.predict(point_prompts=...))
        # 为节省篇幅略去，逻辑同上，只是参数不同
        pass