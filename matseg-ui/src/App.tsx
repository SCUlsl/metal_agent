import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Circle } from 'react-konva'; // 需要 npm install react-konva konva

// --- Types ---

type Point = { x: number; y: number; label: 1 | 0 }; // 1=正样本(前景), 0=负样本(背景)
type Box = { x: number; y: number; w: number; h: number };

const MatSegInterface = () => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [maskImage, setMaskImage] = useState<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<Point[]>([]); // 用户点击的点
  const [chatHistory, setChatHistory] = useState<{sender: string, text: string}[]>([]);
  const [inputText, setInputText] = useState("");
  const [stats, setStats] = useState<any>(null); // 统计数据

  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null); // <-- 新增 Session ID
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);   // <-- 新增图片 URL

  // 模拟加载图片
  // useEffect(() => {
  //   const img = new window.Image();
  //   img.src = "/placeholder_microstructure.png"; // 替换为上传的图片URL
  //   img.onload = () => setImage(img);
  // }, []);


  // --- 新增图片上传 Handler ---
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // A. 准备 FormData
    const formData = new FormData();
    formData.append('file', file);
    
    // B. 调用后端初始化接口
    try {
      const uploadResponse = await fetch('http://localhost:8000/api/v1/session/init', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) throw new Error("上传失败");

      const data: SessionInitResponse = await uploadResponse.json();
      
      // C. 更新前端状态
      setCurrentSessionId(data.session_id);
      
      // 注意：图片的完整URL需要拼接后端地址
      const fullImageUrl = `http://localhost:8000${data.image_url}`; 
      setCurrentImageUrl(fullImageUrl);
      
      // D. 在 Canvas 上显示图片 (需要重新加载 Image)
      const img = new window.Image();
      img.src = fullImageUrl;
      img.onload = () => setImage(img); // Canvas Workspace 将显示这张图
      
      setChatHistory([{ sender: 'system', text: `✅ 图像上传成功，Session ID: ${data.session_id}。请开始提问。` }]);

    } catch (error) {
      console.error("上传错误:", error);
      setChatHistory([{ sender: 'system', text: `❌ 图像上传或会话初始化失败: ${error.message}` }]);
    }
  };

  // --- Handlers ---

  // 1. 处理画布点击 (Visual Prompt)
  const handleCanvasClick = (e: any) => {
    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();
    // 获取相对于图片的坐标 (假设没有缩放，实际需转换)
    const isLeftClick = e.evt.button === 0;
    
    const newPoint: Point = {
      x: pos.x, 
      y: pos.y, 
      label: isLeftClick ? 1 : 0 // 左键选中，右键排除
    };

    setPoints([...points, newPoint]);
    
    // TODO: 触发 API 调用，发送新的点给后端进行 SAM Update
    refineSegmentation([...points, newPoint]); 
  };

// 2. 处理文字输入 (Semantic Prompt)
  const handleSendMessage = async () => {
    if (!inputText) return; // 只需要检查是否有输入文本

    const sessionId = currentSessionId || 'default-chat-session';  // 如果没有 Session ID，使用一个默认的 ID

    if (!currentSessionId) {
        setChatHistory(prev => [...prev, { sender: 'system', text: "当前处于纯文本对话模式，上传图片可启用视觉分析。" }]);
    }  // 提醒用户当前处于纯文本模式 (如果之前没有上传图片)

    // 【修改后的发送逻辑】
    const userMsg = { sender: 'user', text: inputText };
    setChatHistory(prev => [...prev, userMsg]);
    const currentInput = inputText; // 暂存输入
    setInputText(""); // 清空输入框

    try {
      // B. 发送请求给后端
      const response = await fetch('http://localhost:8000/api/v1/analyze/text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId, // <-- 必须携带 Session ID
            text_prompt: currentInput,
            chat_history: []
        }),
      });

        if (!response.ok) throw new Error("Agent 请求失败");

        // NOTE: 这里需要确保 AnalysisResponse 和 SessionInitResponse 已定义在文件顶部
        const data: AnalysisResponse = await response.json(); 

        // 【✅ 新增：处理 Agent 返回并更新聊天历史】
        const botMsg = { sender: 'agent', text: data.message };
        setChatHistory(prev => [...prev, botMsg]);

        // 额外的逻辑：如果 Agent 决定进行分割，这里可以触发 Canvas 更新或提示
        // if (data.stats?.is_segmentation_request) { ... } 
        
        // 【✅ 新增结束】

      } catch (error) {
        console.error("Agent 请求错误:", error);
        setChatHistory(prev => [...prev, { sender: 'system', text: `❌ 代理请求失败: ${error.message}` }]);
      }
  };

  // --- API Calls (Mock) ---
  const refineSegmentation = async (currentPoints: Point[]) => {
    console.log("Sending points to SAM:", currentPoints);
    // fetch('/api/segment/refine', ...)
  };

  const analyzeRequest = async (text: string) => {
    console.log("Sending text to Agents:", text);
    // fetch('/api/agent/analyze', ...)
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      
      {/* --- LEFT: Agent Chat --- */}
      <div className="w-1/4 border-r border-gray-700 flex flex-col p-4">
        <h2 className="text-xl font-bold mb-4 text-blue-400">🤖 Agent Thoughts</h2>

        {/* --- 新增 文件上传按钮 --- */}
        <label className="mb-4 block cursor-pointer bg-gray-700 hover:bg-gray-600 p-2 rounded text-center text-sm">
          {currentImageUrl ? "📂 重新上传图像" : "📂 上传显微图像"}
          <input 
            type="file" 
            accept="image/*" 
            className="hidden" 
            onChange={handleImageUpload} 
          />
        </label>
        {/* --- 结束 文件上传按钮 --- */}

        <div className="flex-1 overflow-y-auto bg-gray-800 p-3 rounded mb-4 space-y-2">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`p-2 rounded ${msg.sender === 'user' ? 'bg-blue-600 ml-4' : 'bg-gray-700 mr-4'}`}>
              <p className="text-sm">{msg.text}</p>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input 
            className="flex-1 bg-gray-700 p-2 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="描述图像特征..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button onClick={handleSendMessage} className="bg-blue-500 px-4 rounded hover:bg-blue-600">Send</button>
        </div>
      </div>

      {/* --- CENTER: Canvas Workspace --- */}
      <div className="flex-1 bg-black relative overflow-hidden flex items-center justify-center">
        <div className="absolute top-4 left-4 z-10 bg-black/50 p-2 rounded text-xs text-gray-300">
          Left Click: Add (+)<br/>Right Click: Remove (-)
        </div>
        {image && (
          <Stage 
            width={800} 
            height={600} 
            onMouseDown={handleCanvasClick}
            onContextMenu={(e) => e.evt.preventDefault()} // 禁止右键菜单
          >
            <Layer>
              {/* 原始显微图 */}
              <KonvaImage image={image} width={800} height={600} />
              
              {/* 分割 Mask (半透明叠加) */}
              {maskImage && <KonvaImage image={maskImage} width={800} height={600} opacity={0.5} />}
              
              {/* 用户点击的交互点 */}
              {points.map((p, i) => (
                <Circle 
                  key={i} 
                  x={p.x} y={p.y} 
                  radius={5} 
                  fill={p.label === 1 ? '#00ff00' : '#ff0000'} 
                  stroke="white" strokeWidth={1}
                />
              ))}
            </Layer>
          </Stage>
        )}
      </div>

      {/* --- RIGHT: Statistics & Control --- */}
      <div className="w-1/4 border-l border-gray-700 p-4 bg-gray-800">
        <h2 className="text-xl font-bold mb-4 text-green-400">📊 Statistics</h2>
        
        {/* 统计卡片 */}
        <div className="space-y-4">
          <div className="bg-gray-700 p-3 rounded">
            <h3 className="text-xs text-gray-400 uppercase">Target</h3>
            <p className="text-lg font-semibold">Lath Martensite</p>
          </div>
          
          <div className="bg-gray-700 p-3 rounded">
            <h3 className="text-xs text-gray-400 uppercase">Volume Fraction</h3>
            <p className="text-2xl font-mono text-green-300">15.4 %</p>
          </div>

          <div className="bg-gray-700 p-3 rounded">
            <h3 className="text-xs text-gray-400 uppercase">Mean Size (Area)</h3>
            <p className="text-lg font-mono">4.2 µm²</p>
          </div>

          <hr className="border-gray-600"/>
          
          <button className="w-full bg-gray-600 hover:bg-gray-500 p-2 rounded text-sm">
            Export Mask (.png)
          </button>
          <button className="w-full bg-gray-600 hover:bg-gray-500 p-2 rounded text-sm">
            Export CSV Report
          </button>
        </div>
      </div>
    </div>
  );
};

export default MatSegInterface;