# main.py
import os
import uuid
import json
import re
import asyncio
import httpx
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, APIRouter, HTTPException, Request, Body, WebSocket
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from celery import Celery, Task
from redis import Redis
from anthropic import AsyncAnthropic
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

# --------------------------
# 配置模块
# --------------------------
load_dotenv()

class Settings(BaseSettings):
    API_HOST: str = Field(default=os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = Field(default=int(os.getenv("API_PORT", "8000")))
    REDIS_HOST: str = Field(default=os.getenv("REDIS_HOST", "localhost"))
    REDIS_PORT: int = Field(default=int(os.getenv("REDIS_PORT", "6379")))
    CELERY_BROKER_URL: str = Field(default=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
    CELERY_RESULT_BACKEND: str = Field(default=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"))
    ANTHROPIC_API_KEY: str = Field(default="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")
    GOOGLE_API_KEY: str = Field(default="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")
    CEREBRAS_API_KEY: str = Field(default="b6c1ee43b7741fb1182391d8cd0bf6716b67e838a442475187dec07f0080d6b6")
    TRELLIS_API_KEY: str = Field(default="b6c1ee43b7741fb1182391d8cd0bf6716b67e838a442475187dec07f0080d6b6")

    class Config:
        env_file = ".env"

settings = Settings()

# --------------------------
# Redis服务
# --------------------------
class RedisService:
    def __init__(self):
        self._client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
    
    def publish_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> int:
        return self._client.publish(
            f"task_stream:{task_id}", 
            json.dumps({"event": event_type, "data": data})
        )
    
    def store_response(self, task_id: str, response_data: Dict[str, Any]) -> bool:
        return self._client.set(
            f"task_response:{task_id}", 
            json.dumps(response_data), 
            ex=3600
        )
    
    def get_value(self, key: str) -> Optional[Dict]:
        if data := self._client.get(key):
            return json.loads(data)
        return None

redis_service = RedisService()

# --------------------------
# 数据模型
# --------------------------
class ClaudeResponse(BaseModel):
    status: str
    content: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    task_id: Optional[str] = None

class GeminiImageResponse(BaseModel):
    status: str
    model: Optional[str] = None
    images: Optional[List[Dict[str, str]]] = None
    text: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None

class CerebrasParseResponse(BaseModel):
    status: str
    content: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None

class TrellisResponse(BaseModel):
    id: str
    status: str
    data: Optional[Dict] = None

class StreamRequest(BaseModel):
    prompt: str
    threejs_code: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    additional_params: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    image_base64: Optional[str] = None

# --------------------------
# Celery配置
# --------------------------
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_time_limit=600,
)

# --------------------------
# AI任务基类
# --------------------------
class AsyncAITask(Task):
    _client = None
    
    async def client(self):
        raise NotImplementedError
    
    def run(self, *args, **kwargs):
        return asyncio.run(self._run_async(*args, **kwargs))

# --------------------------
# Claude 3任务实现
# --------------------------
class Claude3DGenTask(AsyncAITask):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def _run_async(self, task_id: str, image_base64: str, prompt: str, **kwargs):
        try:
            redis_service.publish_event(task_id, "start", {"task_id": task_id})
            
            message_content = [{
                "type": "text",
                "text": "Transform this 2D drawing into a Three.js 3D scene..."
            }]
            
            if image_base64:
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64.split(",")[-1]
                    }
                })
            
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{"role": "user", "content": message_content}],
                system="You are an expert 3D modeler..."
            )
            
            result = ClaudeResponse(
                status="success",
                content=response.content[0].text,
                model=response.model,
                usage=dict(response.usage),
                task_id=task_id
            ).model_dump()
            
            redis_service.publish_event(task_id, "complete", result)
            redis_service.store_response(task_id, result)
            return result
        
        except Exception as e:
            error = ClaudeResponse(
                status="error",
                error=str(e),
                error_type=type(e).__name__,
                task_id=task_id
            ).model_dump()
            redis_service.publish_event(task_id, "error", error)
            return error

# --------------------------
# Gemini图像生成任务
# --------------------------
class GeminiImageGenTask(AsyncAITask):
    def __init__(self):
        self.client = AsyncGenerativeModel("gemini-pro-vision")
        self.client.configure(api_key=settings.GOOGLE_API_KEY)

    async def _run_async(self, task_id: str, image_base64: str, prompt: str, **kwargs):
        try:
            redis_service.publish_event(task_id, "start", {"task_id": task_id})
            
            response = await self.client.generate_content(
                contents=[{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64.split(",")[-1]
                        }}
                    ]
                }]
            )
            
            images = [{
                "mime_type": part.inline_data.mime_type,
                "data": part.inline_data.data
            } for candidate in response.candidates for part in candidate.content.parts]
            
            result = GeminiImageResponse(
                status="success",
                images=images,
                text=response.text,
                task_id=task_id
            ).model_dump()
            
            redis_service.publish_event(task_id, "complete", result)
            redis_service.store_response(task_id, result)
            return result
        
        except Exception as e:
            error = GeminiImageResponse(
                status="error",
                error=str(e),
                error_type=type(e).__name__,
                task_id=task_id
            ).model_dump()
            redis_service.publish_event(task_id, "error", error)
            return error

# --------------------------
# Cerebras代码解析任务
# --------------------------
class CerebrasCodeParseTask(AsyncAITask):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.CEREBRAS_API_KEY)

    async def _run_async(self, task_id: str, code: str):
        try:
            redis_service.publish_event(task_id, "start", {"task_id": task_id})
            
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": f"""Parse this Three.js code:
                    {code}
                    Extract main object creation code"""
                }]
            )
            
            result = CerebrasParseResponse(
                status="success",
                content=response.content[0].text,
                model=response.model,
                task_id=task_id
            ).model_dump()
            
            redis_service.publish_event(task_id, "complete", result)
            redis_service.store_response(task_id, result)
            return result
        
        except Exception as e:
            error = CerebrasParseResponse(
                status="error",
                error=str(e),
                task_id=task_id
            ).model_dump()
            redis_service.publish_event(task_id, "error", error)
            return error

# --------------------------
# PiAPI集成任务
# --------------------------
class PiAPITask:
    API_URL = "https://api.piapi.ai/api/v1/task"
    
    async def create_task(self, input_data: Dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.API_URL,
                headers={"x-api-key": settings.TRELLIS_API_KEY},
                json=input_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

# --------------------------
# FastAPI路由
# --------------------------
app = FastAPI()
router = APIRouter()

@router.post("/queue/{task_type}")
async def queue_task(task_type: str, request: StreamRequest):
    task_id = request.task_id or str(uuid.uuid4())
    
    task_map = {
        "3d": Claude3DGenTask(),
        "image": GeminiImageGenTask(),
        "parse": CerebrasCodeParseTask()
    }
    
    if task_type not in task_map:
        raise HTTPException(400, "Unsupported task type")
    
    # 参数验证
    if task_type in ["3d", "image"] and not request.image_base64:
        raise HTTPException(400, "Image required")
    
    if task_type == "parse" and not request.threejs_code:
        raise HTTPException(400, "Three.js code required")
    
    # 提交任务
    task_args = {
        "3d": [task_id, request.image_base64, request.prompt],
        "image": [task_id, request.image_base64, request.prompt],
        "parse": [task_id, request.threejs_code]
    }[task_type]
    
    task_map[task_type].apply_async(
        args=task_args,
        task_id=task_id
    )
    
    return {"task_id": task_id}

@router.post("/trellis/task")
async def create_trellis_task(input_data: Dict):
    return await PiAPITask().create_task(input_data)

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if data := redis_service.get_value(f"task_response:{task_id}"):
        return data
    return {"status": "pending"}

app.include_router(router, prefix="/api")

# --------------------------
# 运行配置
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)