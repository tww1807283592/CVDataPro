# backend/app/main.py
"""
数据合成平台主应用入口
高性能异步FastAPI应用，支持多线程并发处理
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.v1 import (
    projects, images, matting, synthesis,
    datasets, upload, download
)
from app.config.settings import settings
from app.config.database import engine, Base
from app.config.redis import redis_client
from app.core.middleware import (
    LoggingMiddleware,
    MemoryMonitorMiddleware,
    RequestLimitMiddleware
)
from app.core.exceptions import setup_exception_handlers
from app.core.events import startup_event, shutdown_event
from app.utils.logger_utils import setup_logging
from app.utils.memory_utils import MemoryManager


class AsyncContextManager:
    """异步上下文管理器"""

    def __init__(self):
        self.memory_manager = MemoryManager()

    async def startup(self):
        """应用启动时执行"""
        # 设置日志
        setup_logging()

        # 创建数据库表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 初始化Redis连接
        await redis_client.initialize()

        # 启动内存管理器
        await self.memory_manager.start_monitoring()

        # 创建必要的目录
        os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
        os.makedirs(settings.DATASET_PATH, exist_ok=True)
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        os.makedirs(f"{settings.UPLOAD_PATH}/temp", exist_ok=True)

        print(f"🚀 数据合成平台启动完成 - 环境: {settings.ENVIRONMENT}")

    async def shutdown(self):
        """应用关闭时执行"""
        # 停止内存管理器
        await self.memory_manager.stop_monitoring()

        # 关闭Redis连接
        await redis_client.close()

        # 清理临时文件
        import shutil
        temp_path = f"{settings.UPLOAD_PATH}/temp"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        print("📴 数据合成平台已关闭")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    context_manager = AsyncContextManager()
    await context_manager.startup()
    try:
        yield
    finally:
        await context_manager.shutdown()


# 创建FastAPI应用实例
app = FastAPI(
    title="数据合成平台 API",
    description="高性能计算机视觉数据合成系统",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MemoryMonitorMiddleware)
app.add_middleware(RequestLimitMiddleware, max_requests_per_minute=60)

# 设置异常处理器
setup_exception_handlers(app)

# 挂载静态文件
if os.path.exists(settings.UPLOAD_PATH):
    app.mount("/storage", StaticFiles(directory=settings.UPLOAD_PATH), name="storage")

# 注册API路由
app.include_router(
    projects.router,
    prefix="/api/v1/projects",
    tags=["项目管理"]
)

app.include_router(
    images.router,
    prefix="/api/v1/images",
    tags=["图像处理"]
)

app.include_router(
    matting.router,
    prefix="/api/v1/matting",
    tags=["智能抠图"]
)

app.include_router(
    synthesis.router,
    prefix="/api/v1/synthesis",
    tags=["图像合成"]
)

app.include_router(
    datasets.router,
    prefix="/api/v1/datasets",
    tags=["数据集管理"]
)

app.include_router(
    upload.router,
    prefix="/api/v1/upload",
    tags=["文件上传"]
)

app.include_router(
    download.router,
    prefix="/api/v1/download",
    tags=["文件下载"]
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "数据合成平台 API",
        "version": "1.0.0",
        "status": "运行中",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        from app.db.session import get_async_session
        async with get_async_session() as session:
            await session.execute("SELECT 1")

        # 检查Redis连接
        await redis_client.ping()

        return {
            "status": "healthy",
            "database": "connected",
            "redis": "connected",
            "memory_usage": f"{MemoryManager.get_memory_usage():.2f}%"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/metrics")
async def metrics():
    """系统指标"""
    from app.utils.memory_utils import MemoryManager
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "active_connections": len(asyncio.all_tasks()),
        "memory_manager": {
            "status": "active",
            "cleanup_count": MemoryManager._cleanup_count,
            "last_cleanup": MemoryManager._last_cleanup_time
        }
    }


if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS,
        loop="asyncio",
        log_level="info" if settings.DEBUG else "warning"
    )