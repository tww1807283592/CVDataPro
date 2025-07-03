# backend/app/config/settings.py
"""
应用配置设置
支持从环境变量加载配置，包含所有核心配置项
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """应用配置类"""

    # 基础配置
    APP_NAME: str = Field(default="数据合成平台", description="应用名称")
    VERSION: str = Field(default="1.0.0", description="版本号")
    ENVIRONMENT: str = Field(default="development", description="环境")
    DEBUG: bool = Field(default=True, description="调试模式")
    SECRET_KEY: str = Field(description="安全密钥")

    # 服务器配置
    HOST: str = Field(default="0.0.0.0", description="服务器地址")
    PORT: int = Field(default=8000, description="服务器端口")
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        description="允许的主机列表"
    )

    # 数据库配置
    DATABASE_URL: str = Field(description="数据库连接URL")
    DATABASE_POOL_SIZE: int = Field(default=20, description="数据库连接池大小")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, description="数据库连接池溢出")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, description="连接超时时间")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, description="连接回收时间")

    # Redis配置
    REDIS_URL: str = Field(description="Redis连接URL")
    REDIS_POOL_SIZE: int = Field(default=20, description="Redis连接池大小")
    REDIS_TIMEOUT: int = Field(default=30, description="Redis超时时间")

    # 文件存储配置
    UPLOAD_PATH: str = Field(default="./storage/uploads", description="上传文件路径")
    DATASET_PATH: str = Field(default="./storage/datasets", description="数据集路径")
    MODEL_PATH: str = Field(default="./storage/models", description="模型文件路径")
    CACHE_PATH: str = Field(default="./storage/cache", description="缓存文件路径")
    TEMP_PATH: str = Field(default="./storage/temp", description="临时文件路径")

    # 文件大小限制 (MB)
    MAX_FILE_SIZE: int = Field(default=100, description="单文件最大大小(MB)")
    MAX_BATCH_SIZE: int = Field(default=1000, description="批量处理最大数量")
    MAX_FOLDER_SIZE: int = Field(default=5000, description="文件夹最大文件数")

    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=[
            "jpg", "jpeg", "png", "bmp", "tiff", "tif",
            "webp", "gif", "ico", "psd", "svg"
        ],
        description="支持的图像格式"
    )

    # 并发和性能配置
    MAX_WORKERS: int = Field(default=4, description="最大工作进程数")
    MAX_CONCURRENT_TASKS: int = Field(default=10, description="最大并发任务数")
    TASK_TIMEOUT: int = Field(default=3600, description="任务超时时间(秒)")

    # 内存管理配置
    MEMORY_LIMIT_GB: float = Field(default=8.0, description="内存限制(GB)")
    MEMORY_WARNING_THRESHOLD: float = Field(default=0.8, description="内存警告阈值")
    MEMORY_CLEANUP_THRESHOLD: float = Field(default=0.9, description="内存清理阈值")
    AUTO_CLEANUP_INTERVAL: int = Field(default=300, description="自动清理间隔(秒)")

    # 图像处理配置
    DEFAULT_IMAGE_SIZE: int = Field(default=640, description="默认图像大小")
    MAX_IMAGE_DIMENSION: int = Field(default=4096, description="最大图像尺寸")
    THUMBNAIL_SIZE: int = Field(default=200, description="缩略图大小")
    JPEG_QUALITY: int = Field(default=90, description="JPEG质量")

    # 抠图模型配置
    MATTING_MODELS: List[str] = Field(
        default=["u2net", "silueta", "rembg", "sam"],
        description="支持的抠图模型"
    )
    DEFAULT_MATTING_MODEL: str = Field(default="u2net", description="默认抠图模型")

    # 合成配置
    DEFAULT_SYNTHESIS_COUNT: int = Field(default=100, description="默认合成数量")
    MAX_SYNTHESIS_COUNT: int = Field(default=10000, description="最大合成数量")
    MAX_OBJECTS_PER_IMAGE: int = Field(default=10, description="单图最大目标数")

    # 数据集格式
    DATASET_FORMATS: List[str] = Field(
        default=["txt", "json", "xml", "coco", "yolo"],
        description="支持的数据集格式"
    )

    # Celery配置
    CELERY_BROKER_URL: Optional[str] = Field(default=None, description="Celery代理URL")
    CELERY_RESULT_BACKEND: Optional[str] = Field(default=None, description="Celery结果后端")
    CELERY_TASK_SERIALIZER: str = Field(default="json", description="任务序列化器")
    CELERY_RESULT_SERIALIZER: str = Field(default="json", description="结果序列化器")
    CELERY_TIMEZONE: str = Field(default="UTC", description="时区")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FILE: str = Field(default="./logs/app.log", description="日志文件")
    LOG_MAX_SIZE: int = Field(default=10, description="日志文件最大大小(MB)")
    LOG_BACKUP_COUNT: int = Field(default=5, description="日志备份数量")

    # 安全配置
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, description="访问令牌过期时间")
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(default=10080, description="刷新令牌过期时间")
    PASSWORD_MIN_LENGTH: int = Field(default=8, description="密码最小长度")

    # 监控配置
    ENABLE_METRICS: bool = Field(default=True, description="启用指标收集")
    METRICS_PORT: int = Field(default=9090, description="指标端口")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="健康检查间隔")

    @validator("CELERY_BROKER_URL", pre=True, always=True)
    def set_celery_broker_url(cls, v, values):
        if v is None:
            return values.get("REDIS_URL")
        return v

    @validator("CELERY_RESULT_BACKEND", pre=True, always=True)
    def set_celery_result_backend(cls, v, values):
        if v is None:
            return values.get("REDIS_URL")
        return v

    @validator("UPLOAD_PATH", "DATASET_PATH", "MODEL_PATH", "CACHE_PATH", "TEMP_PATH")
    def create_directories(cls, v):
        """自动创建目录"""
        os.makedirs(v, exist_ok=True)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()

# 导出常用配置
__all__ = ["settings"]