from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean,
    ForeignKey, JSON, Enum as SQLEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class SynthesisTaskStatus(str, Enum):
    """合成任务状态枚举"""
    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    PAUSED = "paused"  # 已暂停


class SynthesisTaskType(str, Enum):
    """合成任务类型枚举"""
    SINGLE = "single"  # 单张合成
    BATCH = "batch"  # 批量合成
    DATASET = "dataset"  # 数据集生成
    AUGMENTATION = "augmentation"  # 数据增强


class SynthesisTaskPriority(str, Enum):
    """合成任务优先级枚举"""
    LOW = "low"  # 低优先级
    NORMAL = "normal"  # 普通优先级
    HIGH = "high"  # 高优先级
    URGENT = "urgent"  # 紧急优先级


class SynthesisTask(Base):
    """合成任务模型"""
    __tablename__ = "synthesis_tasks"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, comment="任务名称")
    description = Column(Text, nullable=True, comment="任务描述")

    # 任务状态和类型
    status = Column(SQLEnum(SynthesisTaskStatus), default=SynthesisTaskStatus.PENDING, comment="任务状态")
    task_type = Column(SQLEnum(SynthesisTaskType), default=SynthesisTaskType.SINGLE, comment="任务类型")
    priority = Column(SQLEnum(SynthesisTaskPriority), default=SynthesisTaskPriority.NORMAL, comment="任务优先级")

    # 关联字段
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, comment="项目ID")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, comment="用户ID")

    # 任务配置
    config = Column(JSON, nullable=False, default=dict, comment="任务配置参数")

    # 输入数据
    foreground_images = Column(JSON, nullable=True, comment="前景图像列表")
    background_images = Column(JSON, nullable=True, comment="背景图像列表")

    # 输出数据
    result_images = Column(JSON, nullable=True, comment="结果图像列表")
    output_path = Column(String(500), nullable=True, comment="输出路径")

    # 进度信息
    progress = Column(Float, default=0.0, comment="任务进度(0-100)")
    total_images = Column(Integer, default=0, comment="总图像数量")
    processed_images = Column(Integer, default=0, comment="已处理图像数量")
    failed_images = Column(Integer, default=0, comment="失败图像数量")

    # 性能指标
    estimated_duration = Column(Integer, nullable=True, comment="预计耗时(秒)")
    actual_duration = Column(Integer, nullable=True, comment="实际耗时(秒)")
    cpu_usage = Column(Float, nullable=True, comment="CPU使用率")
    memory_usage = Column(Float, nullable=True, comment="内存使用量(MB)")

    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")
    error_details = Column(JSON, nullable=True, comment="错误详情")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    # 其他字段
    is_active = Column(Boolean, default=True, comment="是否激活")
    worker_id = Column(String(255), nullable=True, comment="工作进程ID")
    celery_task_id = Column(String(255), nullable=True, comment="Celery任务ID")

    # 关联关系
    project = relationship("Project", back_populates="synthesis_tasks")
    user = relationship("User", back_populates="synthesis_tasks")

    # 索引
    __table_args__ = (
        Index("idx_synthesis_tasks_project_id", "project_id"),
        Index("idx_synthesis_tasks_user_id", "user_id"),
        Index("idx_synthesis_tasks_status", "status"),
        Index("idx_synthesis_tasks_created_at", "created_at"),
        Index("idx_synthesis_tasks_priority", "priority"),
        Index("idx_synthesis_tasks_celery_task_id", "celery_task_id"),
    )

    @validates('progress')
    def validate_progress(self, key, value):
        """验证进度值"""
        if value is not None:
            if not 0 <= value <= 100:
                raise ValueError("进度值必须在0-100之间")
        return value

    @validates('priority')
    def validate_priority(self, key, value):
        """验证优先级"""
        if value not in [p.value for p in SynthesisTaskPriority]:
            raise ValueError(f"无效的优先级值: {value}")
        return value

    @validates('total_images', 'processed_images', 'failed_images')
    def validate_image_counts(self, key, value):
        """验证图像数量"""
        if value is not None and value < 0:
            raise ValueError(f"{key} 不能为负数")
        return value

    @hybrid_property
    def is_completed(self):
        """是否已完成"""
        return self.status == SynthesisTaskStatus.COMPLETED

    @hybrid_property
    def is_failed(self):
        """是否失败"""
        return self.status == SynthesisTaskStatus.FAILED

    @hybrid_property
    def is_processing(self):
        """是否正在处理"""
        return self.status == SynthesisTaskStatus.PROCESSING

    @hybrid_property
    def is_pending(self):
        """是否待处理"""
        return self.status == SynthesisTaskStatus.PENDING

    @hybrid_property
    def success_rate(self):
        """成功率计算"""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images - self.failed_images) / self.total_images * 100

    @hybrid_property
    def remaining_images(self):
        """剩余图像数量"""
        return max(0, self.total_images - self.processed_images)

    @hybrid_property
    def duration_seconds(self):
        """任务持续时间(秒)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return 0

    def update_progress(self, processed_count: int, failed_count: int = 0):
        """更新任务进度"""
        self.processed_images = processed_count
        self.failed_images = failed_count

        if self.total_images > 0:
            self.progress = (processed_count / self.total_images) * 100

        # 自动更新状态
        if processed_count >= self.total_images:
            self.status = SynthesisTaskStatus.COMPLETED
            self.completed_at = datetime.utcnow()

    def start_task(self, worker_id: str = None, celery_task_id: str = None):
        """开始任务"""
        self.status = SynthesisTaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        if worker_id:
            self.worker_id = worker_id
        if celery_task_id:
            self.celery_task_id = celery_task_id

    def complete_task(self, result_images: List[str] = None, output_path: str = None):
        """完成任务"""
        self.status = SynthesisTaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress = 100.0

        if result_images:
            self.result_images = result_images
        if output_path:
            self.output_path = output_path

        # 计算实际耗时
        if self.started_at:
            self.actual_duration = int(self.duration_seconds)

    def fail_task(self, error_message: str, error_details: Dict[str, Any] = None):
        """任务失败"""
        self.status = SynthesisTaskStatus.FAILED
        self.error_message = error_message
        self.error_details = error_details or {}

        # 记录完成时间
        if not self.completed_at:
            self.completed_at = datetime.utcnow()

    def cancel_task(self):
        """取消任务"""
        self.status = SynthesisTaskStatus.CANCELLED
        if not self.completed_at:
            self.completed_at = datetime.utcnow()

    def pause_task(self):
        """暂停任务"""
        if self.status == SynthesisTaskStatus.PROCESSING:
            self.status = SynthesisTaskStatus.PAUSED

    def resume_task(self):
        """恢复任务"""
        if self.status == SynthesisTaskStatus.PAUSED:
            self.status = SynthesisTaskStatus.PROCESSING

    def get_config_value(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default) if self.config else default

    def set_config_value(self, key: str, value: Any):
        """设置配置值"""
        if not self.config:
            self.config = {}
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'project_id': str(self.project_id),
            'user_id': str(self.user_id),
            'config': self.config,
            'foreground_images': self.foreground_images,
            'background_images': self.background_images,
            'result_images': self.result_images,
            'output_path': self.output_path,
            'progress': self.progress,
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'failed_images': self.failed_images,
            'success_rate': self.success_rate,
            'remaining_images': self.remaining_images,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'duration_seconds': self.duration_seconds,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'is_active': self.is_active,
            'worker_id': self.worker_id,
            'celery_task_id': self.celery_task_id,
        }

    def __repr__(self):
        return f"<SynthesisTask(id={self.id}, name='{self.name}', status='{self.status}', progress={self.progress}%)>"