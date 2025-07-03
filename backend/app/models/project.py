# backend/app/models/project.py
"""
项目数据模型
定义项目相关的数据库模型
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.models.base import BaseModel


class Project(BaseModel):
    """项目模型"""

    __tablename__ = "projects"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, index=True, comment="项目名称")
    description = Column(Text, nullable=True, comment="项目描述")

    # 状态字段
    status = Column(String(50), default="active", comment="项目状态")  # active, archived, deleted
    is_active = Column(Boolean, default=True, comment="是否激活")

    # 配置信息
    config = Column(JSON, default=dict, comment="项目配置")
    settings = Column(JSON, default=dict, comment="项目设置")

    # 统计信息
    foreground_count = Column(Integer, default=0, comment="前景图数量")
    background_count = Column(Integer, default=0, comment="背景图数量")
    synthesis_count = Column(Integer, default=0, comment="合成图数量")
    dataset_count = Column(Integer, default=0, comment="数据集数量")

    # 存储信息
    storage_path = Column(String(500), nullable=True, comment="存储路径")
    total_size = Column(Float, default=0.0, comment="总大小(MB)")

    # 时间字段
    last_processed_at = Column(DateTime, nullable=True, comment="最后处理时间")

    # 关联关系
    images = relationship("Image", back_populates="project", cascade="all, delete-orphan")
    synthesis_tasks = relationship("SynthesisTask", back_populates="project", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"

    @property
    def is_empty(self) -> bool:
        """检查项目是否为空"""
        return (self.foreground_count == 0 and
                self.background_count == 0 and
                self.synthesis_count == 0)

    def update_statistics(self):
        """更新统计信息"""
        from app.models.image import Image
        from app.models.synthesis_task import SynthesisTask
        from app.models.dataset import Dataset

        # 这里应该通过数据库查询来更新统计
        # 为了简化，这里只是示例结构
        pass

    def get_config_value(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default) if self.config else default

    def set_config_value(self, key: str, value):
        """设置配置值"""
        if not self.config:
            self.config = {}
        self.config[key] = value

    def get_setting_value(self, key: str, default=None):
        """获取设置值"""
        return self.settings.get(key, default) if self.settings else default

    def set_setting_value(self, key: str, value):
        """设置设置值"""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value


class ProjectTemplate(BaseModel):
    """项目模板模型"""

    __tablename__ = "project_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, comment="模板名称")
    description = Column(Text, nullable=True, comment="模板描述")

    # 模板配置
    template_config = Column(JSON, nullable=False, comment="模板配置")
    default_settings = Column(JSON, default=dict, comment="默认设置")

    # 分类信息
    category = Column(String(100), nullable=True, comment="模板分类")
    tags = Column(JSON, default=list, comment="标签列表")

    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")

    # 模板状态
    is_public = Column(Boolean, default=False, comment="是否公开")
    is_active = Column(Boolean, default=True, comment="是否激活")

    def __repr__(self):
        return f"<ProjectTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"


class ProjectShare(BaseModel):
    """项目分享模型"""

    __tablename__ = "project_shares"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False, comment="项目ID")

    # 分享信息
    share_token = Column(String(255), unique=True, nullable=False, comment="分享令牌")
    share_type = Column(String(50), default="read", comment="分享类型")  # read, write, admin

    # 访问控制
    password = Column(String(255), nullable=True, comment="访问密码")
    max_access_count = Column(Integer, nullable=True, comment="最大访问次数")
    current_access_count = Column(Integer, default=0, comment="当前访问次数")

    # 有效期
    expires_at = Column(DateTime, nullable=True, comment="过期时间")

    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")

    # 关联关系
    project = relationship("Project")

    def __repr__(self):
        return f"<ProjectShare(id={self.id}, project_id={self.project_id}, share_type='{self.share_type}')>"

    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_access_limited(self) -> bool:
        """检查是否达到访问限制"""
        if self.max_access_count is None:
            return False
        return self.current_access_count >= self.max_access_count

    @property
    def can_access(self) -> bool:
        """检查是否可以访问"""
        return (self.is_active and
                not self.is_expired and
                not self.is_access_limited)


class ProjectActivity(BaseModel):
    """项目活动记录模型"""

    __tablename__ = "project_activities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False, comment="项目ID")

    # 活动信息
    activity_type = Column(String(100), nullable=False, comment="活动类型")
    activity_name = Column(String(255), nullable=False, comment="活动名称")
    description = Column(Text, nullable=True, comment="活动描述")

    # 活动数据
    metadata = Column(JSON, default=dict, comment="活动元数据")
    result = Column(JSON, default=dict, comment="活动结果")

    # 状态信息
    status = Column(String(50), default="completed", comment="活动状态")  # started, completed, failed, cancelled

    # 时间信息
    started_at = Column(DateTime, default=datetime.utcnow, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    duration = Column(Float, nullable=True, comment="持续时间(秒)")

    # 关联关系
    project = relationship("Project")

    def __repr__(self):
        return f"<ProjectActivity(id={self.id}, project_id={self.project_id}, type='{self.activity_type}')>"

    def mark_completed(self, result_data: dict = None):
        """标记为完成"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        if result_data:
            self.result = result_data

    def mark_failed(self, error_message: str = None):
        """标记为失败"""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        if error_message:
            self.result = {"error": error_message}