# backend/app/models/user.py
"""
用户数据模型
定义用户相关的数据库模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from passlib.context import CryptContext
import uuid

from app.models.base import BaseModel

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """用户模型"""

    __tablename__ = "users"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    email = Column(String(100), unique=True, nullable=False, index=True, comment="邮箱")
    phone = Column(String(20), nullable=True, comment="手机号")

    # 认证信息
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_verified = Column(Boolean, default=False, comment="是否验证")
    is_superuser = Column(Boolean, default=False, comment="是否超级用户")

    # 个人信息
    full_name = Column(String(100), nullable=True, comment="真实姓名")
    nickname = Column(String(50), nullable=True, comment="昵称")
    avatar = Column(String(500), nullable=True, comment="头像URL")
    bio = Column(Text, nullable=True, comment="个人简介")

    # 配置信息
    preferences = Column(JSON, default=dict, comment="用户偏好设置")
    settings = Column(JSON, default=dict, comment="用户设置")

    # 统计信息
    project_count = Column(Integer, default=0, comment="项目数量")
    storage_used = Column(Float, default=0.0, comment="存储使用量(MB)")
    last_login_at = Column(DateTime, nullable=True, comment="最后登录时间")
    login_count = Column(Integer, default=0, comment="登录次数")

    # 订阅信息
    subscription_type = Column(String(50), default="free", comment="订阅类型")
    subscription_expires_at = Column(DateTime, nullable=True, comment="订阅过期时间")

    # 关联关系
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    user_sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_subscription', 'subscription_type'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

    def set_password(self, password: str):
        """设置密码"""
        self.password_hash = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(password, self.password_hash)

    def update_login_info(self):
        """更新登录信息"""
        self.last_login_at = datetime.utcnow()
        self.login_count += 1

    def is_subscription_valid(self) -> bool:
        """检查订阅是否有效"""
        if self.subscription_type == "free":
            return True
        if self.subscription_expires_at is None:
            return False
        return datetime.utcnow() < self.subscription_expires_at

    def get_preference(self, key: str, default=None):
        """获取偏好设置"""
        return self.preferences.get(key, default) if self.preferences else default

    def set_preference(self, key: str, value):
        """设置偏好"""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value

    def get_setting(self, key: str, default=None):
        """获取设置"""
        return self.settings.get(key, default) if self.settings else default

    def set_setting(self, key: str, value):
        """设置设置"""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value

    def get_storage_limit(self) -> float:
        """获取存储限制(MB)"""
        limits = {
            "free": 1024,  # 1GB
            "basic": 10 * 1024,  # 10GB
            "pro": 100 * 1024,  # 100GB
            "enterprise": float('inf')
        }
        return limits.get(self.subscription_type, 1024)

    def is_storage_limit_exceeded(self) -> bool:
        """检查是否超过存储限制"""
        return self.storage_used > self.get_storage_limit()

    def get_public_info(self) -> Dict[str, Any]:
        """获取公开信息"""
        return {
            'id': str(self.id),
            'username': self.username,
            'nickname': self.nickname,
            'avatar': self.avatar,
            'bio': self.bio,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def get_profile_info(self) -> Dict[str, Any]:
        """获取个人资料信息"""
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'full_name': self.full_name,
            'nickname': self.nickname,
            'avatar': self.avatar,
            'bio': self.bio,
            'is_verified': self.is_verified,
            'subscription_type': self.subscription_type,
            'subscription_expires_at': self.subscription_expires_at.isoformat() if self.subscription_expires_at else None,
            'project_count': self.project_count,
            'storage_used': self.storage_used,
            'storage_limit': self.get_storage_limit(),
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'login_count': self.login_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'preferences': self.preferences,
            'settings': self.settings,
            'is_subscription_valid': self.is_subscription_valid(),
            'is_storage_limit_exceeded': self.is_storage_limit_exceeded()
        }


class UserSession(BaseModel):
    """用户会话模型"""

    __tablename__ = "user_sessions"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True, comment="用户ID")
    session_token = Column(String(255), unique=True, nullable=False, index=True, comment="会话令牌")

    # 会话信息
    device_info = Column(JSON, default=dict, comment="设备信息")
    ip_address = Column(String(45), nullable=True, comment="IP地址")
    user_agent = Column(Text, nullable=True, comment="用户代理")
    location = Column(String(100), nullable=True, comment="地理位置")

    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否激活")
    last_activity_at = Column(DateTime, nullable=True, comment="最后活动时间")
    expires_at = Column(DateTime, nullable=False, comment="过期时间")

    # 关联关系
    user = relationship("User", back_populates="user_sessions")

    # 索引
    __table_args__ = (
        Index('idx_session_user_id', 'user_id'),
        Index('idx_session_token', 'session_token'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_active', 'is_active'),
    )

    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, is_active={self.is_active})>"

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.utcnow() > self.expires_at

    def update_activity(self):
        """更新活动时间"""
        self.last_activity_at = datetime.utcnow()

    def revoke(self):
        """撤销会话"""
        self.is_active = False
        self.updated_at = datetime.utcnow()


class ApiKey(BaseModel):
    """API密钥模型"""

    __tablename__ = "api_keys"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True, comment="用户ID")
    key_hash = Column(String(255), unique=True, nullable=False, index=True, comment="密钥哈希")
    key_prefix = Column(String(10), nullable=False, comment="密钥前缀")

    # 密钥信息
    name = Column(String(100), nullable=False, comment="密钥名称")
    description = Column(Text, nullable=True, comment="密钥描述")
    permissions = Column(JSON, default=list, comment="权限列表")

    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否激活")
    last_used_at = Column(DateTime, nullable=True, comment="最后使用时间")
    usage_count = Column(Integer, default=0, comment="使用次数")
    expires_at = Column(DateTime, nullable=True, comment="过期时间")

    # 关联关系
    user = relationship("User", back_populates="api_keys")

    # 索引
    __table_args__ = (
        Index('idx_apikey_user_id', 'user_id'),
        Index('idx_apikey_hash', 'key_hash'),
        Index('idx_apikey_active', 'is_active'),
        Index('idx_apikey_expires', 'expires_at'),
    )

    def __repr__(self):
        return f"<ApiKey(id={self.id}, user_id={self.user_id}, name='{self.name}')>"

    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def update_usage(self):
        """更新使用信息"""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1

    def revoke(self):
        """撤销密钥"""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions if self.permissions else False

    def get_info(self) -> Dict[str, Any]:
        """获取密钥信息"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'key_prefix': self.key_prefix,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'usage_count': self.usage_count,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_expired': self.is_expired()
        }