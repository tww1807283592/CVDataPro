# backend/app/models/base.py
"""
基础数据模型
定义所有模型的基类
"""

from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, DateTime, Boolean, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

# 创建基础模型类
Base = declarative_base()


class BaseModel(Base):
    """基础模型类"""

    __abstract__ = True

    # 公共字段
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, comment="更新时间")
    is_deleted = Column(Boolean, default=False, nullable=False, comment="是否删除")

    def __init__(self, **kwargs):
        """初始化模型"""
        super().__init__(**kwargs)

    def save(self, db: Session) -> 'BaseModel':
        """保存模型到数据库"""
        try:
            db.add(self)
            db.commit()
            db.refresh(self)
            return self
        except Exception as e:
            db.rollback()
            raise e

    def delete(self, db: Session, soft: bool = True) -> bool:
        """删除模型"""
        try:
            if soft:
                # 软删除
                self.is_deleted = True
                self.updated_at = datetime.utcnow()
                db.commit()
            else:
                # 硬删除
                db.delete(self)
                db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e

    def update(self, db: Session, **kwargs) -> 'BaseModel':
        """更新模型"""
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(self)
            return self
        except Exception as e:
            db.rollback()
            raise e

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            result[column.name] = value
        return result

    def from_dict(self, data: Dict[str, Any]) -> 'BaseModel':
        """从字典创建模型"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @classmethod
    def get_by_id(cls, db: Session, id: Any, include_deleted: bool = False):
        """根据ID获取模型"""
        query = db.query(cls).filter(cls.id == id)
        if not include_deleted:
            query = query.filter(cls.is_deleted == False)
        return query.first()

    @classmethod
    def get_all(cls, db: Session, include_deleted: bool = False, limit: int = None, offset: int = None):
        """获取所有模型"""
        query = db.query(cls)
        if not include_deleted:
            query = query.filter(cls.is_deleted == False)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()

    @classmethod
    def count(cls, db: Session, include_deleted: bool = False) -> int:
        """获取模型数量"""
        query = db.query(cls)
        if not include_deleted:
            query = query.filter(cls.is_deleted == False)
        return query.count()

    @classmethod
    def exists(cls, db: Session, **kwargs) -> bool:
        """检查模型是否存在"""
        query = db.query(cls)
        for key, value in kwargs.items():
            if hasattr(cls, key):
                query = query.filter(getattr(cls, key) == value)
        query = query.filter(cls.is_deleted == False)
        return query.first() is not None

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={getattr(self, 'id', 'None')})>"


# 模型事件监听器
@event.listens_for(BaseModel, 'before_update', propagate=True)
def receive_before_update(mapper, connection, target):
    """更新前事件监听器"""
    target.updated_at = datetime.utcnow()


@event.listens_for(BaseModel, 'before_insert', propagate=True)
def receive_before_insert(mapper, connection, target):
    """插入前事件监听器"""
    now = datetime.utcnow()
    target.created_at = now
    target.updated_at = now


class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class SoftDeleteMixin:
    """软删除混入类"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    def soft_delete(self):
        """软删除"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

    def restore(self):
        """恢复"""
        self.is_deleted = False
        self.deleted_at = None


class UUIDMixin:
    """UUID混入类"""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)


class MetadataMixin:
    """元数据混入类"""
    from sqlalchemy import JSON

    metadata = Column(JSON, default=dict, comment="元数据")

    def get_metadata(self, key: str, default=None):
        """获取元数据"""
        return self.metadata.get(key, default) if self.metadata else default

    def set_metadata(self, key: str, value):
        """设置元数据"""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value

    def update_metadata(self, data: dict):
        """更新元数据"""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(data)

    def remove_metadata(self, key: str):
        """移除元数据"""
        if self.metadata and key in self.metadata:
            del self.metadata[key]