# backend/app/models/dataset.py
"""
数据集数据模型
定义数据集相关的数据库模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.models.base import BaseModel


class DatasetType(str, Enum):
    """数据集类型枚举"""
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    CUSTOM = "custom"


class DatasetStatus(str, Enum):
    """数据集状态枚举"""
    CREATING = "creating"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    EXPORTING = "exporting"
    ARCHIVED = "archived"


class AnnotationFormat(str, Enum):
    """标注格式枚举"""
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    LABELME = "labelme"
    CVAT = "cvat"
    CUSTOM_JSON = "custom_json"
    CUSTOM_XML = "custom_xml"
    CUSTOM_TXT = "custom_txt"


class Dataset(BaseModel):
    """数据集模型"""

    __tablename__ = "datasets"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(200), nullable=False, index=True, comment="数据集名称")
    description = Column(Text, nullable=True, comment="数据集描述")

    # 关联字段
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True, comment="创建者ID")
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True, comment="所属项目ID")

    # 数据集配置
    dataset_type = Column(String(50), nullable=False, default=DatasetType.OBJECT_DETECTION, comment="数据集类型")
    annotation_format = Column(String(50), nullable=False, default=AnnotationFormat.YOLO, comment="标注格式")

    # 状态信息
    status = Column(String(20), nullable=False, default=DatasetStatus.CREATING, comment="数据集状态")
    progress = Column(Float, default=0.0, comment="处理进度(0-100)")
    error_message = Column(Text, nullable=True, comment="错误信息")

    # 数据集配置
    config = Column(JSON, default=dict, comment="数据集配置")
    class_names = Column(JSON, default=list, comment="类别名称列表")
    class_mapping = Column(JSON, default=dict, comment="类别映射")

    # 数据统计
    total_images = Column(Integer, default=0, comment="总图像数量")
    total_annotations = Column(Integer, default=0, comment="总标注数量")
    train_images = Column(Integer, default=0, comment="训练集图像数量")
    val_images = Column(Integer, default=0, comment="验证集图像数量")
    test_images = Column(Integer, default=0, comment="测试集图像数量")

    # 分割比例
    train_ratio = Column(Float, default=0.8, comment="训练集比例")
    val_ratio = Column(Float, default=0.15, comment="验证集比例")
    test_ratio = Column(Float, default=0.05, comment="测试集比例")

    # 数据增强配置
    augmentation_config = Column(JSON, default=dict, comment="数据增强配置")
    augmentation_enabled = Column(Boolean, default=False, comment="是否启用数据增强")

    # 文件信息
    file_path = Column(String(500), nullable=True, comment="数据集文件路径")
    file_size = Column(Float, default=0.0, comment="文件大小(MB)")
    file_format = Column(String(20), nullable=True, comment="文件格式")

    # 版本信息
    version = Column(String(20), default="1.0.0", comment="版本号")
    is_public = Column(Boolean, default=False, comment="是否公开")

    # 导出信息
    exported_at = Column(DateTime, nullable=True, comment="导出时间")
    export_count = Column(Integer, default=0, comment="导出次数")
    download_count = Column(Integer, default=0, comment="下载次数")

    # 关联关系
    user = relationship("User", back_populates="datasets")
    project = relationship("Project", back_populates="datasets")
    dataset_images = relationship("DatasetImage", back_populates="dataset", cascade="all, delete-orphan")
    dataset_annotations = relationship("DatasetAnnotation", back_populates="dataset", cascade="all, delete-orphan")
    dataset_exports = relationship("DatasetExport", back_populates="dataset", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_dataset_user_id', 'user_id'),
        Index('idx_dataset_project_id', 'project_id'),
        Index('idx_dataset_status', 'status'),
        Index('idx_dataset_type', 'dataset_type'),
        Index('idx_dataset_public', 'is_public'),
        Index('idx_dataset_created', 'created_at'),
    )

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', status='{self.status}')>"

    def get_split_info(self) -> Dict[str, Any]:
        """获取数据集分割信息"""
        return {
            'train': {
                'count': self.train_images,
                'ratio': self.train_ratio
            },
            'val': {
                'count': self.val_images,
                'ratio': self.val_ratio
            },
            'test': {
                'count': self.test_images,
                'ratio': self.test_ratio
            },
            'total': self.total_images
        }

    def get_class_info(self) -> Dict[str, Any]:
        """获取类别信息"""
        return {
            'names': self.class_names,
            'mapping': self.class_mapping,
            'count': len(self.class_names) if self.class_names else 0
        }

    def update_statistics(self):
        """更新统计信息"""
        # 这里可以添加统计逻辑
        pass

    def set_status(self, status: DatasetStatus, error_message: str = None):
        """设置状态"""
        self.status = status
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.utcnow()

    def update_progress(self, progress: float):
        """更新进度"""
        self.progress = max(0, min(100, progress))
        self.updated_at = datetime.utcnow()

    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return self.status == DatasetStatus.READY

    def get_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'dataset_type': self.dataset_type,
            'annotation_format': self.annotation_format,
            'status': self.status,
            'progress': self.progress,
            'config': self.config,
            'class_info': self.get_class_info(),
            'split_info': self.get_split_info(),
            'total_images': self.total_images,
            'total_annotations': self.total_annotations,
            'file_size': self.file_size,
            'version': self.version,
            'is_public': self.is_public,
            'augmentation_enabled': self.augmentation_enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'exported_at': self.exported_at.isoformat() if self.exported_at else None
        }


class DatasetImage(BaseModel):
    """数据集图像模型"""

    __tablename__ = "dataset_images"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True, comment="数据集ID")
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, index=True, comment="图像ID")

    # 图像信息
    filename = Column(String(255), nullable=False, comment="文件名")
    file_path = Column(String(500), nullable=False, comment="文件路径")
    file_size = Column(Float, nullable=False, comment="文件大小(MB)")

    # 图像属性
    width = Column(Integer, nullable=False, comment="图像宽度")
    height = Column(Integer, nullable=False, comment="图像高度")
    channels = Column(Integer, default=3, comment="图像通道数")

    # 数据集分割
    split_type = Column(String(10), nullable=False, default="train", comment="数据集分割类型")

    # 标注状态
    is_annotated = Column(Boolean, default=False, comment="是否已标注")
    annotation_count = Column(Integer, default=0, comment="标注数量")

    # 质量信息
    quality_score = Column(Float, nullable=True, comment="质量评分")
    is_valid = Column(Boolean, default=True, comment="是否有效")

    # 关联关系
    dataset = relationship("Dataset", back_populates="dataset_images")
    image = relationship("Image", back_populates="dataset_images")
    annotations = relationship("DatasetAnnotation", back_populates="dataset_image", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_dataset_image_dataset_id', 'dataset_id'),
        Index('idx_dataset_image_image_id', 'image_id'),
        Index('idx_dataset_image_split', 'split_type'),
        Index('idx_dataset_image_annotated', 'is_annotated'),
    )

    def __repr__(self):
        return f"<DatasetImage(id={self.id}, filename='{self.filename}', split='{self.split_type}')>"

    def get_info(self) -> Dict[str, Any]:
        """获取图像信息"""
        return {
            'id': str(self.id),
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'split_type': self.split_type,
            'is_annotated': self.is_annotated,
            'annotation_count': self.annotation_count,
            'quality_score': self.quality_score,
            'is_valid': self.is_valid,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DatasetAnnotation(BaseModel):
    """数据集标注模型"""

    __tablename__ = "dataset_annotations"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True, comment="数据集ID")
    dataset_image_id = Column(UUID(as_uuid=True), ForeignKey("dataset_images.id"), nullable=False, index=True,
                              comment="数据集图像ID")

    # 标注信息
    class_id = Column(Integer, nullable=False, comment="类别ID")
    class_name = Column(String(100), nullable=False, comment="类别名称")

    # 标注数据
    annotation_data = Column(JSON, nullable=False, comment="标注数据")
    annotation_format = Column(String(50), nullable=False, comment="标注格式")

    # 边界框信息 (适用于目标检测)
    bbox_x = Column(Float, nullable=True, comment="边界框X坐标")
    bbox_y = Column(Float, nullable=True, comment="边界框Y坐标")
    bbox_width = Column(Float, nullable=True, comment="边界框宽度")
    bbox_height = Column(Float, nullable=True, comment="边界框高度")

    # 分割信息 (适用于实例分割)
    segmentation = Column(JSON, nullable=True, comment="分割数据")
    area = Column(Float, nullable=True, comment="面积")

    # 关键点信息 (适用于关键点检测)
    keypoints = Column(JSON, nullable=True, comment="关键点数据")
    num_keypoints = Column(Integer, nullable=True, comment="关键点数量")

    # 标注属性
    is_crowd = Column(Boolean, default=False, comment="是否为群体")
    difficulty = Column(Integer, default=0, comment="难度等级")
    confidence = Column(Float, default=1.0, comment="置信度")

    # 关联关系
    dataset = relationship("Dataset", back_populates="dataset_annotations")
    dataset_image = relationship("DatasetImage", back_populates="annotations")

    # 索引
    __table_args__ = (
        Index('idx_dataset_annotation_dataset_id', 'dataset_id'),
        Index('idx_dataset_annotation_image_id', 'dataset_image_id'),
        Index('idx_dataset_annotation_class', 'class_id'),
        Index('idx_dataset_annotation_format', 'annotation_format'),
    )

    def __repr__(self):
        return f"<DatasetAnnotation(id={self.id}, class_name='{self.class_name}')>"

    def get_bbox_info(self) -> Dict[str, Any]:
        """获取边界框信息"""
        if all(v is not None for v in [self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height]):
            return {
                'x': self.bbox_x,
                'y': self.bbox_y,
                'width': self.bbox_width,
                'height': self.bbox_height,
                'x_center': self.bbox_x + self.bbox_width / 2,
                'y_center': self.bbox_y + self.bbox_height / 2
            }
        return None

    def get_info(self) -> Dict[str, Any]:
        """获取标注信息"""
        return {
            'id': str(self.id),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'annotation_data': self.annotation_data,
            'annotation_format': self.annotation_format,
            'bbox_info': self.get_bbox_info(),
            'segmentation': self.segmentation,
            'area': self.area,
            'keypoints': self.keypoints,
            'num_keypoints': self.num_keypoints,
            'is_crowd': self.is_crowd,
            'difficulty': self.difficulty,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DatasetExport(BaseModel):
    """数据集导出模型"""

    __tablename__ = "dataset_exports"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True, comment="数据集ID")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True, comment="导出用户ID")

    # 导出配置
    export_format = Column(String(50), nullable=False, comment="导出格式")
    export_config = Column(JSON, default=dict, comment="导出配置")

    # 导出范围
    include_train = Column(Boolean, default=True, comment="包含训练集")
    include_val = Column(Boolean, default=True, comment="包含验证集")
    include_test = Column(Boolean, default=True, comment="包含测试集")

    # 导出状态
    status = Column(String(20), nullable=False, default="pending", comment="导出状态")
    progress = Column(Float, default=0.0, comment="导出进度")
    error_message = Column(Text, nullable=True, comment="错误信息")

    # 文件信息
    file_path = Column(String(500), nullable=True, comment="导出文件路径")
    file_size = Column(Float, nullable=True, comment="文件大小(MB)")
    file_hash = Column(String(64), nullable=True, comment="文件哈希")

    # 导出统计
    exported_images = Column(Integer, default=0, comment="导出图像数量")
    exported_annotations = Column(Integer, default=0, comment="导出标注数量")

    # 过期信息
    expires_at = Column(DateTime, nullable=True, comment="过期时间")
    download_count = Column(Integer, default=0, comment="下载次数")

    # 关联关系
    dataset = relationship("Dataset", back_populates="dataset_exports")
    user = relationship("User", back_populates="dataset_exports")

    # 索引
    __table_args__ = (
        Index('idx_dataset_export_dataset_id', 'dataset_id'),
        Index('idx_dataset_export_user_id', 'user_id'),
        Index('idx_dataset_export_status', 'status'),
        Index('idx_dataset_export_expires', 'expires_at'),
    )

    def __repr__(self):
        return f"<DatasetExport(id={self.id}, format='{self.export_format}', status='{self.status}')>"

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return self.status == "ready" and not self.is_expired()

    def update_download_count(self):
        """更新下载次数"""
        self.download_count += 1
        self.updated_at = datetime.utcnow()

    def get_info(self) -> Dict[str, Any]:
        """获取导出信息"""
        return {
            'id': str(self.id),
            'export_format': self.export_format,
            'export_config': self.export_config,
            'include_train': self.include_train,
            'include_val': self.include_val,
            'include_test': self.include_test,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'exported_images': self.exported_images,
            'exported_annotations': self.exported_annotations,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'download_count': self.download_count,
            'is_expired': self.is_expired(),
            'is_ready': self.is_ready(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }