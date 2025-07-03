# backend/app/models/image.py
"""
图像数据模型
定义图像相关的数据库模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.models.base import BaseModel


class Image(BaseModel):
    """图像模型"""

    __tablename__ = "images"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False, comment="项目ID")

    # 文件信息
    filename = Column(String(255), nullable=False, comment="文件名")
    original_filename = Column(String(255), nullable=False, comment="原始文件名")
    file_path = Column(String(500), nullable=False, comment="文件路径")
    file_size = Column(Float, nullable=False, comment="文件大小(MB)")
    file_format = Column(String(20), nullable=False, comment="文件格式")
    mime_type = Column(String(100), nullable=False, comment="MIME类型")

    # 图像属性
    width = Column(Integer, nullable=False, comment="宽度")
    height = Column(Integer, nullable=False, comment="高度")
    channels = Column(Integer, default=3, comment="通道数")
    color_mode = Column(String(20), default="RGB", comment="颜色模式")

    # 图像类型
    image_type = Column(String(50), nullable=False, comment="图像类型")  # foreground, background, synthesized
    category = Column(String(100), nullable=True, comment="图像分类")

    # 处理状态
    status = Column(String(50), default="uploaded", comment="状态")  # uploaded, processing, processed, failed
    processing_progress = Column(Float, default=0.0, comment="处理进度")

    # 质量评估
    quality_score = Column(Float, nullable=True, comment="质量评分")
    blur_score = Column(Float, nullable=True, comment="模糊度评分")
    brightness = Column(Float, nullable=True, comment="亮度")
    contrast = Column(Float, nullable=True, comment="对比度")

    # 元数据
    metadata = Column(JSON, default=dict, comment="图像元数据")
    exif_data = Column(JSON, default=dict, comment="EXIF数据")

    # 处理信息
    processed_versions = Column(JSON, default=list, comment="处理版本列表")
    thumbnail_path = Column(String(500), nullable=True, comment="缩略图路径")

    # 标注信息
    annotations = Column(JSON, default=list, comment="标注信息")
    labels = Column(JSON, default=list, comment="标签列表")

    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    synthesis_count = Column(Integer, default=0, comment="参与合成次数")

    # 时间字段
    uploaded_at = Column(DateTime, default=datetime.utcnow, comment="上传时间")
    processed_at = Column(DateTime, nullable=True, comment="处理时间")
    last_used_at = Column(DateTime, nullable=True, comment="最后使用时间")

    # 关联关系
    project = relationship("Project", back_populates="images")
    matting_results = relationship("MattingResult", back_populates="source_image", cascade="all, delete-orphan")
    synthesis_tasks = relationship("SynthesisTask", back_populates="images", secondary="synthesis_task_images")

    # 索引
    __table_args__ = (
        Index('idx_image_project_type', 'project_id', 'image_type'),
        Index('idx_image_status', 'status'),
        Index('idx_image_uploaded_at', 'uploaded_at'),
        Index('idx_image_category', 'category'),
        Index('idx_image_quality', 'quality_score'),
    )

    def __repr__(self):
        return f"<Image(id={self.id}, filename='{self.filename}', type='{self.image_type}')>"

    @property
    def is_foreground(self) -> bool:
        """是否为前景图"""
        return self.image_type == "foreground"

    @property
    def is_background(self) -> bool:
        """是否为背景图"""
        return self.image_type == "background"

    @property
    def is_synthesized(self) -> bool:
        """是否为合成图"""
        return self.image_type == "synthesized"

    @property
    def aspect_ratio(self) -> float:
        """宽高比"""
        return self.width / self.height if self.height > 0 else 0

    @property
    def resolution(self) -> str:
        """分辨率字符串"""
        return f"{self.width}x{self.height}"

    @property
    def file_size_mb(self) -> float:
        """文件大小(MB)，保留2位小数"""
        return round(self.file_size, 2)

    def get_metadata_value(self, key: str, default=None):
        """获取元数据值"""
        return self.metadata.get(key, default) if self.metadata else default

    def set_metadata_value(self, key: str, value: Any):
        """设置元数据值"""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value

    def update_metadata(self, metadata_dict: Dict[str, Any]):
        """更新元数据"""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(metadata_dict)

    def add_annotation(self, annotation: Dict[str, Any]):
        """添加标注"""
        if not self.annotations:
            self.annotations = []
        self.annotations.append(annotation)

    def add_label(self, label: str):
        """添加标签"""
        if not self.labels:
            self.labels = []
        if label not in self.labels:
            self.labels.append(label)

    def remove_label(self, label: str):
        """移除标签"""
        if self.labels and label in self.labels:
            self.labels.remove(label)

    def add_processed_version(self, version_info: Dict[str, Any]):
        """添加处理版本"""
        if not self.processed_versions:
            self.processed_versions = []
        version_info['created_at'] = datetime.utcnow().isoformat()
        self.processed_versions.append(version_info)

    def get_latest_processed_version(self) -> Optional[Dict[str, Any]]:
        """获取最新处理版本"""
        if not self.processed_versions:
            return None
        return max(self.processed_versions, key=lambda x: x.get('created_at', ''))

    def update_usage_stats(self):
        """更新使用统计"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()

    def update_synthesis_stats(self):
        """更新合成统计"""
        self.synthesis_count += 1
        self.last_used_at = datetime.utcnow()

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """判断是否为高质量图像"""
        return self.quality_score is not None and self.quality_score >= threshold

    def is_suitable_for_synthesis(self) -> bool:
        """判断是否适合用于合成"""
        if self.status != "processed":
            return False
        if self.quality_score is not None and self.quality_score < 0.5:
            return False
        if self.blur_score is not None and self.blur_score < 0.6:
            return False
        return True

    def get_display_info(self) -> Dict[str, Any]:
        """获取显示信息"""
        return {
            'id': str(self.id),
            'filename': self.filename,
            'original_filename': self.original_filename,
            'image_type': self.image_type,
            'category': self.category,
            'resolution': self.resolution,
            'file_size': self.file_size_mb,
            'status': self.status,
            'quality_score': self.quality_score,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'usage_count': self.usage_count,
            'synthesis_count': self.synthesis_count,
            'labels': self.labels or [],
            'thumbnail_path': self.thumbnail_path
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': str(self.id),
            'project_id': str(self.project_id),
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_format': self.file_format,
            'mime_type': self.mime_type,
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'color_mode': self.color_mode,
            'image_type': self.image_type,
            'category': self.category,
            'status': self.status,
            'processing_progress': self.processing_progress,
            'quality_score': self.quality_score,
            'blur_score': self.blur_score,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'metadata': self.metadata,
            'exif_data': self.exif_data,
            'processed_versions': self.processed_versions,
            'thumbnail_path': self.thumbnail_path,
            'annotations': self.annotations,
            'labels': self.labels,
            'usage_count': self.usage_count,
            'synthesis_count': self.synthesis_count,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def create_from_upload(cls, project_id: str, file_info: Dict[str, Any], image_type: str = "foreground"):
        """从上传信息创建图像记录"""
        return cls(
            project_id=project_id,
            filename=file_info['filename'],
            original_filename=file_info['original_filename'],
            file_path=file_info['file_path'],
            file_size=file_info['file_size'],
            file_format=file_info['file_format'],
            mime_type=file_info['mime_type'],
            width=file_info['width'],
            height=file_info['height'],
            channels=file_info.get('channels', 3),
            color_mode=file_info.get('color_mode', 'RGB'),
            image_type=image_type,
            metadata=file_info.get('metadata', {}),
            exif_data=file_info.get('exif_data', {})
        )


class MattingResult(BaseModel):
    """抠图结果模型"""

    __tablename__ = "matting_results"

    # 基础字段
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    source_image_id = Column(UUID(as_uuid=True), ForeignKey('images.id'), nullable=False, comment="源图像ID")

    # 抠图信息
    model_name = Column(String(100), nullable=False, comment="使用的模型名称")
    model_version = Column(String(50), nullable=True, comment="模型版本")
    target_object = Column(String(200), nullable=True, comment="目标对象描述")

    # 结果文件
    result_path = Column(String(500), nullable=False, comment="结果文件路径")
    mask_path = Column(String(500), nullable=True, comment="掩码文件路径")
    preview_path = Column(String(500), nullable=True, comment="预览图路径")

    # 处理参数
    processing_params = Column(JSON, default=dict, comment="处理参数")

    # 质量评估
    quality_score = Column(Float, nullable=True, comment="抠图质量评分")
    edge_quality = Column(Float, nullable=True, comment="边缘质量")
    transparency_quality = Column(Float, nullable=True, comment="透明度质量")

    # 统计信息
    processing_time = Column(Float, nullable=True, comment="处理耗时(秒)")
    file_size = Column(Float, nullable=True, comment="结果文件大小(MB)")

    # 状态
    status = Column(String(50), default="processing", comment="状态")  # processing, completed, failed
    error_message = Column(Text, nullable=True, comment="错误信息")

    # 时间字段
    started_at = Column(DateTime, default=datetime.utcnow, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    # 关联关系
    source_image = relationship("Image", back_populates="matting_results")

    # 索引
    __table_args__ = (
        Index('idx_matting_source_image', 'source_image_id'),
        Index('idx_matting_status', 'status'),
        Index('idx_matting_model', 'model_name'),
    )

    def __repr__(self):
        return f"<MattingResult(id={self.id}, model='{self.model_name}', status='{self.status}')>"

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == "failed"

    @property
    def processing_duration(self) -> Optional[float]:
        """处理持续时间"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def mark_completed(self, result_info: Dict[str, Any]):
        """标记为完成"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.result_path = result_info.get('result_path')
        self.mask_path = result_info.get('mask_path')
        self.preview_path = result_info.get('preview_path')
        self.quality_score = result_info.get('quality_score')
        self.edge_quality = result_info.get('edge_quality')
        self.transparency_quality = result_info.get('transparency_quality')
        self.processing_time = result_info.get('processing_time')
        self.file_size = result_info.get('file_size')

    def mark_failed(self, error_message: str):
        """标记为失败"""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

    def get_result_info(self) -> Dict[str, Any]:
        """获取结果信息"""
        return {
            'id': str(self.id),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'target_object': self.target_object,
            'result_path': self.result_path,
            'mask_path': self.mask_path,
            'preview_path': self.preview_path,
            'quality_score': self.quality_score,
            'edge_quality': self.edge_quality,
            'transparency_quality': self.transparency_quality,
            'processing_time': self.processing_time,
            'file_size': self.file_size,
            'status': self.status,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }