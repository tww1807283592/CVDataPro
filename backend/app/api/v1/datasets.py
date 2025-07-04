"""
数据集管理API
提供数据集的创建、管理、标注格式转换、导出下载等功能
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload
import asyncio
import json
import zipfile
import tempfile
import os
from pathlib import Path as PathLib
from datetime import datetime, timedelta
import logging
from uuid import uuid4
import aiofiles
import shutil

from app.core.deps import (
    get_async_db,
    get_current_active_user,
    get_redis,
    require_project_access,
    get_pagination_params,
    api_rate_limiter,
    heavy_task_rate_limiter,
    get_cached_response,
    set_cache_response,
    get_task_status
)
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset, DatasetImage, DatasetAnnotation
from app.models.image import Image
from app.models.synthesis_task import SynthesisTask
from app.schemas.user import UserInDB
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetInDB,
    DatasetWithStats,
    DatasetImageCreate,
    DatasetImageUpdate,
    DatasetImageInDB,
    DatasetAnnotationCreate,
    DatasetAnnotationUpdate,
    DatasetAnnotationInDB,
    DatasetExportRequest,
    DatasetExportResponse,
    DatasetImportRequest,
    DatasetImportResponse,
    DatasetStatsResponse,
    DatasetValidationResponse,
    AnnotationFormat,
    DatasetSplitRequest,
    DatasetMergeRequest
)
from app.services.dataset_service import DatasetService
from app.tasks.dataset_tasks import (
    export_dataset_task,
    import_dataset_task,
    validate_dataset_task,
    split_dataset_task,
    merge_datasets_task,
    auto_generate_dataset_task
)
from app.utils.file_utils import ensure_directory, get_file_size, get_file_hash
from app.utils.export_utils import create_export_archive
from app.utils.validation_utils import validate_annotation_format

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ================================
# 数据集CRUD操作
# ================================

@router.post("/", response_model=DatasetInDB, status_code=status.HTTP_201_CREATED)
async def create_dataset(
        dataset_data: DatasetCreate,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(require_project_access("write"))
):
    """
    创建新数据集
    """
    try:
        # 检查数据集名称是否重复
        existing_dataset = await db.execute(
            select(Dataset).where(
                and_(
                    Dataset.project_id == dataset_data.project_id,
                    Dataset.name == dataset_data.name
                )
            )
        )
        if existing_dataset.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="数据集名称已存在"
            )

        # 创建数据集
        dataset = Dataset(
            **dataset_data.model_dump(),
            created_by=current_user.id,
            updated_by=current_user.id
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # 异步初始化数据集目录
        background_tasks.add_task(
            DatasetService.initialize_dataset_directory,
            dataset.id
        )

        logger.info(f"用户 {current_user.username} 创建了数据集 {dataset.name}")

        return DatasetInDB.model_validate(dataset)

    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建数据集失败"
        )


@router.get("/", response_model=List[DatasetWithStats])
async def get_datasets(
        project_id: Optional[int] = Query(None, description="项目ID"),
        search: Optional[str] = Query(None, description="搜索关键词"),
        annotation_format: Optional[AnnotationFormat] = Query(None, description="标注格式"),
        status_filter: Optional[str] = Query(None, description="状态筛选"),
        pagination: Dict[str, int] = Depends(get_pagination_params),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        redis_client=Depends(get_redis),
        _: None = Depends(api_rate_limiter)
):
    """
    获取数据集列表
    """
    try:
        # 构建查询
        query = select(Dataset).options(
            selectinload(Dataset.project),
            selectinload(Dataset.created_by_user),
            selectinload(Dataset.updated_by_user)
        )

        # 权限过滤
        if not current_user.is_superuser:
            # 只能查看自己创建的数据集或有权限的项目数据集
            query = query.where(
                or_(
                    Dataset.created_by == current_user.id,
                    Dataset.project_id.in_(
                        select(Project.id).where(
                            or_(
                                Project.owner_id == current_user.id,
                                Project.id.in_(
                                    select(ProjectMember.project_id).where(
                                        ProjectMember.user_id == current_user.id
                                    )
                                )
                            )
                        )
                    )
                )
            )

        # 项目筛选
        if project_id:
            query = query.where(Dataset.project_id == project_id)

        # 搜索筛选
        if search:
            query = query.where(
                or_(
                    Dataset.name.ilike(f"%{search}%"),
                    Dataset.description.ilike(f"%{search}%")
                )
            )

        # 格式筛选
        if annotation_format:
            query = query.where(Dataset.annotation_format == annotation_format)

        # 状态筛选
        if status_filter:
            query = query.where(Dataset.status == status_filter)

        # 分页
        query = query.offset(pagination["offset"]).limit(pagination["limit"])

        # 执行查询
        result = await db.execute(query)
        datasets = result.scalars().all()

        # 获取统计信息
        datasets_with_stats = []
        for dataset in datasets:
            stats = await DatasetService.get_dataset_stats(db, dataset.id)
            dataset_dict = dataset.__dict__.copy()
            dataset_dict.update(stats)
            datasets_with_stats.append(DatasetWithStats(**dataset_dict))

        return datasets_with_stats

    except Exception as e:
        logger.error(f"获取数据集列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取数据集列表失败"
        )


@router.get("/{dataset_id}", response_model=DatasetWithStats)
async def get_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        redis_client=Depends(get_redis)
):
    """
    获取数据集详情
    """
    try:
        # 获取数据集
        result = await db.execute(
            select(Dataset).options(
                selectinload(Dataset.project),
                selectinload(Dataset.created_by_user),
                selectinload(Dataset.updated_by_user)
            ).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据集不存在"
            )

        # 权限检查
        if not current_user.is_superuser:
            if dataset.created_by != current_user.id:
                # 检查项目权限
                project_access = await DatasetService.check_project_access(
                    db, dataset.project_id, current_user.id
                )
                if not project_access:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="无权访问该数据集"
                    )

        # 获取统计信息
        stats = await DatasetService.get_dataset_stats(db, dataset_id)

        # 构建响应
        dataset_dict = dataset.__dict__.copy()
        dataset_dict.update(stats)

        return DatasetWithStats(**dataset_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取数据集详情失败"
        )


@router.put("/{dataset_id}", response_model=DatasetInDB)
async def update_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        dataset_data: DatasetUpdate = None,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    更新数据集
    """
    try:
        # 获取数据集
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据集不存在"
            )

        # 权限检查
        if not current_user.is_superuser:
            if dataset.created_by != current_user.id:
                project_access = await DatasetService.check_project_access(
                    db, dataset.project_id, current_user.id, "write"
                )
                if not project_access:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="无权修改该数据集"
                    )

        # 更新数据集
        update_data = dataset_data.model_dump(exclude_unset=True)
        if update_data:
            for field, value in update_data.items():
                setattr(dataset, field, value)

            dataset.updated_by = current_user.id
            dataset.updated_at = datetime.utcnow()

            await db.commit()
            await db.refresh(dataset)

        logger.info(f"用户 {current_user.username} 更新了数据集 {dataset.name}")

        return DatasetInDB.model_validate(dataset)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据集失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新数据集失败"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        force: bool = Query(False, description="强制删除"),
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    删除数据集
    """
    try:
        # 获取数据集
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据集不存在"
            )

        # 权限检查
        if not current_user.is_superuser:
            if dataset.created_by != current_user.id:
                project_access = await DatasetService.check_project_access(
                    db, dataset.project_id, current_user.id, "write"
                )
                if not project_access:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="无权删除该数据集"
                    )

        # 检查是否有关联的导出任务
        if not force:
            export_count = await db.execute(
                select(func.count(DatasetExport.id)).where(
                    and_(
                        DatasetExport.dataset_id == dataset_id,
                        DatasetExport.status.in_(["pending", "processing"])
                    )
                )
            )
            if export_count.scalar() > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="数据集有正在进行的导出任务，请先取消或等待完成"
                )

        # 删除数据集
        await db.delete(dataset)
        await db.commit()

        # 异步删除相关文件
        background_tasks.add_task(
            DatasetService.cleanup_dataset_files,
            dataset_id
        )

        logger.info(f"用户 {current_user.username} 删除了数据集 {dataset.name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据集失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除数据集失败"
        )


# ================================
# 数据集图像管理
# ================================

@router.get("/{dataset_id}/images", response_model=List[DatasetImageInDB])
async def get_dataset_images(
        dataset_id: int = Path(..., description="数据集ID"),
        search: Optional[str] = Query(None, description="搜索关键词"),
        has_annotation: Optional[bool] = Query(None, description="是否有标注"),
        pagination: Dict[str, int] = Depends(get_pagination_params),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    获取数据集图像列表
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 构建查询
        query = select(DatasetImage).options(
            selectinload(DatasetImage.image),
            selectinload(DatasetImage.annotations)
        ).where(DatasetImage.dataset_id == dataset_id)

        # 搜索筛选
        if search:
            query = query.join(Image).where(
                or_(
                    Image.filename.ilike(f"%{search}%"),
                    Image.original_filename.ilike(f"%{search}%")
                )
            )

        # 标注筛选
        if has_annotation is not None:
            if has_annotation:
                query = query.where(DatasetImage.annotations.any())
            else:
                query = query.where(~DatasetImage.annotations.any())

        # 分页
        query = query.offset(pagination["offset"]).limit(pagination["limit"])

        # 执行查询
        result = await db.execute(query)
        dataset_images = result.scalars().all()

        return [DatasetImageInDB.model_validate(img) for img in dataset_images]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集图像列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取数据集图像列表失败"
        )


@router.post("/{dataset_id}/images", response_model=List[DatasetImageInDB])
async def add_images_to_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        image_ids: List[int],
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    向数据集添加图像
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 验证图像是否存在
        result = await db.execute(
            select(Image).where(Image.id.in_(image_ids))
        )
        images = result.scalars().all()

        if len(images) != len(image_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="部分图像不存在"
            )

        # 检查图像是否已在数据集中
        existing_result = await db.execute(
            select(DatasetImage.image_id).where(
                and_(
                    DatasetImage.dataset_id == dataset_id,
                    DatasetImage.image_id.in_(image_ids)
                )
            )
        )
        existing_image_ids = {row[0] for row in existing_result.fetchall()}

        # 添加新图像
        new_dataset_images = []
        for image_id in image_ids:
            if image_id not in existing_image_ids:
                dataset_image = DatasetImage(
                    dataset_id=dataset_id,
                    image_id=image_id,
                    added_by=current_user.id
                )
                db.add(dataset_image)
                new_dataset_images.append(dataset_image)

        await db.commit()

        # 刷新数据
        for dataset_image in new_dataset_images:
            await db.refresh(dataset_image)

        # 更新数据集统计
        await DatasetService.update_dataset_stats(db, dataset_id)

        logger.info(f"用户 {current_user.username} 向数据集 {dataset_id} 添加了 {len(new_dataset_images)} 张图像")

        return [DatasetImageInDB.model_validate(img) for img in new_dataset_images]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加图像到数据集失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="添加图像到数据集失败"
        )


@router.delete("/{dataset_id}/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_image_from_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        image_id: int = Path(..., description="图像ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    从数据集中移除图像
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 获取数据集图像
        result = await db.execute(
            select(DatasetImage).where(
                and_(
                    DatasetImage.dataset_id == dataset_id,
                    DatasetImage.image_id == image_id
                )
            )
        )
        dataset_image = result.scalar_one_or_none()

        if not dataset_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="图像不在数据集中"
            )

        # 删除数据集图像及其标注
        await db.delete(dataset_image)
        await db.commit()

        # 更新数据集统计
        await DatasetService.update_dataset_stats(db, dataset_id)

        logger.info(f"用户 {current_user.username} 从数据集 {dataset_id} 中移除了图像 {image_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从数据集中移除图像失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="从数据集中移除图像失败"
        )


# ================================
# 数据集标注管理
# ================================

@router.get("/{dataset_id}/annotations", response_model=List[DatasetAnnotationInDB])
async def get_dataset_annotations(
        dataset_id: int = Path(..., description="数据集ID"),
        image_id: Optional[int] = Query(None, description="图像ID"),
        annotation_type: Optional[str] = Query(None, description="标注类型"),
        pagination: Dict[str, int] = Depends(get_pagination_params),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    获取数据集标注列表
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 构建查询
        query = select(DatasetAnnotation).options(
            selectinload(DatasetAnnotation.dataset_image),
            selectinload(DatasetAnnotation.created_by_user)
        ).join(DatasetImage).where(
            DatasetImage.dataset_id == dataset_id
        )

        # 图像筛选
        if image_id:
            query = query.where(DatasetAnnotation.dataset_image_id == image_id)

        # 标注类型筛选
        if annotation_type:
            query = query.where(DatasetAnnotation.annotation_type == annotation_type)

        # 分页
        query = query.offset(pagination["offset"]).limit(pagination["limit"])

        # 执行查询
        result = await db.execute(query)
        annotations = result.scalars().all()

        return [DatasetAnnotationInDB.model_validate(ann) for ann in annotations]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集标注列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取数据集标注列表失败"
        )


@router.post("/{dataset_id}/annotations", response_model=DatasetAnnotationInDB)
async def create_dataset_annotation(
        dataset_id: int = Path(..., description="数据集ID"),
        annotation_data: DatasetAnnotationCreate,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    创建数据集标注
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 验证数据集图像
        result = await db.execute(
            select(DatasetImage).where(
                and_(
                    DatasetImage.id == annotation_data.dataset_image_id,
                    DatasetImage.dataset_id == dataset_id
                )
            )
        )
        dataset_image = result.scalar_one_or_none()

        if not dataset_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据集图像不存在"
            )

        # 验证标注数据格式
        if not validate_annotation_format(
                annotation_data.annotation_data,
                annotation_data.annotation_type
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="标注数据格式不正确"
            )

        # 创建标注
        annotation = DatasetAnnotation(
            **annotation_data.model_dump(),
            created_by=current_user.id,
            updated_by=current_user.id
        )

        db.add(annotation)
        await db.commit()
        await db.refresh(annotation)

        # 更新数据集统计
        await DatasetService.update_dataset_stats(db, dataset_id)

        logger.info(f"用户 {current_user.username} 为数据集 {dataset_id} 创建了标注")

        return DatasetAnnotationInDB.model_validate(annotation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建数据集标注失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建数据集标注失败"
        )


@router.put("/{dataset_id}/annotations/{annotation_id}", response_model=DatasetAnnotationInDB)
async def update_dataset_annotation(
        dataset_id: int = Path(..., description="数据集ID"),
        annotation_id: int = Path(..., description="标注ID"),
        annotation_data: DatasetAnnotationUpdate,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    更新数据集标注
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 获取标注
        result = await db.execute(
            select(DatasetAnnotation).join(DatasetImage).where(
                and_(
                    DatasetAnnotation.id == annotation_id,
                    DatasetImage.dataset_id == dataset_id
                )
            )
        )
        annotation = result.scalar_one_or_none()

        if not annotation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="标注不存在"
            )

        # 验证标注数据格式
        update_data = annotation_data.model_dump(exclude_unset=True)
        if "annotation_data" in update_data:
            if not validate_annotation_format(
                    update_data["annotation_data"],
                    update_data.get("annotation_type", annotation.annotation_type)
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="标注数据格式不正确"
                )

        # 更新标注
        for field, value in update_data.items():
            setattr(annotation, field, value)

        annotation.updated_by = current_user.id
        annotation.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(annotation)

        logger.info(f"用户 {current_user.username} 更新了数据集 {dataset_id} 的标注 {annotation_id}")

        return DatasetAnnotationInDB.model_validate(annotation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据集标注失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新数据集标注失败"
        )


@router.delete("/{dataset_id}/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset_annotation(
        dataset_id: int = Path(..., description="数据集ID"),
        annotation_id: int = Path(..., description="标注ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    删除数据集标注
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 获取标注
        result = await db.execute(
            select(DatasetAnnotation).join(DatasetImage).where(
                and_(
                    DatasetAnnotation.id == annotation_id,
                    DatasetImage.dataset_id == dataset_id
                )
            )
        )
        annotation = result.scalar_one_or_none()

        if not annotation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="标注不存在"
            )

        # 删除标注
        await db.delete(annotation)
        await db.commit()

        # 更新数据集统计
        await DatasetService.update_dataset_stats(db, dataset_id)

        logger.info(f"用户 {current_user.username} 删除了数据集 {dataset_id} 的标注 {annotation_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据集标注失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除数据集标注失败"
        )


# ================================
# 数据集统计信息
# ================================

@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
        dataset_id: int = Path(..., description="数据集ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        redis_client=Depends(get_redis)
):
    """
    获取数据集统计信息
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 尝试从缓存获取
        cache_key = f"dataset_stats:{dataset_id}"
        cached_stats = await get_cached_response(redis_client, cache_key)
        if cached_stats:
            return DatasetStatsResponse.model_validate(cached_stats)

        # 获取统计信息
        stats = await DatasetService.get_detailed_dataset_stats(db, dataset_id)

        # 缓存结果（5分钟）
        await set_cache_response(redis_client, cache_key, stats, expire=300)

        return DatasetStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取数据集统计信息失败"
        )


# ================================
# 数据集导出功能
# ================================

@router.post("/{dataset_id}/export", response_model=DatasetExportResponse)
async def export_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        export_request: DatasetExportRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    导出数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 验证导出格式
        if export_request.format not in ["coco", "yolo", "voc", "custom"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不支持的导出格式"
            )

        # 创建导出任务
        task_id = str(uuid4())
        export_task = {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "format": export_request.format,
            "options": export_request.options,
            "created_by": current_user.id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }

        # 启动后台任务
        background_tasks.add_task(
            export_dataset_task,
            task_id,
            dataset_id,
            export_request.format,
            export_request.options,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始导出数据集 {dataset_id}")

        return DatasetExportResponse(
            task_id=task_id,
            status="pending",
            message="导出任务已启动"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="导出数据集失败"
        )


@router.get("/{dataset_id}/export/{task_id}/status")
async def get_export_status(
        dataset_id: int = Path(..., description="数据集ID"),
        task_id: str = Path(..., description="任务ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        redis_client=Depends(get_redis)
):
    """
    获取导出任务状态
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 获取任务状态
        task_status = await get_task_status(redis_client, task_id)
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任务不存在"
            )

        return task_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取导出任务状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取导出任务状态失败"
        )


@router.get("/{dataset_id}/export/{task_id}/download")
async def download_exported_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        task_id: str = Path(..., description="任务ID"),
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        redis_client=Depends(get_redis)
):
    """
    下载导出的数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 检查任务状态
        task_status = await get_task_status(redis_client, task_id)
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任务不存在"
            )

        if task_status.get("status") != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="任务尚未完成"
            )

        # 获取导出文件路径
        export_file_path = task_status.get("export_file_path")
        if not export_file_path or not os.path.exists(export_file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="导出文件不存在"
            )

        # 返回文件
        filename = f"dataset_{dataset_id}_{task_id}.zip"
        return FileResponse(
            path=export_file_path,
            filename=filename,
            media_type="application/zip"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载导出数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="下载导出数据集失败"
        )


# ================================
# 数据集导入功能
# ================================

@router.post("/{dataset_id}/import", response_model=DatasetImportResponse)
async def import_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        import_request: DatasetImportRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    导入数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 验证导入格式
        if import_request.format not in ["coco", "yolo", "voc", "custom"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不支持的导入格式"
            )

        # 创建导入任务
        task_id = str(uuid4())
        import_task = {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "format": import_request.format,
            "source": import_request.source,
            "options": import_request.options,
            "created_by": current_user.id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }

        # 启动后台任务
        background_tasks.add_task(
            import_dataset_task,
            task_id,
            dataset_id,
            import_request.format,
            import_request.source,
            import_request.options,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始导入数据集 {dataset_id}")

        return DatasetImportResponse(
            task_id=task_id,
            status="pending",
            message="导入任务已启动"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="导入数据集失败"
        )


# ================================
# 数据集验证功能
# ================================

@router.post("/{dataset_id}/validate", response_model=DatasetValidationResponse)
async def validate_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    验证数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id
        )

        # 创建验证任务
        task_id = str(uuid4())

        # 启动后台任务
        background_tasks.add_task(
            validate_dataset_task,
            task_id,
            dataset_id,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始验证数据集 {dataset_id}")

        return DatasetValidationResponse(
            task_id=task_id,
            status="pending",
            message="验证任务已启动"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="验证数据集失败"
        )


# ================================
# 数据集分割功能
# ================================

@router.post("/{dataset_id}/split", response_model=Dict[str, Any])
async def split_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        split_request: DatasetSplitRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    分割数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 验证分割比例
        total_ratio = sum([
            split_request.train_ratio,
            split_request.val_ratio,
            split_request.test_ratio
        ])
        if abs(total_ratio - 1.0) > 0.001:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="分割比例总和必须为1"
            )

        # 创建分割任务
        task_id = str(uuid4())

        # 启动后台任务
        background_tasks.add_task(
            split_dataset_task,
            task_id,
            dataset_id,
            split_request.train_ratio,
            split_request.val_ratio,
            split_request.test_ratio,
            split_request.random_seed,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始分割数据集 {dataset_id}")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "分割任务已启动"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分割数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="分割数据集失败"
        )


# ================================
# 数据集合并功能
# ================================

@router.post("/merge", response_model=Dict[str, Any])
async def merge_datasets(
        merge_request: DatasetMergeRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    合并数据集
    """
    try:
        # 验证源数据集
        for dataset_id in merge_request.source_dataset_ids:
            dataset = await DatasetService.get_dataset_with_permission_check(
                db, dataset_id, current_user.id
            )

        # 验证目标数据集名称
        existing_dataset = await db.execute(
            select(Dataset).where(
                and_(
                    Dataset.project_id == merge_request.target_project_id,
                    Dataset.name == merge_request.target_dataset_name
                )
            )
        )
        if existing_dataset.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="目标数据集名称已存在"
            )

        # 创建合并任务
        task_id = str(uuid4())

        # 启动后台任务
        background_tasks.add_task(
            merge_datasets_task,
            task_id,
            merge_request.source_dataset_ids,
            merge_request.target_project_id,
            merge_request.target_dataset_name,
            merge_request.target_dataset_description,
            merge_request.merge_strategy,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始合并数据集 {merge_request.source_dataset_ids}")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "合并任务已启动"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"合并数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="合并数据集失败"
        )


# ================================
# 数据集自动生成功能
# ================================

@router.post("/{dataset_id}/auto-generate", response_model=Dict[str, Any])
async def auto_generate_dataset(
        dataset_id: int = Path(..., description="数据集ID"),
        generation_config: Dict[str, Any],
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_async_db),
        current_user: UserInDB = Depends(get_current_active_user),
        _: None = Depends(heavy_task_rate_limiter)
):
    """
    自动生成数据集
    """
    try:
        # 权限检查
        dataset = await DatasetService.get_dataset_with_permission_check(
            db, dataset_id, current_user.id, "write"
        )

        # 验证生成配置
        required_fields = ["generation_type", "count", "parameters"]
        for field in required_fields:
            if field not in generation_config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"缺少必需字段: {field}"
                )

        # 创建生成任务
        task_id = str(uuid4())

        # 启动后台任务
        background_tasks.add_task(
            auto_generate_dataset_task,
            task_id,
            dataset_id,
            generation_config,
            current_user.id
        )

        logger.info(f"用户 {current_user.username} 开始自动生成数据集 {dataset_id}")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "自动生成任务已启动"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自动生成数据集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="自动生成数据集失败"
        )