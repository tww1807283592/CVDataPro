"""
图像管理API路由模块
支持图像上传、获取、分类、删除、批量操作等功能
"""

from typing import List, Optional, Dict, Any
import asyncio
import aiofiles
from pathlib import Path
from uuid import uuid4
import mimetypes
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, func, or_, and_

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.image import Image, ImageCategory, ImageStatus
from app.models.project import Project
from app.schemas.image import (
    ImageCreate, ImageResponse, ImageUpdate, ImageListResponse,
    ImageUploadResponse, ImageBatchResponse, ImageStatistics,
    ImageSearchRequest, ImageMetadata, ImageCategoryUpdate
)
from app.services.image_service import ImageService
from app.utils.file_utils import FileUtils
from app.utils.image_utils import ImageUtils
from app.utils.validation_utils import ValidationUtils
from app.tasks.celery_app import celery_app

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        project_id: Optional[int] = Form(None),
        category: Optional[ImageCategory] = Form(ImageCategory.UNKNOWN),
        description: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    上传单个图像文件

    Args:
        file: 上传的图像文件
        project_id: 项目ID（可选）
        category: 图像分类
        description: 图像描述
        tags: 图像标签，逗号分隔

    Returns:
        ImageUploadResponse: 上传结果
    """
    try:
        # 验证文件格式
        if not ValidationUtils.is_valid_image_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail="不支持的图像格式"
            )

        # 验证文件大小
        if file.size > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({settings.MAX_IMAGE_SIZE / 1024 / 1024}MB)"
            )

        # 验证项目权限
        if project_id:
            project = await db.get(Project, project_id)
            if not project or project.user_id != current_user.id:
                raise HTTPException(
                    status_code=404,
                    detail="项目不存在或无权限"
                )

        # 生成唯一文件名
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid4()}{file_extension}"

        # 创建存储路径
        upload_dir = Path(settings.UPLOAD_DIR) / str(current_user.id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / unique_filename

        # 保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # 提取图像元数据
        try:
            metadata = await ImageUtils.extract_metadata(str(file_path))
        except Exception as e:
            # 如果元数据提取失败，删除文件并返回错误
            await FileUtils.delete_file(str(file_path))
            raise HTTPException(
                status_code=400,
                detail="图像文件损坏或格式不支持"
            )

        # 创建图像记录
        image_data = ImageCreate(
            filename=file.filename,
            file_path=str(file_path),
            file_size=file.size,
            mime_type=mimetypes.guess_type(file.filename)[0] or "application/octet-stream",
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            category=category,
            description=description,
            tags=tags.split(",") if tags else [],
            project_id=project_id,
            user_id=current_user.id
        )

        image_service = ImageService(db)
        image = await image_service.create_image(image_data)

        # 添加后台任务：生成缩略图
        background_tasks.add_task(
            _generate_thumbnail_task,
            image.id,
            str(file_path)
        )

        return ImageUploadResponse(
            success=True,
            message="图像上传成功",
            image_id=image.id,
            filename=image.filename,
            file_size=image.file_size,
            thumbnail_url=f"/api/v1/images/{image.id}/thumbnail"
        )

    except HTTPException:
        raise
    except Exception as e:
        # 清理已上传的文件
        if 'file_path' in locals():
            await FileUtils.delete_file(str(file_path))
        raise HTTPException(
            status_code=500,
            detail=f"上传失败: {str(e)}"
        )


@router.post("/upload/batch", response_model=ImageBatchResponse)
async def upload_images_batch(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        project_id: Optional[int] = Form(None),
        category: Optional[ImageCategory] = Form(ImageCategory.UNKNOWN),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    批量上传图像文件

    Args:
        files: 上传的图像文件列表
        project_id: 项目ID（可选）
        category: 图像分类

    Returns:
        ImageBatchResponse: 批量上传结果
    """
    if len(files) > settings.MAX_BATCH_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"批量上传文件数量超过限制 ({settings.MAX_BATCH_UPLOAD})"
        )

    # 验证项目权限
    if project_id:
        project = await db.get(Project, project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(
                status_code=404,
                detail="项目不存在或无权限"
            )

    successful_uploads = []
    failed_uploads = []

    for file in files:
        try:
            # 验证文件
            if not ValidationUtils.is_valid_image_file(file.filename):
                failed_uploads.append({
                    "filename": file.filename,
                    "error": "不支持的图像格式"
                })
                continue

            if file.size > settings.MAX_IMAGE_SIZE:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": "文件大小超过限制"
                })
                continue

            # 处理单个文件上传
            result = await _process_single_upload(
                file, project_id, category, current_user, db
            )

            if result["success"]:
                successful_uploads.append(result["data"])
                # 添加后台任务
                background_tasks.add_task(
                    _generate_thumbnail_task,
                    result["data"]["image_id"],
                    result["data"]["file_path"]
                )
            else:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": result["error"]
                })

        except Exception as e:
            failed_uploads.append({
                "filename": file.filename,
                "error": str(e)
            })

    return ImageBatchResponse(
        success=len(failed_uploads) == 0,
        message=f"成功上传 {len(successful_uploads)} 个文件，失败 {len(failed_uploads)} 个文件",
        successful_count=len(successful_uploads),
        failed_count=len(failed_uploads),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads
    )


@router.get("/", response_model=ImageListResponse)
async def get_images(
        project_id: Optional[int] = Query(None),
        category: Optional[ImageCategory] = Query(None),
        status: Optional[ImageStatus] = Query(None),
        search: Optional[str] = Query(None),
        tags: Optional[str] = Query(None),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        sort_by: str = Query("created_at"),
        sort_order: str = Query("desc"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取图像列表

    Args:
        project_id: 项目ID筛选
        category: 图像分类筛选
        status: 图像状态筛选
        search: 搜索关键词
        tags: 标签筛选，逗号分隔
        page: 页码
        page_size: 每页数量
        sort_by: 排序字段
        sort_order: 排序方向

    Returns:
        ImageListResponse: 图像列表响应
    """
    try:
        image_service = ImageService(db)

        # 构建查询条件
        filters = {"user_id": current_user.id}

        if project_id:
            filters["project_id"] = project_id
        if category:
            filters["category"] = category
        if status:
            filters["status"] = status

        # 构建搜索条件
        search_conditions = []
        if search:
            search_conditions.append(
                or_(
                    Image.filename.ilike(f"%{search}%"),
                    Image.description.ilike(f"%{search}%")
                )
            )

        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            for tag in tag_list:
                search_conditions.append(
                    Image.tags.any(tag)
                )

        # 获取图像列表
        images, total = await image_service.get_images_with_pagination(
            filters=filters,
            search_conditions=search_conditions,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )

        # 构建响应数据
        image_responses = []
        for image in images:
            image_responses.append(ImageResponse(
                id=image.id,
                filename=image.filename,
                file_size=image.file_size,
                mime_type=image.mime_type,
                width=image.width,
                height=image.height,
                category=image.category,
                status=image.status,
                description=image.description,
                tags=image.tags,
                project_id=image.project_id,
                thumbnail_url=f"/api/v1/images/{image.id}/thumbnail",
                preview_url=f"/api/v1/images/{image.id}/preview",
                created_at=image.created_at,
                updated_at=image.updated_at
            ))

        return ImageListResponse(
            items=image_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取图像列表失败: {str(e)}"
        )


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
        image_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取单个图像详情

    Args:
        image_id: 图像ID

    Returns:
        ImageResponse: 图像详情
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    return ImageResponse(
        id=image.id,
        filename=image.filename,
        file_size=image.file_size,
        mime_type=image.mime_type,
        width=image.width,
        height=image.height,
        category=image.category,
        status=image.status,
        description=image.description,
        tags=image.tags,
        project_id=image.project_id,
        thumbnail_url=f"/api/v1/images/{image.id}/thumbnail",
        preview_url=f"/api/v1/images/{image.id}/preview",
        download_url=f"/api/v1/images/{image.id}/download",
        created_at=image.created_at,
        updated_at=image.updated_at
    )


@router.get("/{image_id}/thumbnail")
async def get_thumbnail(
        image_id: int,
        size: int = Query(200, ge=50, le=500),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取图像缩略图

    Args:
        image_id: 图像ID
        size: 缩略图大小

    Returns:
        FileResponse: 缩略图文件
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    # 生成缩略图路径
    thumbnail_dir = Path(settings.THUMBNAIL_DIR) / str(current_user.id)
    thumbnail_path = thumbnail_dir / f"{image.id}_{size}.jpg"

    # 如果缩略图不存在，生成缩略图
    if not thumbnail_path.exists():
        try:
            await ImageUtils.generate_thumbnail(
                image.file_path,
                str(thumbnail_path),
                size
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"生成缩略图失败: {str(e)}"
            )

    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg",
        filename=f"{image.filename}_thumbnail.jpg"
    )


@router.get("/{image_id}/preview")
async def get_preview(
        image_id: int,
        width: Optional[int] = Query(None, ge=100, le=1920),
        height: Optional[int] = Query(None, ge=100, le=1080),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取图像预览

    Args:
        image_id: 图像ID
        width: 预览宽度
        height: 预览高度

    Returns:
        StreamingResponse: 预览图像流
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    try:
        # 生成预览图像
        preview_data = await ImageUtils.generate_preview(
            image.file_path,
            width,
            height
        )

        return StreamingResponse(
            preview_data,
            media_type=image.mime_type,
            headers={
                "Content-Disposition": f"inline; filename={image.filename}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成预览失败: {str(e)}"
        )


@router.get("/{image_id}/download")
async def download_image(
        image_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    下载原图像文件

    Args:
        image_id: 图像ID

    Returns:
        FileResponse: 原图像文件
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    if not Path(image.file_path).exists():
        raise HTTPException(
            status_code=404,
            detail="图像文件不存在"
        )

    return FileResponse(
        path=image.file_path,
        media_type=image.mime_type,
        filename=image.filename
    )


@router.put("/{image_id}", response_model=ImageResponse)
async def update_image(
        image_id: int,
        image_update: ImageUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    更新图像信息

    Args:
        image_id: 图像ID
        image_update: 更新数据

    Returns:
        ImageResponse: 更新后的图像信息
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    # 验证项目权限
    if image_update.project_id:
        project = await db.get(Project, image_update.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(
                status_code=404,
                detail="项目不存在或无权限"
            )

    try:
        updated_image = await image_service.update_image(image_id, image_update)

        return ImageResponse(
            id=updated_image.id,
            filename=updated_image.filename,
            file_size=updated_image.file_size,
            mime_type=updated_image.mime_type,
            width=updated_image.width,
            height=updated_image.height,
            category=updated_image.category,
            status=updated_image.status,
            description=updated_image.description,
            tags=updated_image.tags,
            project_id=updated_image.project_id,
            thumbnail_url=f"/api/v1/images/{updated_image.id}/thumbnail",
            preview_url=f"/api/v1/images/{updated_image.id}/preview",
            download_url=f"/api/v1/images/{updated_image.id}/download",
            created_at=updated_image.created_at,
            updated_at=updated_image.updated_at
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新图像失败: {str(e)}"
        )


@router.patch("/{image_id}/category", response_model=ImageResponse)
async def update_image_category(
        image_id: int,
        category_update: ImageCategoryUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    更新图像分类

    Args:
        image_id: 图像ID
        category_update: 分类更新数据

    Returns:
        ImageResponse: 更新后的图像信息
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    try:
        updated_image = await image_service.update_image_category(
            image_id, category_update.category
        )

        return ImageResponse(
            id=updated_image.id,
            filename=updated_image.filename,
            file_size=updated_image.file_size,
            mime_type=updated_image.mime_type,
            width=updated_image.width,
            height=updated_image.height,
            category=updated_image.category,
            status=updated_image.status,
            description=updated_image.description,
            tags=updated_image.tags,
            project_id=updated_image.project_id,
            thumbnail_url=f"/api/v1/images/{updated_image.id}/thumbnail",
            preview_url=f"/api/v1/images/{updated_image.id}/preview",
            download_url=f"/api/v1/images/{updated_image.id}/download",
            created_at=updated_image.created_at,
            updated_at=updated_image.updated_at
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新图像分类失败: {str(e)}"
        )


@router.delete("/{image_id}")
async def delete_image(
        image_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    删除图像

    Args:
        image_id: 图像ID

    Returns:
        dict: 删除结果
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    try:
        await image_service.delete_image(image_id)

        # 删除相关文件
        await _cleanup_image_files(image)

        return {"message": "图像删除成功"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除图像失败: {str(e)}"
        )


@router.post("/batch/delete")
async def batch_delete_images(
        image_ids: List[int],
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    批量删除图像

    Args:
        image_ids: 图像ID列表

    Returns:
        dict: 批量删除结果
    """
    if len(image_ids) > settings.MAX_BATCH_DELETE:
        raise HTTPException(
            status_code=400,
            detail=f"批量删除数量超过限制 ({settings.MAX_BATCH_DELETE})"
        )

    image_service = ImageService(db)

    # 验证所有图像的权限
    images = await image_service.get_images_by_ids(image_ids)
    for image in images:
        if image.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail=f"图像 {image.filename} 无权限删除"
            )

    try:
        deleted_count = await image_service.batch_delete_images(image_ids)

        # 异步清理文件
        for image in images:
            await _cleanup_image_files(image)

        return {
            "message": f"成功删除 {deleted_count} 个图像",
            "deleted_count": deleted_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量删除失败: {str(e)}"
        )


@router.get("/statistics", response_model=ImageStatistics)
async def get_image_statistics(
        project_id: Optional[int] = Query(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取图像统计信息

    Args:
        project_id: 项目ID（可选）

    Returns:
        ImageStatistics: 统计信息
    """
    try:
        image_service = ImageService(db)
        stats = await image_service.get_image_statistics(
            user_id=current_user.id,
            project_id=project_id
        )

        return ImageStatistics(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/{image_id}/metadata", response_model=ImageMetadata)
async def get_image_metadata(
        image_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    获取图像元数据

    Args:
        image_id: 图像ID

    Returns:
        ImageMetadata: 图像元数据
    """
    image_service = ImageService(db)
    image = await image_service.get_image_by_id(image_id)

    if not image or image.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="图像不存在或无权限访问"
        )

    try:
        metadata = await ImageUtils.extract_detailed_metadata(image.file_path)

        return ImageMetadata(
            filename=image.filename,
            file_size=image.file_size,
            mime_type=image.mime_type,
            width=image.width,
            height=image.height,
            color_mode=metadata.get("color_mode"),
            has_alpha=metadata.get("has_alpha", False),
            dpi=metadata.get("dpi"),
            exif_data=metadata.get("exif_data", {}),
            created_at=image.created_at,
            updated_at=image.updated_at
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取元数据失败: {str(e)}"
        )


# 辅助函数

async def _process_single_upload(
        file: UploadFile,
        project_id: Optional[int],
        category: ImageCategory,
        user: User,
        db: AsyncSession
) -> Dict[str, Any]:
    """处理单个文件上传"""
    try:
        # 生成唯一文件名
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid4()}{file_extension}"

        # 创建存储路径
        upload_dir = Path(settings.UPLOAD_DIR) / str(user.id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / unique_filename

        # 保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # 提取元数据
        metadata = await ImageUtils.extract_metadata(str(file_path))

        # 创建图像记录
        image_data = ImageCreate(
            filename=file.filename,
            file_path=str(file_path),
            file_size=file.size,
            mime_type=mimetypes.guess_type(file.filename)[0] or "application/octet-stream",
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            category=category,
            project_id=project_id,
            user_id=user.id
        )

        image_service = ImageService(db)
        image = await image_service.create_image(image_data)

        return {
            "success": True,
            "data": {
                "image_id": image.id,
                "filename": image.filename,
                "file_size": image.file_size,
                "file_path": str(file_path)
            }
        }

    except Exception as e:
        # 清理文件
        if 'file_path' in locals():
            await FileUtils.delete_file(str(file_path))

        return {
            "success": False,
            "error": str(e)
        }


async def _generate_thumbnail_task(image_id: int, file_path: str):
    """生成缩略图的后台任务"""
    try:
        # 调用Celery任务生成缩略图
        celery_app.send_task(
            'app.tasks.image_tasks.generate_thumbnail',
            args=[image_id, file_path]
        )

    except Exception as e:
        # 记录错误但不抛出异常，避免影响主要的上传流程
        # 这里应该使用日志记录系统
        print(f"生成缩略图任务失败 - Image ID: {image_id}, Error: {str(e)}")


async def _cleanup_image_files(image):
    """清理图像相关文件"""
    try:
        # 删除原始图像文件
        if Path(image.file_path).exists():
            await FileUtils.delete_file(image.file_path)

        # 删除缩略图文件
        user_id = image.user_id
        thumbnail_dir = Path(settings.THUMBNAIL_DIR) / str(user_id)

        if thumbnail_dir.exists():
            # 删除该图像的所有缩略图（不同尺寸）
            thumbnail_pattern = f"{image.id}_*.jpg"
            for thumbnail_file in thumbnail_dir.glob(thumbnail_pattern):
                await FileUtils.delete_file(str(thumbnail_file))

    except Exception as e:
        # 记录错误但不抛出异常
        print(f"清理图像文件失败 - Image ID: {image.id}, Error: {str(e)}")