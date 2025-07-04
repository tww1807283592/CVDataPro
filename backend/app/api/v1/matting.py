"""
智能抠图API路由模块
支持多种抠图算法：SAM、U2Net、RemBG等
提供自动抠图和手动指定物体抠图功能
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import os
from datetime import datetime
import json
import io
from PIL import Image
import aiofiles

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.image import Image as ImageModel
from app.models.synthesis_task import SynthesisTask
from app.schemas.image import ImageResponse, ImageCreate
from app.schemas.synthesis import (
    MattingTaskCreate,
    MattingTaskResponse,
    MattingRequest,
    MattingPreviewRequest,
    MattingBatchRequest,
    MattingQualityAssessment,
    MattingProgressResponse
)
from app.services.matting_service import MattingService
from app.services.image_service import ImageService
from app.tasks.matting_tasks import (
    process_matting_task,
    process_batch_matting_task,
    assess_matting_quality
)
from app.utils.file_utils import save_uploaded_file, generate_file_path
from app.utils.image_utils import validate_image_format, get_image_info
from app.utils.validation_utils import validate_matting_parameters
from app.core.redis import get_redis_client
from app.core.logging import get_logger

router = APIRouter(prefix="/matting", tags=["智能抠图"])
logger = get_logger(__name__)


# 依赖注入
async def get_matting_service(db: AsyncSession = Depends(get_db)) -> MattingService:
    """获取抠图服务实例"""
    return MattingService(db)


async def get_image_service(db: AsyncSession = Depends(get_db)) -> ImageService:
    """获取图像服务实例"""
    return ImageService(db)


@router.post("/upload", response_model=ImageResponse)
async def upload_image_for_matting(
        file: UploadFile = File(...),
        project_id: Optional[int] = None,
        current_user: User = Depends(get_current_user),
        image_service: ImageService = Depends(get_image_service)
):
    """
    上传图像用于抠图处理

    Args:
        file: 上传的图像文件
        project_id: 项目ID（可选）
        current_user: 当前用户
        image_service: 图像服务

    Returns:
        ImageResponse: 上传的图像信息
    """
    try:
        # 验证图像格式
        if not validate_image_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail="不支持的图像格式。支持的格式：JPG, PNG, WEBP, BMP, TIFF"
            )

        # 读取文件内容
        file_content = await file.read()

        # 验证文件大小
        if len(file_content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB)"
            )

        # 生成唯一文件名
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"

        # 保存文件
        file_path = await save_uploaded_file(
            file_content,
            unique_filename,
            settings.UPLOAD_DIR
        )

        # 获取图像信息
        image_info = await get_image_info(file_path)

        # 创建图像记录
        image_data = ImageCreate(
            filename=file.filename,
            file_path=file_path,
            file_size=len(file_content),
            width=image_info["width"],
            height=image_info["height"],
            format=image_info["format"],
            mode=image_info["mode"],
            project_id=project_id,
            user_id=current_user.id,
            category="source"  # 源图像
        )

        # 保存到数据库
        image = await image_service.create_image(image_data)

        logger.info(f"用户 {current_user.id} 上传图像用于抠图: {file.filename}")

        return ImageResponse.from_orm(image)

    except Exception as e:
        logger.error(f"上传图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail="上传图像失败")


@router.post("/process", response_model=MattingTaskResponse)
async def create_matting_task(
        request: MattingRequest,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    创建抠图任务

    Args:
        request: 抠图请求参数
        background_tasks: 后台任务
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        MattingTaskResponse: 创建的抠图任务信息
    """
    try:
        # 验证抠图参数
        validation_result = await validate_matting_parameters(request)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"参数验证失败: {validation_result.error_message}"
            )

        # 检查图像是否存在
        image = await matting_service.get_image_by_id(request.image_id)
        if not image:
            raise HTTPException(status_code=404, detail="图像不存在")

        # 检查用户权限
        if image.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限访问此图像")

        # 创建抠图任务
        task_data = MattingTaskCreate(
            image_id=request.image_id,
            algorithm=request.algorithm,
            parameters=request.parameters,
            user_id=current_user.id,
            project_id=request.project_id
        )

        task = await matting_service.create_matting_task(task_data)

        # 添加后台任务
        background_tasks.add_task(
            process_matting_task,
            task_id=task.id,
            algorithm=request.algorithm,
            parameters=request.parameters
        )

        logger.info(f"用户 {current_user.id} 创建抠图任务: {task.id}")

        return MattingTaskResponse.from_orm(task)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建抠图任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="创建抠图任务失败")


@router.post("/batch", response_model=List[MattingTaskResponse])
async def create_batch_matting_tasks(
        request: MattingBatchRequest,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    批量创建抠图任务

    Args:
        request: 批量抠图请求参数
        background_tasks: 后台任务
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        List[MattingTaskResponse]: 创建的抠图任务列表
    """
    try:
        # 验证图像数量限制
        if len(request.image_ids) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"批量处理数量超过限制 ({settings.MAX_BATCH_SIZE})"
            )

        # 验证所有图像是否存在且属于当前用户
        images = await matting_service.get_images_by_ids(request.image_ids)
        if len(images) != len(request.image_ids):
            raise HTTPException(status_code=404, detail="部分图像不存在")

        for image in images:
            if image.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="无权限访问部分图像")

        # 创建批量任务
        tasks = []
        for image_id in request.image_ids:
            task_data = MattingTaskCreate(
                image_id=image_id,
                algorithm=request.algorithm,
                parameters=request.parameters,
                user_id=current_user.id,
                project_id=request.project_id
            )
            task = await matting_service.create_matting_task(task_data)
            tasks.append(task)

        # 添加批量后台任务
        background_tasks.add_task(
            process_batch_matting_task,
            task_ids=[task.id for task in tasks],
            algorithm=request.algorithm,
            parameters=request.parameters
        )

        logger.info(f"用户 {current_user.id} 创建批量抠图任务: {len(tasks)} 个任务")

        return [MattingTaskResponse.from_orm(task) for task in tasks]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建批量抠图任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="创建批量抠图任务失败")


@router.post("/preview", response_model=Dict[str, Any])
async def preview_matting_result(
        request: MattingPreviewRequest,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    预览抠图结果（快速处理，低质量）

    Args:
        request: 预览请求参数
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        Dict[str, Any]: 预览结果
    """
    try:
        # 检查图像是否存在
        image = await matting_service.get_image_by_id(request.image_id)
        if not image:
            raise HTTPException(status_code=404, detail="图像不存在")

        # 检查用户权限
        if image.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限访问此图像")

        # 生成预览
        preview_result = await matting_service.generate_preview(
            image_path=image.file_path,
            algorithm=request.algorithm,
            parameters=request.parameters,
            preview_size=request.preview_size
        )

        return {
            "preview_url": preview_result.preview_url,
            "thumbnail_url": preview_result.thumbnail_url,
            "processing_time": preview_result.processing_time,
            "estimated_quality": preview_result.estimated_quality
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail="生成预览失败")


@router.get("/tasks", response_model=List[MattingTaskResponse])
async def get_matting_tasks(
        project_id: Optional[int] = Query(None, description="项目ID"),
        status: Optional[str] = Query(None, description="任务状态"),
        algorithm: Optional[str] = Query(None, description="算法类型"),
        limit: int = Query(20, ge=1, le=100, description="每页数量"),
        offset: int = Query(0, ge=0, description="偏移量"),
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    获取抠图任务列表

    Args:
        project_id: 项目ID（可选）
        status: 任务状态（可选）
        algorithm: 算法类型（可选）
        limit: 每页数量
        offset: 偏移量
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        List[MattingTaskResponse]: 抠图任务列表
    """
    try:
        tasks = await matting_service.get_user_matting_tasks(
            user_id=current_user.id,
            project_id=project_id,
            status=status,
            algorithm=algorithm,
            limit=limit,
            offset=offset
        )

        return [MattingTaskResponse.from_orm(task) for task in tasks]

    except Exception as e:
        logger.error(f"获取抠图任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取抠图任务列表失败")


@router.get("/tasks/{task_id}", response_model=MattingTaskResponse)
async def get_matting_task(
        task_id: int,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    获取指定抠图任务详情

    Args:
        task_id: 任务ID
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        MattingTaskResponse: 抠图任务详情
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限访问此任务")

        return MattingTaskResponse.from_orm(task)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取抠图任务详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取抠图任务详情失败")


@router.get("/tasks/{task_id}/progress", response_model=MattingProgressResponse)
async def get_matting_progress(
        task_id: int,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    获取抠图任务进度

    Args:
        task_id: 任务ID
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        MattingProgressResponse: 任务进度信息
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限访问此任务")

        # 从Redis获取进度信息
        redis_client = await get_redis_client()
        progress_key = f"matting_progress:{task_id}"
        progress_data = await redis_client.get(progress_key)

        if progress_data:
            progress_info = json.loads(progress_data)
        else:
            # 如果Redis中没有进度信息，根据任务状态返回默认进度
            progress_info = {
                "progress": 100 if task.status == "completed" else 0,
                "current_step": task.status,
                "total_steps": ["pending", "processing", "completed"],
                "estimated_time_remaining": 0 if task.status == "completed" else None
            }

        return MattingProgressResponse(
            task_id=task_id,
            status=task.status,
            progress=progress_info["progress"],
            current_step=progress_info["current_step"],
            total_steps=progress_info["total_steps"],
            estimated_time_remaining=progress_info.get("estimated_time_remaining"),
            error_message=task.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取抠图任务进度失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取抠图任务进度失败")


@router.post("/tasks/{task_id}/cancel")
async def cancel_matting_task(
        task_id: int,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    取消抠图任务

    Args:
        task_id: 任务ID
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        Dict[str, str]: 操作结果
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限操作此任务")

        # 检查任务状态
        if task.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail="无法取消已完成、失败或已取消的任务"
            )

        # 取消任务
        await matting_service.cancel_matting_task(task_id)

        logger.info(f"用户 {current_user.id} 取消抠图任务: {task_id}")

        return {"message": "任务已取消"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消抠图任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="取消抠图任务失败")


@router.delete("/tasks/{task_id}")
async def delete_matting_task(
        task_id: int,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    删除抠图任务

    Args:
        task_id: 任务ID
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        Dict[str, str]: 操作结果
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限删除此任务")

        # 删除任务
        await matting_service.delete_matting_task(task_id)

        logger.info(f"用户 {current_user.id} 删除抠图任务: {task_id}")

        return {"message": "任务已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除抠图任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="删除抠图任务失败")


@router.get("/tasks/{task_id}/download")
async def download_matting_result(
        task_id: int,
        format: str = Query("png", description="下载格式"),
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    下载抠图结果

    Args:
        task_id: 任务ID
        format: 下载格式
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        FileResponse: 抠图结果文件
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限下载此文件")

        # 检查任务状态
        if task.status != "completed":
            raise HTTPException(status_code=400, detail="任务未完成，无法下载")

        # 检查结果文件是否存在
        if not task.result_path or not os.path.exists(task.result_path):
            raise HTTPException(status_code=404, detail="结果文件不存在")

        # 根据格式转换文件
        if format.lower() != "png":
            converted_path = await matting_service.convert_result_format(
                task.result_path, format
            )
            file_path = converted_path
        else:
            file_path = task.result_path

        # 生成下载文件名
        original_name = os.path.splitext(task.image.filename)[0]
        download_filename = f"{original_name}_matted.{format.lower()}"

        return FileResponse(
            path=file_path,
            filename=download_filename,
            media_type=f"image/{format.lower()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载抠图结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail="下载抠图结果失败")


@router.post("/tasks/{task_id}/quality", response_model=MattingQualityAssessment)
async def assess_matting_quality(
        task_id: int,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    评估抠图质量

    Args:
        task_id: 任务ID
        background_tasks: 后台任务
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        MattingQualityAssessment: 质量评估结果
    """
    try:
        task = await matting_service.get_matting_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 检查用户权限
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权限访问此任务")

        # 检查任务状态
        if task.status != "completed":
            raise HTTPException(status_code=400, detail="任务未完成，无法评估质量")

        # 添加质量评估后台任务
        background_tasks.add_task(
            assess_matting_quality,
            task_id=task_id
        )

        # 返回评估结果（如果已存在）或启动评估
        quality_assessment = await matting_service.get_quality_assessment(task_id)

        if quality_assessment:
            return quality_assessment
        else:
            # 返回评估中状态
            return MattingQualityAssessment(
                task_id=task_id,
                overall_score=None,
                edge_quality=None,
                detail_preservation=None,
                color_consistency=None,
                assessment_status="assessing"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评估抠图质量失败: {str(e)}")
        raise HTTPException(status_code=500, detail="评估抠图质量失败")


@router.get("/algorithms", response_model=List[Dict[str, Any]])
async def get_available_algorithms():
    """
    获取可用的抠图算法列表

    Returns:
        List[Dict[str, Any]]: 算法列表及其配置
    """
    try:
        algorithms = [
            {
                "name": "sam",
                "display_name": "SAM (Segment Anything Model)",
                "description": "Meta开发的通用分割模型，支持点击、框选等交互方式",
                "parameters": {
                    "points": {
                        "type": "array",
                        "description": "点击坐标列表，格式：[[x1,y1], [x2,y2]]",
                        "required": False
                    },
                    "boxes": {
                        "type": "array",
                        "description": "框选区域列表，格式：[[x1,y1,x2,y2]]",
                        "required": False
                    },
                    "model_type": {
                        "type": "string",
                        "description": "模型类型",
                        "options": ["vit_h", "vit_l", "vit_b"],
                        "default": "vit_h"
                    }
                }
            },
            {
                "name": "u2net",
                "display_name": "U2Net",
                "description": "基于U-Net的深度学习抠图模型，适合复杂场景",
                "parameters": {
                    "input_size": {
                        "type": "integer",
                        "description": "输入图像尺寸",
                        "options": [320, 512, 1024],
                        "default": 320
                    },
                    "refine": {
                        "type": "boolean",
                        "description": "是否使用精细化处理",
                        "default": True
                    }
                }
            },
            {
                "name": "rembg",
                "display_name": "RemBG",
                "description": "快速背景移除工具，支持多种预训练模型",
                "parameters": {
                    "model_name": {
                        "type": "string",
                        "description": "模型名称",
                        "options": ["u2net", "silueta", "isnet-general-use"],
                        "default": "u2net"
                    },
                    "alpha_matting": {
                        "type": "boolean",
                        "description": "是否使用Alpha matting后处理",
                        "default": False
                    }
                }
            }
        ]

        return algorithms

    except Exception as e:
        logger.error(f"获取算法列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取算法列表失败")


@router.get("/statistics", response_model=Dict[str, Any])
async def get_matting_statistics(
        current_user: User = Depends(get_current_user),
        matting_service: MattingService = Depends(get_matting_service)
):
    """
    获取抠图统计信息

    Args:
        current_user: 当前用户
        matting_service: 抠图服务

    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        stats = await matting_service.get_user_statistics(current_user.id)

        return {
            "total_tasks": stats.total_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "processing_tasks": stats.processing_tasks,
            "total_processing_time": stats.total_processing_time,
            "average_processing_time": stats.average_processing_time,
            "algorithm_usage": stats.algorithm_usage,
            "quality_distribution": stats.quality_distribution,
            "monthly_usage": stats.monthly_usage
        }

    except Exception as e:
        logger.error(f"获取抠图统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取抠图统计信息失败")