"""
FastAPI依赖注入管理模块
用于管理数据库连接、Redis连接、用户认证、权限验证等核心依赖
"""

from typing import AsyncGenerator, Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import redis.asyncio as redis
from jose import JWTError, jwt
from datetime import datetime, timedelta
import logging
from functools import wraps
import asyncio
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import async_session_maker, get_sync_session
from app.core.redis import get_redis_client
from app.core.security import verify_token, create_access_token
from app.models.user import User
from app.schemas.user import UserInDB
from app.utils.async_utils import run_in_threadpool

logger = logging.getLogger(__name__)

# HTTPBearer 安全方案
security = HTTPBearer(auto_error=False)


# ================================
# 数据库依赖
# ================================

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取异步数据库会话
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"数据库会话错误: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db() -> Session:
    """
    获取同步数据库会话（用于同步操作）
    """
    with get_sync_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"同步数据库会话错误: {e}")
            session.rollback()
            raise
        finally:
            session.close()


# ================================
# Redis依赖
# ================================

async def get_redis() -> redis.Redis:
    """
    获取Redis连接
    """
    return await get_redis_client()


# ================================
# 认证依赖
# ================================

async def get_current_user_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    获取当前用户的JWT令牌
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证令牌缺失",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


async def get_current_user(
        db: AsyncSession = Depends(get_async_db),
        token: str = Depends(get_current_user_token)
) -> UserInDB:
    """
    获取当前认证用户
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        # 检查令牌是否过期
        exp = payload.get("exp")
        if exp is None or datetime.fromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )

    except JWTError:
        raise credentials_exception

    # 从数据库获取用户信息
    try:
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

        if user is None:
            raise credentials_exception

        # 检查用户是否被禁用
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户账户已被禁用"
            )

        return UserInDB.from_orm(user)

    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        raise credentials_exception


async def get_current_active_user(
        current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    获取当前活跃用户（已验证active状态）
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户未激活"
        )
    return current_user


async def get_current_superuser(
        current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    获取当前超级用户
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足"
        )
    return current_user


# ================================
# 可选认证依赖
# ================================

async def get_current_user_optional(
        db: AsyncSession = Depends(get_async_db),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserInDB]:
    """
    获取当前用户（可选，用于可选认证的接口）
    """
    if not credentials:
        return None

    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            return None

        # 检查令牌是否过期
        exp = payload.get("exp")
        if exp is None or datetime.fromtimestamp(exp) < datetime.utcnow():
            return None

        # 从数据库获取用户信息
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

        if user is None or not user.is_active:
            return None

        return UserInDB.from_orm(user)

    except Exception as e:
        logger.warning(f"可选认证失败: {e}")
        return None


# ================================
# 权限验证依赖
# ================================

def require_permissions(*permissions: str):
    """
    需要特定权限的装饰器依赖
    """

    async def permission_checker(
            current_user: UserInDB = Depends(get_current_active_user)
    ) -> UserInDB:
        if not current_user.is_superuser:
            # 检查用户是否有所需权限
            user_permissions = getattr(current_user, 'permissions', [])
            if not all(perm in user_permissions for perm in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"需要权限: {', '.join(permissions)}"
                )
        return current_user

    return permission_checker


def require_project_access(permission_type: str = "read"):
    """
    需要项目访问权限的装饰器依赖
    """

    async def project_access_checker(
            project_id: int,
            current_user: UserInDB = Depends(get_current_active_user),
            db: AsyncSession = Depends(get_async_db)
    ) -> UserInDB:
        if current_user.is_superuser:
            return current_user

        # 检查用户是否有项目访问权限
        from sqlalchemy import select
        from app.models.project import Project, ProjectMember

        # 检查项目是否存在
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="项目不存在"
            )

        # 检查用户是否是项目所有者
        if project.owner_id == current_user.id:
            return current_user

        # 检查用户是否是项目成员
        result = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == current_user.id
            )
        )
        member = result.scalar_one_or_none()

        if not member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问该项目"
            )

        # 检查权限类型
        if permission_type == "write" and member.role not in ["admin", "editor"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权修改该项目"
            )

        return current_user

    return project_access_checker


# ================================
# 限流依赖
# ================================

class RateLimiter:
    """
    基于Redis的限流器
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def __call__(
            self,
            request: Request,
            redis_client: redis.Redis = Depends(get_redis),
            current_user: Optional[UserInDB] = Depends(get_current_user_optional)
    ):
        # 构建限流键
        if current_user:
            key = f"rate_limit:user:{current_user.id}:{request.url.path}"
        else:
            client_ip = request.client.host
            key = f"rate_limit:ip:{client_ip}:{request.url.path}"

        # 获取当前请求次数
        current_requests = await redis_client.get(key)

        if current_requests is None:
            # 首次请求
            await redis_client.setex(key, self.window_seconds, 1)
        else:
            current_requests = int(current_requests)
            if current_requests >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="请求过于频繁，请稍后再试"
                )

            # 增加请求次数
            await redis_client.incr(key)


# 预定义的限流器
upload_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)  # 上传限流
api_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)  # API限流
heavy_task_rate_limiter = RateLimiter(max_requests=5, window_seconds=300)  # 重任务限流


# ================================
# 缓存依赖
# ================================

async def get_cache_key(
        request: Request,
        current_user: Optional[UserInDB] = Depends(get_current_user_optional)
) -> str:
    """
    生成缓存键
    """
    base_key = f"cache:{request.url.path}:{request.method}"

    # 添加查询参数
    if request.query_params:
        query_str = "&".join([f"{k}={v}" for k, v in request.query_params.items()])
        base_key += f":{query_str}"

    # 添加用户信息
    if current_user:
        base_key += f":user:{current_user.id}"

    return base_key


async def get_cached_response(
        cache_key: str = Depends(get_cache_key),
        redis_client: redis.Redis = Depends(get_redis)
) -> Optional[Dict[str, Any]]:
    """
    获取缓存响应
    """
    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            import json
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.warning(f"获取缓存失败: {e}")
        return None


async def set_cache_response(
        data: Dict[str, Any],
        cache_key: str = Depends(get_cache_key),
        redis_client: redis.Redis = Depends(get_redis),
        ttl: int = 300
):
    """
    设置缓存响应
    """
    try:
        import json
        await redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"设置缓存失败: {e}")


# ================================
# 分页依赖
# ================================

async def get_pagination_params(
        page: int = 1,
        size: int = 20,
        max_size: int = 100
) -> Dict[str, int]:
    """
    获取分页参数
    """
    if page < 1:
        page = 1
    if size < 1:
        size = 20
    if size > max_size:
        size = max_size

    offset = (page - 1) * size

    return {
        "page": page,
        "size": size,
        "offset": offset,
        "limit": size
    }


# ================================
# 文件上传依赖
# ================================

async def validate_file_upload(
        request: Request,
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
        allowed_types: Optional[list] = None
) -> bool:
    """
    验证文件上传
    """
    if allowed_types is None:
        allowed_types = [
            "image/jpeg", "image/png", "image/webp",
            "image/bmp", "image/tiff", "image/gif"
        ]

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超过限制 ({max_file_size / 1024 / 1024:.1f}MB)"
        )

    content_type = request.headers.get("content-type", "")
    if content_type and not any(allowed_type in content_type for allowed_type in allowed_types):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件类型，支持的类型: {', '.join(allowed_types)}"
        )

    return True


# ================================
# 任务状态依赖
# ================================

async def get_task_status(
        task_id: str,
        redis_client: redis.Redis = Depends(get_redis)
) -> Dict[str, Any]:
    """
    获取任务状态
    """
    try:
        import json
        task_data = await redis_client.get(f"task:{task_id}")
        if task_data:
            return json.loads(task_data)
        return {"status": "not_found", "message": "任务不存在"}
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        return {"status": "error", "message": "获取任务状态失败"}


# ================================
# 健康检查依赖
# ================================

async def health_check_db(
        db: AsyncSession = Depends(get_async_db)
) -> Dict[str, str]:
    """
    数据库健康检查
    """
    try:
        await db.execute("SELECT 1")
        return {"database": "healthy"}
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return {"database": "unhealthy"}


async def health_check_redis(
        redis_client: redis.Redis = Depends(get_redis)
) -> Dict[str, str]:
    """
    Redis健康检查
    """
    try:
        await redis_client.ping()
        return {"redis": "healthy"}
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
        return {"redis": "unhealthy"}


# ================================
# 共享依赖组合
# ================================

async def get_common_deps(
        db: AsyncSession = Depends(get_async_db),
        redis_client: redis.Redis = Depends(get_redis),
        current_user: UserInDB = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    获取常用依赖的组合
    """
    return {
        "db": db,
        "redis": redis_client,
        "current_user": current_user
    }


async def get_common_deps_optional(
        db: AsyncSession = Depends(get_async_db),
        redis_client: redis.Redis = Depends(get_redis),
        current_user: Optional[UserInDB] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """
    获取常用依赖的组合（可选用户认证）
    """
    return {
        "db": db,
        "redis": redis_client,
        "current_user": current_user
    }