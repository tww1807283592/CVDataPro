"""
数据库配置模块
支持异步PostgreSQL操作和连接池管理
"""

import asyncio
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from urllib.parse import quote_plus

from sqlalchemy import event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    async_scoped_session
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError
from sqlalchemy import create_engine

from .config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy基础模型类"""
    pass


class DatabaseConfig:
    """数据库配置类"""

    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self.sync_engine: Optional[Engine] = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self.async_scoped_session_factory: Optional[async_scoped_session] = None
        self._is_initialized = False

    def get_database_url(self, async_driver: bool = True) -> str:
        """构建数据库连接URL"""
        # URL编码用户名和密码中的特殊字符
        username = quote_plus(settings.DATABASE_USERNAME)
        password = quote_plus(settings.DATABASE_PASSWORD)

        # 根据是否异步选择驱动
        if async_driver:
            driver = "postgresql+asyncpg"
        else:
            driver = "postgresql+psycopg2"

        return (
            f"{driver}://{username}:{password}@"
            f"{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/"
            f"{settings.DATABASE_NAME}"
        )

    def create_async_engine(self) -> AsyncEngine:
        """创建异步数据库引擎"""
        database_url = self.get_database_url(async_driver=True)

        # 引擎配置
        engine_config = {
            "url": database_url,
            "echo": settings.DATABASE_ECHO,
            "echo_pool": settings.DATABASE_ECHO_POOL,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
            "poolclass": QueuePool,
            "connect_args": {
                "server_settings": {
                    "application_name": settings.APP_NAME,
                    "jit": "off",  # 禁用JIT以提高连接速度
                },
                "command_timeout": settings.DATABASE_COMMAND_TIMEOUT,
                "statement_cache_size": 0,  # 禁用语句缓存
            },
        }

        # 开发环境特殊配置
        if settings.ENVIRONMENT == "development":
            engine_config["echo"] = True
            engine_config["echo_pool"] = True

        # 生产环境特殊配置
        elif settings.ENVIRONMENT == "production":
            engine_config["echo"] = False
            engine_config["echo_pool"] = False
            engine_config["pool_size"] = max(engine_config["pool_size"], 20)
            engine_config["max_overflow"] = max(engine_config["max_overflow"], 40)

        return create_async_engine(**engine_config)

    def create_sync_engine(self) -> Engine:
        """创建同步数据库引擎（用于迁移等操作）"""
        database_url = self.get_database_url(async_driver=False)

        engine_config = {
            "url": database_url,
            "echo": settings.DATABASE_ECHO,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
            "poolclass": QueuePool,
            "connect_args": {
                "application_name": settings.APP_NAME,
            },
        }

        return create_engine(**engine_config)

    async def initialize(self):
        """初始化数据库连接"""
        if self._is_initialized:
            return

        try:
            # 创建异步引擎
            self.async_engine = self.create_async_engine()

            # 创建同步引擎
            self.sync_engine = self.create_sync_engine()

            # 创建会话工厂
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )

            # 创建作用域会话工厂
            self.async_scoped_session_factory = async_scoped_session(
                self.async_session_factory,
                scopefunc=asyncio.current_task,
            )

            # 添加事件监听器
            self._setup_event_listeners()

            # 测试连接
            await self.test_connection()

            self._is_initialized = True
            logger.info("数据库连接初始化成功")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def _setup_event_listeners(self):
        """设置数据库事件监听器"""

        @event.listens_for(self.async_engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """连接时设置PostgreSQL参数"""
            if hasattr(dbapi_connection, 'execute'):
                # 设置连接参数
                dbapi_connection.execute("SET timezone TO 'UTC'")
                dbapi_connection.execute("SET statement_timeout TO '300s'")
                dbapi_connection.execute("SET lock_timeout TO '30s'")

        @event.listens_for(self.async_engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出时的处理"""
            logger.debug("数据库连接检出")

        @event.listens_for(self.async_engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接检入时的处理"""
            logger.debug("数据库连接检入")

        @event.listens_for(self.async_engine.sync_engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """连接失效时的处理"""
            logger.warning(f"数据库连接失效: {exception}")

    async def test_connection(self):
        """测试数据库连接"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                row = result.fetchone()
                assert row[0] == 1
                logger.info("数据库连接测试成功")
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            raise

    async def close(self):
        """关闭数据库连接"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("异步数据库引擎已关闭")

        if self.sync_engine:
            self.sync_engine.dispose()
            logger.info("同步数据库引擎已关闭")

        self._is_initialized = False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话上下文管理器"""
        if not self._is_initialized:
            await self.initialize()

        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"数据库会话错误: {e}")
                raise
            finally:
                await session.close()

    async def get_scoped_session(self) -> AsyncSession:
        """获取作用域会话"""
        if not self._is_initialized:
            await self.initialize()

        return self.async_scoped_session_factory()

    async def create_all_tables(self):
        """创建所有表"""
        if not self._is_initialized:
            await self.initialize()

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("所有数据表创建成功")

    async def drop_all_tables(self):
        """删除所有表"""
        if not self._is_initialized:
            await self.initialize()

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("所有数据表删除成功")

    async def get_connection_info(self) -> dict:
        """获取连接信息"""
        if not self.async_engine:
            return {"status": "未初始化"}

        pool = self.async_engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "status": "已连接" if self._is_initialized else "未连接",
        }


# 全局数据库配置实例
database_config = DatabaseConfig()


# 便捷函数
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（依赖注入使用）"""
    async with database_config.get_session() as session:
        yield session


async def get_db() -> AsyncSession:
    """获取数据库会话（直接使用）"""
    return await database_config.get_scoped_session()


async def init_database():
    """初始化数据库"""
    await database_config.initialize()


async def close_database():
    """关闭数据库连接"""
    await database_config.close()


async def create_tables():
    """创建数据表"""
    await database_config.create_all_tables()


async def drop_tables():
    """删除数据表"""
    await database_config.drop_all_tables()


# 数据库健康检查
async def check_database_health() -> dict:
    """检查数据库健康状态"""
    try:
        if not database_config._is_initialized:
            return {"status": "error", "message": "数据库未初始化"}

        # 测试连接
        async with database_config.get_session() as session:
            result = await session.execute("SELECT 1")
            row = result.fetchone()

            if row and row[0] == 1:
                connection_info = await database_config.get_connection_info()
                return {
                    "status": "healthy",
                    "message": "数据库连接正常",
                    "connection_info": connection_info,
                }
            else:
                return {"status": "error", "message": "数据库查询失败"}

    except DisconnectionError:
        return {"status": "error", "message": "数据库连接断开"}
    except OperationalError as e:
        return {"status": "error", "message": f"数据库操作错误: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"数据库未知错误: {str(e)}"}


# 数据库事务装饰器
def db_transaction(func):
    """数据库事务装饰器"""

    async def wrapper(*args, **kwargs):
        async with database_config.get_session() as session:
            try:
                # 将session注入到函数参数中
                if 'session' in func.__code__.co_varnames:
                    kwargs['session'] = session

                result = await func(*args, **kwargs)
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                logger.error(f"事务执行失败: {e}")
                raise

    return wrapper


# 数据库迁移相关
def get_sync_engine():
    """获取同步引擎（用于Alembic迁移）"""
    return database_config.sync_engine or database_config.create_sync_engine()


def get_sync_session():
    """获取同步会话（用于迁移脚本）"""
    from sqlalchemy.orm import sessionmaker

    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# 导出Base类供模型使用
__all__ = [
    "Base",
    "database_config",
    "get_db_session",
    "get_db",
    "init_database",
    "close_database",
    "create_tables",
    "drop_tables",
    "check_database_health",
    "db_transaction",
    "get_sync_engine",
    "get_sync_session",
]