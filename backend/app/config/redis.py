"""
Redis配置和连接管理模块

提供Redis连接池、异步操作、缓存管理等功能
支持集群模式和单机模式
"""

import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.cluster import RedisCluster
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from loguru import logger

from .config import get_settings


class RedisManager:
    """Redis连接管理器"""

    def __init__(self):
        self.settings = get_settings()
        self._redis_pool: Optional[ConnectionPool] = None
        self._redis_client: Optional[Redis] = None
        self._redis_cluster: Optional[RedisCluster] = None
        self._is_cluster = False

    async def initialize(self):
        """初始化Redis连接"""
        try:
            if self.settings.redis_cluster_enabled:
                await self._initialize_cluster()
            else:
                await self._initialize_single()

            # 测试连接
            await self.ping()
            logger.info("Redis连接初始化成功")

        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            raise

    async def _initialize_single(self):
        """初始化单机Redis连接"""
        self._redis_pool = ConnectionPool(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            encoding='utf-8',
            decode_responses=True,
            max_connections=self.settings.redis_max_connections,
            retry_on_timeout=True,
            socket_connect_timeout=self.settings.redis_socket_timeout,
            socket_timeout=self.settings.redis_socket_timeout,
            health_check_interval=30,
        )

        self._redis_client = Redis(
            connection_pool=self._redis_pool,
            socket_connect_timeout=self.settings.redis_socket_timeout,
            socket_timeout=self.settings.redis_socket_timeout,
        )

    async def _initialize_cluster(self):
        """初始化Redis集群连接"""
        startup_nodes = [
            {"host": node.split(":")[0], "port": int(node.split(":")[1])}
            for node in self.settings.redis_cluster_nodes
        ]

        self._redis_cluster = RedisCluster(
            startup_nodes=startup_nodes,
            password=self.settings.redis_password,
            decode_responses=True,
            skip_full_coverage_check=True,
            max_connections_per_node=self.settings.redis_max_connections_per_node,
            socket_connect_timeout=self.settings.redis_socket_timeout,
            socket_timeout=self.settings.redis_socket_timeout,
            retry_on_timeout=True,
        )

        self._is_cluster = True

    async def close(self):
        """关闭Redis连接"""
        try:
            if self._redis_client:
                await self._redis_client.close()
            if self._redis_cluster:
                await self._redis_cluster.close()
            if self._redis_pool:
                await self._redis_pool.disconnect()
            logger.info("Redis连接已关闭")
        except Exception as e:
            logger.error(f"关闭Redis连接时出错: {e}")

    @property
    def client(self) -> Union[Redis, RedisCluster]:
        """获取Redis客户端"""
        if self._is_cluster:
            return self._redis_cluster
        return self._redis_client

    async def ping(self) -> bool:
        """测试Redis连接"""
        try:
            result = await self.client.ping()
            return result
        except Exception as e:
            logger.error(f"Redis ping失败: {e}")
            return False

    async def get_info(self) -> Dict[str, Any]:
        """获取Redis信息"""
        try:
            return await self.client.info()
        except Exception as e:
            logger.error(f"获取Redis信息失败: {e}")
            return {}


class RedisCache:
    """Redis缓存操作类"""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.client = redis_manager.client

    async def set(
            self,
            key: str,
            value: Any,
            expire: Optional[int] = None,
            serialize: bool = True
    ) -> bool:
        """设置缓存"""
        try:
            if serialize:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                elif not isinstance(value, (str, int, float, bool)):
                    value = pickle.dumps(value)

            result = await self.client.set(key, value, ex=expire)
            return result
        except Exception as e:
            logger.error(f"设置缓存失败 key={key}: {e}")
            return False

    async def get(
            self,
            key: str,
            default: Any = None,
            deserialize: bool = True
    ) -> Any:
        """获取缓存"""
        try:
            value = await self.client.get(key)
            if value is None:
                return default

            if deserialize and isinstance(value, str):
                # 尝试JSON反序列化
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # 尝试pickle反序列化
                    try:
                        return pickle.loads(value.encode())
                    except:
                        return value

            return value
        except Exception as e:
            logger.error(f"获取缓存失败 key={key}: {e}")
            return default

    async def delete(self, *keys: str) -> int:
        """删除缓存"""
        try:
            return await self.client.delete(*keys)
        except Exception as e:
            logger.error(f"删除缓存失败 keys={keys}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            return await self.client.exists(key)
        except Exception as e:
            logger.error(f"检查缓存存在性失败 key={key}: {e}")
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            logger.error(f"设置过期时间失败 key={key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"获取TTL失败 key={key}: {e}")
            return -1

    async def increment(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        try:
            return await self.client.incr(key, amount)
        except Exception as e:
            logger.error(f"递增计数器失败 key={key}: {e}")
            return 0

    async def decrement(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        try:
            return await self.client.decr(key, amount)
        except Exception as e:
            logger.error(f"递减计数器失败 key={key}: {e}")
            return 0


class RedisQueue:
    """Redis队列操作类"""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.client = redis_manager.client

    async def lpush(self, key: str, *values: Any) -> int:
        """从左侧推入队列"""
        try:
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))

            return await self.client.lpush(key, *serialized_values)
        except Exception as e:
            logger.error(f"队列lpush失败 key={key}: {e}")
            return 0

    async def rpush(self, key: str, *values: Any) -> int:
        """从右侧推入队列"""
        try:
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))

            return await self.client.rpush(key, *serialized_values)
        except Exception as e:
            logger.error(f"队列rpush失败 key={key}: {e}")
            return 0

    async def lpop(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """从左侧弹出队列"""
        try:
            value = await self.client.lpop(key)
            if value is None:
                return None

            if deserialize:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            return value
        except Exception as e:
            logger.error(f"队列lpop失败 key={key}: {e}")
            return None

    async def rpop(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """从右侧弹出队列"""
        try:
            value = await self.client.rpop(key)
            if value is None:
                return None

            if deserialize:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            return value
        except Exception as e:
            logger.error(f"队列rpop失败 key={key}: {e}")
            return None

    async def llen(self, key: str) -> int:
        """获取队列长度"""
        try:
            return await self.client.llen(key)
        except Exception as e:
            logger.error(f"获取队列长度失败 key={key}: {e}")
            return 0

    async def lrange(self, key: str, start: int, end: int, deserialize: bool = True) -> List[Any]:
        """获取队列范围内的元素"""
        try:
            values = await self.client.lrange(key, start, end)
            if deserialize:
                result = []
                for value in values:
                    try:
                        result.append(json.loads(value))
                    except json.JSONDecodeError:
                        result.append(value)
                return result
            return values
        except Exception as e:
            logger.error(f"获取队列范围失败 key={key}: {e}")
            return []


class RedisSession:
    """Redis会话管理类"""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.client = redis_manager.client
        self.session_prefix = "session:"
        self.default_expire = 3600  # 1小时

    async def create_session(self, session_id: str, data: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """创建会话"""
        try:
            key = f"{self.session_prefix}{session_id}"
            expire_time = expire or self.default_expire

            session_data = {
                "data": data,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat()
            }

            return await self.client.setex(
                key,
                expire_time,
                json.dumps(session_data, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"创建会话失败 session_id={session_id}: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        try:
            key = f"{self.session_prefix}{session_id}"
            value = await self.client.get(key)

            if value is None:
                return None

            session_data = json.loads(value)

            # 更新最后访问时间
            session_data["last_accessed"] = datetime.now().isoformat()
            await self.client.set(key, json.dumps(session_data, ensure_ascii=False))

            return session_data.get("data")
        except Exception as e:
            logger.error(f"获取会话失败 session_id={session_id}: {e}")
            return None

    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话"""
        try:
            key = f"{self.session_prefix}{session_id}"
            existing_data = await self.client.get(key)

            if existing_data is None:
                return False

            session_data = json.loads(existing_data)
            session_data["data"].update(data)
            session_data["last_accessed"] = datetime.now().isoformat()

            # 保持原有的过期时间
            ttl = await self.client.ttl(key)
            if ttl > 0:
                return await self.client.setex(
                    key,
                    ttl,
                    json.dumps(session_data, ensure_ascii=False)
                )
            else:
                return await self.client.set(
                    key,
                    json.dumps(session_data, ensure_ascii=False)
                )
        except Exception as e:
            logger.error(f"更新会话失败 session_id={session_id}: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            key = f"{self.session_prefix}{session_id}"
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"删除会话失败 session_id={session_id}: {e}")
            return False

    async def refresh_session(self, session_id: str, expire: Optional[int] = None) -> bool:
        """刷新会话过期时间"""
        try:
            key = f"{self.session_prefix}{session_id}"
            expire_time = expire or self.default_expire
            return await self.client.expire(key, expire_time)
        except Exception as e:
            logger.error(f"刷新会话失败 session_id={session_id}: {e}")
            return False


class RedisLock:
    """Redis分布式锁"""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.client = redis_manager.client

    @asynccontextmanager
    async def acquire_lock(
            self,
            key: str,
            expire: int = 60,
            timeout: int = 10,
            identifier: Optional[str] = None
    ):
        """获取分布式锁"""
        if identifier is None:
            identifier = f"{asyncio.current_task().get_name()}:{id(asyncio.current_task())}"

        lock_key = f"lock:{key}"
        acquired = False

        try:
            # 尝试获取锁
            start_time = asyncio.get_event_loop().time()
            while True:
                if await self.client.set(lock_key, identifier, nx=True, ex=expire):
                    acquired = True
                    break

                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(f"获取锁超时: {key}")

                await asyncio.sleep(0.1)

            yield

        finally:
            if acquired:
                # 释放锁 - 使用Lua脚本保证原子性
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                await self.client.eval(lua_script, 1, lock_key, identifier)


# 全局Redis管理器实例
redis_manager = RedisManager()
redis_cache = RedisCache(redis_manager)
redis_queue = RedisQueue(redis_manager)
redis_session = RedisSession(redis_manager)
redis_lock = RedisLock(redis_manager)


async def get_redis_manager() -> RedisManager:
    """获取Redis管理器"""
    return redis_manager


async def get_redis_cache() -> RedisCache:
    """获取Redis缓存"""
    return redis_cache


async def get_redis_queue() -> RedisQueue:
    """获取Redis队列"""
    return redis_queue


async def get_redis_session() -> RedisSession:
    """获取Redis会话管理"""
    return redis_session


async def get_redis_lock() -> RedisLock:
    """获取Redis分布式锁"""
    return redis_lock


async def init_redis():
    """初始化Redis连接"""
    await redis_manager.initialize()


async def close_redis():
    """关闭Redis连接"""
    await redis_manager.close()


# 健康检查函数
async def redis_health_check() -> Dict[str, Any]:
    """Redis健康检查"""
    try:
        # 基本连接检查
        ping_result = await redis_manager.ping()

        # 获取Redis信息
        info = await redis_manager.get_info()

        # 性能测试
        start_time = asyncio.get_event_loop().time()
        await redis_cache.set("health_check", "test", expire=10)
        await redis_cache.get("health_check")
        await redis_cache.delete("health_check")
        response_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return {
            "status": "healthy" if ping_result else "unhealthy",
            "ping": ping_result,
            "response_time_ms": round(response_time, 2),
            "memory_usage": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "uptime_seconds": info.get("uptime_in_seconds", 0)
        }
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "ping": False,
            "response_time_ms": 0
        }


# 缓存装饰器
def cache_result(
        key_prefix: str = "",
        expire: int = 3600,
        serialize: bool = True
):
    """缓存结果装饰器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 尝试从缓存获取
            cached_result = await redis_cache.get(cache_key, deserialize=serialize)
            if cached_result is not None:
                return cached_result

            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await redis_cache.set(cache_key, result, expire=expire, serialize=serialize)

            return result

        return wrapper

    return decorator