# backend/app/utils/memory_utils.py
"""
内存管理工具类
实现智能内存监控、清理和优化机制
"""

import asyncio
import gc
import os
import psutil
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from loguru import logger

import numpy as np
import cv2
from PIL import Image

from app.config.settings import settings


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent: float
    process_mb: float
    timestamp: float


class MemoryPool:
    """内存池管理器"""

    def __init__(self, max_size_mb: int = 1024):
        self.max_size_mb = max_size_mb
        self.pools: Dict[str, List[np.ndarray]] = {}
        self.lock = threading.Lock()

    def get_array(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        """获取数组"""
        key = f"{shape}_{dtype}"
        with self.lock:
            if key in self.pools and self.pools[key]:
                return self.pools[key].pop()
            return np.empty(shape, dtype=dtype)

    def return_array(self, arr: np.ndarray):
        """归还数组"""
        if arr is None:
            return

        key = f"{arr.shape}_{arr.dtype}"
        with self.lock:
            if key not in self.pools:
                self.pools[key] = []

            # 限制池大小
            if len(self.pools[key]) < 10:
                arr.fill(0)  # 清零数组
                self.pools[key].append(arr)

    def clear(self):
        """清空内存池"""
        with self.lock:
            self.pools.clear()
            gc.collect()


class AsyncMemoryManager:
    """异步内存管理器"""

    def __init__(self):
        self.cleanup_tasks: List[asyncio.Task] = []
        self.cleanup_callbacks: List[Callable] = []
        self.lock = asyncio.Lock()

    async def register_cleanup_callback(self, callback: Callable):
        """注册清理回调"""
        async with self.lock:
            self.cleanup_callbacks.append(callback)

    async def cleanup(self):
        """执行异步清理"""
        async with self.lock:
            for callback in self.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"清理回调执行失败: {e}")

            # 取消未完成的任务
            for task in self.cleanup_tasks:
                if not task.done():
                    task.cancel()

            self.cleanup_tasks.clear()

    async def add_cleanup_task(self, coro):
        """添加清理任务"""
        task = asyncio.create_task(coro)
        async with self.lock:
            self.cleanup_tasks.append(task)
        return task


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.stats_history: List[MemoryStats] = []
        self.max_history = 100

    async def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("内存监控已启动")

    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("内存监控已停止")

    async def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                stats = self.get_memory_stats()
                self.stats_history.append(stats)

                # 保持历史记录大小
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)

                # 检查内存使用情况
                if stats.percent > settings.MEMORY_WARNING_THRESHOLD * 100:
                    logger.warning(f"内存使用率过高: {stats.percent:.1f}%")

                    if stats.percent > settings.MEMORY_CLEANUP_THRESHOLD * 100:
                        logger.warning("触发自动内存清理")
                        await self._emergency_cleanup()

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                await asyncio.sleep(self.check_interval)

    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()

        return MemoryStats(
            total_mb=memory.total / 1024 / 1024,
            used_mb=memory.used / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            percent=memory.percent,
            process_mb=process_memory.rss / 1024 / 1024,
            timestamp=time.time()
        )

    async def _emergency_cleanup(self):
        """紧急内存清理"""
        try:
            # 强制垃圾回收
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0.1)

            # 清理图像缓存
            if hasattr(cv2, 'setUseOptimized'):
                cv2.setUseOptimized(True)

            logger.info("紧急内存清理完成")

        except Exception as e:
            logger.error(f"紧急内存清理失败: {e}")


class MemoryManager:
    """主内存管理器"""

    _instance: Optional['MemoryManager'] = None
    _lock = threading.Lock()
    _cleanup_count = 0
    _last_cleanup_time = 0

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.memory_pool = MemoryPool(max_size_mb=settings.MEMORY_LIMIT_GB * 1024)
        self.async_manager = AsyncMemoryManager()
        self.monitor = MemoryMonitor(check_interval=30)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")
        self.image_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 100

    async def start_monitoring(self):
        """启动内存监控"""
        await self.monitor.start_monitoring()

        # 注册清理回调
        await self.async_manager.register_cleanup_callback(self.cleanup_image_cache)
        await self.async_manager.register_cleanup_callback(self.cleanup_memory_pool)

        logger.info("内存管理器启动完成")

    async def stop_monitoring(self):
        """停止内存监控"""
        await self.monitor.stop_monitoring()
        await self.async_manager.cleanup()
        self.executor.shutdown(wait=True)
        logger.info("内存管理器已停止")

    @staticmethod
    def get_memory_usage() -> float:
        """获取内存使用率"""
        return psutil.virtual_memory().percent

    @staticmethod
    def get_process_memory() -> float:
        """获取进程内存使用(MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_array(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        """获取数组"""
        return self.memory_pool.get_array(shape, dtype)

    def return_array(self, arr: np.ndarray):
        """归还数组"""
        self.memory_pool.return_array(arr)

    @asynccontextmanager
    async def managed_array(self, shape: tuple, dtype=np.uint8):
        """托管数组上下文管理器"""
        arr = self.get_array(shape, dtype)
        try:
            yield arr
        finally:
            self.return_array(arr)

    def cache_image(self, key: str, image: Any, max_size: Optional[int] = None):
        """缓存图像"""
        max_size = max_size or self.max_cache_size

        with self.cache_lock:
            # 如果缓存已满，移除最旧的
            if len(self.image_cache) >= max_size:
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]

            self.image_cache[key] = {
                'data': image,
                'timestamp': time.time(),
                'access_count': 0
            }

    def get_cached_image(self, key: str) -> Optional[Any]:
        """获取缓存图像"""
        with self.cache_lock:
            if key in self.image_cache:
                self.image_cache[key]['access_count'] += 1
                return self.image_cache[key]['data']
            return None

    async def cleanup_image_cache(self):
        """清理图像缓存"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = []

            for key, info in self.image_cache.items():
                # 清理超过1小时未访问的缓存
                if current_time - info['timestamp'] > 3600:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.image_cache[key]

            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期图像缓存")

    async def cleanup_memory_pool(self):
        """清理内存池"""
        self.memory_pool.clear()
        logger.info("内存池已清理")

    async def force_cleanup(self):
        """强制清理"""
        self.__class__._cleanup_count += 1
        self.__class__._last_cleanup_time = time.time()

        try:
            # 清理缓存
            await self.cleanup_image_cache()
            await self.cleanup_memory_pool()

            # 执行异步清理
            await self.async_manager.cleanup()

            # 强制垃圾回收
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0.1)

            logger.info(f"强制内存清理完成 (第{self._cleanup_count}次)")

        except Exception as e:
            logger.error(f"强制内存清理失败: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        stats = self.monitor.get_memory_stats()
        return {
            'total_mb': stats.total_mb,
            'used_mb': stats.used_mb,
            'available_mb': stats.available_mb,
            'percent': stats.percent,
            'process_mb': stats.process_mb,
            'cache_size': len(self.image_cache),
            'pool_size': sum(len(pool) for pool in self.memory_pool.pools.values()),
            'cleanup_count': self._cleanup_count,
            'last_cleanup': self._last_cleanup_time
        }


class ImageMemoryOptimizer:
    """图像内存优化器"""

    @staticmethod
    def optimize_image_loading(image_path: str, max_size: Optional[tuple] = None) -> np.ndarray:
        """优化图像加载"""
        try:
            # 使用PIL加载图像，内存效率更好
            with Image.open(image_path) as img:
                if max_size:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 转换为numpy数组
                return np.array(img)

        except Exception as e:
            logger.error(f"图像加载优化失败: {e}")
            # 备用方案：使用OpenCV
            return cv2.imread(image_path)

    @staticmethod
    def optimize_image_processing(image: np.ndarray) -> np.ndarray:
        """优化图像处理"""
        # 确保图像是连续的内存布局
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        return image

    @staticmethod
    def batch_process_images(images: List[np.ndarray],
                             process_func: Callable,
                             batch_size: int = 10) -> List[Any]:
        """批量处理图像，优化内存使用"""
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []

            for img in batch:
                result = process_func(img)
                batch_results.append(result)

            results.extend(batch_results)

            # 强制垃圾回收
            gc.collect()

        return results


# 全局内存管理器实例
memory_manager = MemoryManager()


# 便捷函数
async def start_memory_management():
    """启动内存管理"""
    await memory_manager.start_monitoring()


async def stop_memory_management():
    """停止内存管理"""
    await memory_manager.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """获取内存统计"""
    return memory_manager.get_memory_stats()


async def force_memory_cleanup():
    """强制内存清理"""
    await memory_manager.force_cleanup()


# 装饰器
def memory_efficient(func):
    """内存效率装饰器"""
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            finally:
                gc.collect()

        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                gc.collect()

        return sync_wrapper