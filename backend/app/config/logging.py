"""
日志配置模块
基于loguru的高性能异步日志系统
支持多种日志级别、格式化、轮转和监控功能
"""

import sys
import os
import json
import asyncio
import traceback
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from contextvars import ContextVar
from dataclasses import dataclass, asdict

from loguru import logger
from pydantic import BaseSettings, Field

# 上下文变量，用于存储请求相关信息
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


@dataclass
class LogContext:
    """日志上下文信息"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    resource: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class LogSettings(BaseSettings):
    """日志配置"""

    # 基础配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="detailed", description="日志格式: simple, detailed, json")
    log_colorize: bool = Field(default=True, description="是否启用颜色输出")

    # 文件配置
    log_dir: str = Field(default="logs", description="日志文件目录")
    log_file_max_size: str = Field(default="10 MB", description="单个日志文件最大大小")
    log_file_retention: str = Field(default="30 days", description="日志文件保留时间")
    log_file_compression: str = Field(default="gz", description="日志文件压缩格式")
    log_file_encoding: str = Field(default="utf-8", description="日志文件编码")

    # 异步配置
    log_async_enabled: bool = Field(default=True, description="是否启用异步日志")
    log_async_buffer_size: int = Field(default=1000, description="异步日志缓冲区大小")
    log_async_flush_interval: float = Field(default=1.0, description="异步日志刷新间隔(秒)")

    # 监控配置
    log_metrics_enabled: bool = Field(default=True, description="是否启用日志指标")
    log_alert_enabled: bool = Field(default=True, description="是否启用日志告警")
    log_alert_error_threshold: int = Field(default=10, description="错误日志告警阈值")
    log_alert_critical_threshold: int = Field(default=5, description="严重错误日志告警阈值")

    # 性能配置
    log_performance_enabled: bool = Field(default=True, description="是否启用性能日志")
    log_slow_query_threshold: float = Field(default=1.0, description="慢查询阈值(秒)")
    log_memory_threshold: int = Field(default=500, description="内存使用告警阈值(MB)")

    # 安全配置
    log_sensitive_data_mask: bool = Field(default=True, description="是否屏蔽敏感数据")
    log_max_line_length: int = Field(default=8192, description="单行日志最大长度")

    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class LogMetrics:
    """日志指标统计"""

    def __init__(self):
        self.counters: Dict[str, int] = {
            'debug': 0,
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        self.start_time = datetime.now(timezone.utc)
        self.last_reset = self.start_time

    def increment(self, level: str):
        """增加计数"""
        level_lower = level.lower()
        if level_lower in self.counters:
            self.counters[level_lower] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        now = datetime.now(timezone.utc)
        uptime = (now - self.start_time).total_seconds()

        return {
            'counters': self.counters.copy(),
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat(),
            'last_reset': self.last_reset.isoformat(),
            'total_logs': sum(self.counters.values()),
            'error_rate': self.counters['error'] / max(sum(self.counters.values()), 1),
            'critical_rate': self.counters['critical'] / max(sum(self.counters.values()), 1)
        }

    def reset(self):
        """重置统计"""
        self.counters = {k: 0 for k in self.counters}
        self.last_reset = datetime.now(timezone.utc)


class AsyncLogHandler:
    """异步日志处理器"""

    def __init__(self, settings: LogSettings):
        self.settings = settings
        self.buffer: List[str] = []
        self.buffer_lock = asyncio.Lock()
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """启动异步处理"""
        if self.settings.log_async_enabled and not self.running:
            self.running = True
            self.flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """停止异步处理"""
        if self.running:
            self.running = False
            if self.flush_task:
                self.flush_task.cancel()
                try:
                    await self.flush_task
                except asyncio.CancelledError:
                    pass
            await self._flush_buffer()

    async def add_log(self, record: str):
        """添加日志记录"""
        if not self.settings.log_async_enabled:
            return

        async with self.buffer_lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.settings.log_async_buffer_size:
                await self._flush_buffer()

    async def _flush_loop(self):
        """异步刷新循环"""
        while self.running:
            try:
                await asyncio.sleep(self.settings.log_async_flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 避免日志处理器本身出错导致的无限循环
                print(f"Log flush error: {e}", file=sys.stderr)

    async def _flush_buffer(self):
        """刷新缓冲区"""
        async with self.buffer_lock:
            if self.buffer:
                # 这里可以实现批量写入日志文件或发送到远程服务
                # 目前简单清空缓冲区
                self.buffer.clear()


class LogFormatter:
    """日志格式化器"""

    def __init__(self, settings: LogSettings):
        self.settings = settings
        self.sensitive_fields = {
            'password', 'token', 'secret', 'key', 'auth', 'credential',
            'passwd', 'pwd', 'api_key', 'access_token', 'refresh_token'
        }

    def format_record(self, record: Dict[str, Any]) -> str:
        """格式化日志记录"""
        if self.settings.log_format == "json":
            return self._format_json(record)
        elif self.settings.log_format == "simple":
            return self._format_simple(record)
        else:  # detailed
            return self._format_detailed(record)

    def _format_json(self, record: Dict[str, Any]) -> str:
        """JSON格式"""
        log_data = {
            'timestamp': record['time'].isoformat(),
            'level': record['level'].name,
            'message': record['message'],
            'module': record['name'],
            'function': record['function'],
            'line': record['line'],
            'thread': record['thread'].name,
            'process': record['process'].name,
        }

        # 添加上下文信息
        context = self._get_context()
        if context:
            log_data['context'] = context

        # 添加额外字段
        if 'extra' in record:
            log_data['extra'] = self._mask_sensitive_data(record['extra'])

        return json.dumps(log_data, ensure_ascii=False)

    def _format_simple(self, record: Dict[str, Any]) -> str:
        """简单格式"""
        timestamp = record['time'].strftime('%Y-%m-%d %H:%M:%S')
        level = record['level'].name
        message = record['message']

        context = self._get_context()
        if context and context.get('request_id'):
            return f"[{timestamp}] [{level}] [{context['request_id']}] {message}"
        else:
            return f"[{timestamp}] [{level}] {message}"

    def _format_detailed(self, record: Dict[str, Any]) -> str:
        """详细格式"""
        timestamp = record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record['level'].name
        module = record['name']
        function = record['function']
        line = record['line']
        message = record['message']

        base_info = f"[{timestamp}] [{level}] [{module}:{function}:{line}]"

        # 添加上下文信息
        context = self._get_context()
        if context:
            context_parts = []
            if context.get('request_id'):
                context_parts.append(f"req:{context['request_id']}")
            if context.get('user_id'):
                context_parts.append(f"user:{context['user_id']}")
            if context.get('operation'):
                context_parts.append(f"op:{context['operation']}")

            if context_parts:
                base_info += f" [{'/'.join(context_parts)}]"

        return f"{base_info} {message}"

    def _get_context(self) -> Optional[Dict[str, Any]]:
        """获取当前上下文"""
        context = {}

        if request_id := request_id_var.get():
            context['request_id'] = request_id
        if user_id := user_id_var.get():
            context['user_id'] = user_id
        if session_id := session_id_var.get():
            context['session_id'] = session_id

        return context if context else None

    def _mask_sensitive_data(self, data: Any) -> Any:
        """屏蔽敏感数据"""
        if not self.settings.log_sensitive_data_mask:
            return data

        if isinstance(data, dict):
            return {
                k: self._mask_value(k, v) for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data

    def _mask_value(self, key: str, value: Any) -> Any:
        """屏蔽敏感值"""
        if isinstance(key, str) and any(
                sensitive in key.lower() for sensitive in self.sensitive_fields
        ):
            if isinstance(value, str) and len(value) > 4:
                return f"{value[:2]}***{value[-2:]}"
            else:
                return "***"
        return self._mask_sensitive_data(value)


class LoggingManager:
    """日志管理器"""

    def __init__(self, settings: Optional[LogSettings] = None):
        self.settings = settings or LogSettings()
        self.metrics = LogMetrics()
        self.formatter = LogFormatter(self.settings)
        self.async_handler = AsyncLogHandler(self.settings)
        self.initialized = False

    def setup_logging(self):
        """设置日志系统"""
        if self.initialized:
            return

        # 移除默认处理器
        logger.remove()

        # 添加控制台处理器
        self._setup_console_handler()

        # 添加文件处理器
        self._setup_file_handlers()

        # 设置日志级别
        logger.configure(
            handlers=[],
            extra={"formatter": self.formatter}
        )

        # 添加自定义处理器
        self._setup_custom_handlers()

        self.initialized = True

    def _setup_console_handler(self):
        """设置控制台处理器"""
        format_str = self._get_console_format()

        logger.add(
            sys.stdout,
            format=format_str,
            level=self.settings.log_level,
            colorize=self.settings.log_colorize,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )

    def _setup_file_handlers(self):
        """设置文件处理器"""
        log_dir = Path(self.settings.log_dir)
        log_dir.mkdir(exist_ok=True)

        # 通用日志文件
        logger.add(
            log_dir / "app.log",
            format=self._get_file_format(),
            level=self.settings.log_level,
            rotation=self.settings.log_file_max_size,
            retention=self.settings.log_file_retention,
            compression=self.settings.log_file_compression,
            encoding=self.settings.log_file_encoding,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )

        # 错误日志文件
        logger.add(
            log_dir / "error.log",
            format=self._get_file_format(),
            level="ERROR",
            rotation=self.settings.log_file_max_size,
            retention=self.settings.log_file_retention,
            compression=self.settings.log_file_compression,
            encoding=self.settings.log_file_encoding,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )

        # 访问日志文件
        logger.add(
            log_dir / "access.log",
            format=self._get_access_format(),
            level="INFO",
            rotation=self.settings.log_file_max_size,
            retention=self.settings.log_file_retention,
            compression=self.settings.log_file_compression,
            encoding=self.settings.log_file_encoding,
            filter=lambda record: record["extra"].get("log_type") == "access",
            enqueue=True
        )

    def _setup_custom_handlers(self):
        """设置自定义处理器"""
        # 添加指标收集处理器
        if self.settings.log_metrics_enabled:
            logger.add(
                self._metrics_sink,
                format="{message}",
                level="DEBUG",
                enqueue=True
            )

    def _get_console_format(self) -> str:
        """获取控制台格式"""
        if self.settings.log_format == "json":
            return "{extra[formatter].format_record(record)}"
        elif self.settings.log_format == "simple":
            return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
        else:  # detailed
            return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {extra[request_id]} | <level>{message}</level>"

    def _get_file_format(self) -> str:
        """获取文件格式"""
        if self.settings.log_format == "json":
            return "{extra[formatter].format_record(record)}"
        else:
            return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {message}"

    def _get_access_format(self) -> str:
        """获取访问日志格式"""
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"

    def _metrics_sink(self, record):
        """指标收集处理器"""
        self.metrics.increment(record["level"].name)

        # 错误告警
        if self.settings.log_alert_enabled:
            if record["level"].name == "ERROR":
                self._check_error_threshold()
            elif record["level"].name == "CRITICAL":
                self._check_critical_threshold()

    def _check_error_threshold(self):
        """检查错误阈值"""
        if self.metrics.counters['error'] >= self.settings.log_alert_error_threshold:
            # 这里可以实现告警逻辑，例如发送邮件或通知
            pass

    def _check_critical_threshold(self):
        """检查严重错误阈值"""
        if self.metrics.counters['critical'] >= self.settings.log_alert_critical_threshold:
            # 这里可以实现告警逻辑，例如发送邮件或通知
            pass

    async def start_async_logging(self):
        """启动异步日志"""
        await self.async_handler.start()

    async def stop_async_logging(self):
        """停止异步日志"""
        await self.async_handler.stop()

    def get_metrics(self) -> Dict[str, Any]:
        """获取日志指标"""
        return self.metrics.get_stats()

    def reset_metrics(self):
        """重置指标"""
        self.metrics.reset()


# 全局日志管理器实例
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """获取日志管理器实例"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
        _logging_manager.setup_logging()
    return _logging_manager


def setup_logging(settings: Optional[LogSettings] = None):
    """设置日志系统"""
    global _logging_manager
    _logging_manager = LoggingManager(settings)
    _logging_manager.setup_logging()


def set_log_context(
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
):
    """设置日志上下文"""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_log_context():
    """清除日志上下文"""
    request_id_var.set(None)
    user_id_var.set(None)
    session_id_var.set(None)


def log_performance(operation: str, threshold: Optional[float] = None):
    """性能日志装饰器"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()

                settings = get_logging_manager().settings
                log_threshold = threshold or settings.log_slow_query_threshold

                if duration > log_threshold:
                    logger.warning(
                        f"Slow operation: {operation} took {duration:.3f}s",
                        extra={
                            "log_type": "performance",
                            "operation": operation,
                            "duration": duration,
                            "threshold": log_threshold
                        }
                    )
                else:
                    logger.info(
                        f"Operation completed: {operation} took {duration:.3f}s",
                        extra={
                            "log_type": "performance",
                            "operation": operation,
                            "duration": duration
                        }
                    )

                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Operation failed: {operation} after {duration:.3f}s - {str(e)}",
                    extra={
                        "log_type": "performance",
                        "operation": operation,
                        "duration": duration,
                        "error": str(e)
                    }
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()

                settings = get_logging_manager().settings
                log_threshold = threshold or settings.log_slow_query_threshold

                if duration > log_threshold:
                    logger.warning(
                        f"Slow operation: {operation} took {duration:.3f}s",
                        extra={
                            "log_type": "performance",
                            "operation": operation,
                            "duration": duration,
                            "threshold": log_threshold
                        }
                    )
                else:
                    logger.info(
                        f"Operation completed: {operation} took {duration:.3f}s",
                        extra={
                            "log_type": "performance",
                            "operation": operation,
                            "duration": duration
                        }
                    )

                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Operation failed: {operation} after {duration:.3f}s - {str(e)}",
                    extra={
                        "log_type": "performance",
                        "operation": operation,
                        "duration": duration,
                        "error": str(e)
                    }
                )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def log_access(
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
):
    """记录访问日志"""
    logger.info(
        f"{method} {path} {status_code} {duration:.3f}s",
        extra={
            "log_type": "access",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent
        }
    )


def log_security_event(
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
):
    """记录安全事件"""
    log_level = getattr(logger, severity.lower(), logger.warning)

    log_level(
        f"Security event: {event_type} - {description}",
        extra={
            "log_type": "security",
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "metadata": metadata or {}
        }
    )


# 导出常用的logger实例
__all__ = [
    'logger',
    'LogSettings',
    'LoggingManager',
    'LogContext',
    'setup_logging',
    'get_logging_manager',
    'set_log_context',
    'clear_log_context',
    'log_performance',
    'log_access',
    'log_security_event'
]