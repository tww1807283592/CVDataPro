"""
Celery异步任务配置模块

该模块配置了Celery应用程序，用于处理图像处理、抠图、合成等耗时任务。
支持Redis作为消息代理和结果后端，提供任务监控和错误处理功能。
"""

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from kombu import Queue, Exchange
from datetime import timedelta
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取配置
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TIMEZONE = os.getenv('CELERY_TIMEZONE', 'UTC')

# 创建Celery应用实例
celery_app = Celery(
    'image_synthesis_platform',
    broker=REDIS_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.matting_tasks',
        'app.tasks.synthesis_tasks',
        'app.tasks.dataset_tasks',
    ]
)

# 配置交换机和队列
default_exchange = Exchange('default', type='direct')
high_priority_exchange = Exchange('high_priority', type='direct')
low_priority_exchange = Exchange('low_priority', type='direct')

# Celery配置
celery_app.conf.update(
    # 时区设置
    timezone=CELERY_TIMEZONE,
    enable_utc=True,

    # 任务序列化
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # 结果过期时间
    result_expires=timedelta(hours=24),

    # 任务路由配置
    task_routes={
        # 抠图任务 - 高优先级队列
        'app.tasks.matting_tasks.process_matting': {
            'queue': 'high_priority',
            'exchange': 'high_priority',
            'routing_key': 'high_priority'
        },
        'app.tasks.matting_tasks.batch_matting': {
            'queue': 'high_priority',
            'exchange': 'high_priority',
            'routing_key': 'high_priority'
        },

        # 合成任务 - 默认队列
        'app.tasks.synthesis_tasks.process_synthesis': {
            'queue': 'default',
            'exchange': 'default',
            'routing_key': 'default'
        },
        'app.tasks.synthesis_tasks.batch_synthesis': {
            'queue': 'default',
            'exchange': 'default',
            'routing_key': 'default'
        },

        # 数据集任务 - 低优先级队列
        'app.tasks.dataset_tasks.generate_dataset': {
            'queue': 'low_priority',
            'exchange': 'low_priority',
            'routing_key': 'low_priority'
        },
        'app.tasks.dataset_tasks.export_dataset': {
            'queue': 'low_priority',
            'exchange': 'low_priority',
            'routing_key': 'low_priority'
        },
    },

    # 队列配置
    task_queues=(
        Queue('high_priority',
              exchange=high_priority_exchange,
              routing_key='high_priority',
              queue_arguments={'x-max-priority': 10}),
        Queue('default',
              exchange=default_exchange,
              routing_key='default',
              queue_arguments={'x-max-priority': 5}),
        Queue('low_priority',
              exchange=low_priority_exchange,
              routing_key='low_priority',
              queue_arguments={'x-max-priority': 1}),
    ),

    # 默认队列
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',

    # Worker配置
    worker_prefetch_multiplier=1,  # 每个worker一次只处理一个任务
    task_acks_late=True,  # 任务完成后才确认
    task_reject_on_worker_lost=True,  # Worker丢失时拒绝任务

    # 任务时间限制
    task_time_limit=30 * 60,  # 30分钟硬时间限制
    task_soft_time_limit=25 * 60,  # 25分钟软时间限制

    # 重试配置
    task_default_retry_delay=60,  # 重试延迟60秒
    task_max_retries=3,  # 最多重试3次

    # 结果后端配置
    result_backend_transport_options={
        'visibility_timeout': 3600,  # 结果可见性超时
        'retry_policy': {
            'timeout': 5.0
        }
    },

    # 监控配置
    worker_send_task_events=True,
    task_send_sent_event=True,

    # 内存管理
    worker_max_tasks_per_child=1000,  # 每个子进程最多处理1000个任务后重启
    worker_max_memory_per_child=200000,  # 每个子进程最大内存200MB

    # 日志配置
    worker_hijack_root_logger=False,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',

    # 安全配置
    worker_disable_rate_limits=False,
    task_always_eager=False,  # 生产环境设置为False

    # 自定义配置
    task_annotations={
        '*': {'rate_limit': '10/s'},  # 全局速率限制
        'app.tasks.matting_tasks.process_matting': {'rate_limit': '5/s'},
        'app.tasks.synthesis_tasks.process_synthesis': {'rate_limit': '8/s'},
        'app.tasks.dataset_tasks.generate_dataset': {'rate_limit': '2/s'},
    }
)

# 任务状态配置
TASK_STATES = {
    'PENDING': 'pending',
    'STARTED': 'started',
    'SUCCESS': 'success',
    'FAILURE': 'failure',
    'RETRY': 'retry',
    'REVOKED': 'revoked',
}


# 自定义任务基类
class BaseTask(celery_app.Task):
    """自定义任务基类，提供通用的错误处理和日志记录"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败时的回调"""
        logger.error(f'Task {task_id} failed: {exc}')
        logger.error(f'Task info: {einfo}')

        # 这里可以添加失败通知逻辑，比如发送邮件或更新数据库状态
        # 例如：notify_task_failure(task_id, exc, args, kwargs)

    def on_success(self, retval, task_id, args, kwargs):
        """任务成功时的回调"""
        logger.info(f'Task {task_id} succeeded with result: {retval}')

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """任务重试时的回调"""
        logger.warning(f'Task {task_id} retry: {exc}')


# 设置默认任务基类
celery_app.Task = BaseTask


# Worker事件处理
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Worker准备就绪时的处理"""
    logger.info(f'Worker {sender} is ready')


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Worker关闭时的处理"""
    logger.info(f'Worker {sender} is shutting down')


# 健康检查任务
@celery_app.task(bind=True, name='health_check')
def health_check(self):
    """健康检查任务"""
    return {
        'status': 'healthy',
        'worker_id': self.request.id,
        'timestamp': self.request.called_directly
    }


# 获取任务状态的辅助函数
def get_task_info(task_id: str) -> Dict[str, Any]:
    """获取任务详细信息"""
    task_result = celery_app.AsyncResult(task_id)

    return {
        'task_id': task_id,
        'status': task_result.status,
        'result': task_result.result,
        'info': task_result.info,
        'traceback': task_result.traceback,
        'successful': task_result.successful(),
        'failed': task_result.failed(),
        'ready': task_result.ready(),
    }


# 取消任务的辅助函数
def cancel_task(task_id: str, terminate: bool = False) -> bool:
    """取消任务"""
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info(f'Task {task_id} cancelled successfully')
        return True
    except Exception as e:
        logger.error(f'Failed to cancel task {task_id}: {e}')
        return False


# 获取队列信息的辅助函数
def get_queue_info() -> Dict[str, Any]:
    """获取队列信息"""
    try:
        inspect = celery_app.control.inspect()

        return {
            'active_queues': inspect.active_queues(),
            'scheduled_tasks': inspect.scheduled(),
            'active_tasks': inspect.active(),
            'reserved_tasks': inspect.reserved(),
            'stats': inspect.stats(),
        }
    except Exception as e:
        logger.error(f'Failed to get queue info: {e}')
        return {}


# 清理过期任务的辅助函数
def cleanup_expired_tasks():
    """清理过期任务"""
    try:
        # 清理过期的任务结果
        celery_app.control.purge()
        logger.info('Expired tasks cleaned up successfully')
    except Exception as e:
        logger.error(f'Failed to cleanup expired tasks: {e}')


# 动态调整Worker数量的辅助函数
def scale_workers(queue_name: str, concurrency: int):
    """动态调整Worker并发数"""
    try:
        celery_app.control.pool_restart()
        logger.info(f'Worker pool restarted for queue {queue_name}')
    except Exception as e:
        logger.error(f'Failed to scale workers: {e}')


# 如果直接运行此文件，启动Celery worker
if __name__ == '__main__':
    celery_app.start()