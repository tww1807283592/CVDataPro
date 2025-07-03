# 图像数据合成系统 - 项目结构框架

## 项目概述
基于Python的异步高并发图像数据合成系统，支持前景图智能抠图、背景图管理、图像合成和训练数据集生成。

## 技术栈

### 后端
- **框架**: FastAPI (异步高性能)
- **数据库**: PostgreSQL + SQLAlchemy (异步ORM)
- **图像处理**: OpenCV, Pillow, rembg (智能抠图)
- **异步任务**: Celery + Redis
- **文件存储**: 本地存储 + 可选云存储
- **并发处理**: asyncio, aiofiles, concurrent.futures
- **内存管理**: 内存池, 批处理, 流式处理

### 前端
- **框架**: Vue.js 3 + TypeScript
- **UI组件**: Element Plus
- **状态管理**: Pinia
- **图像处理**: Fabric.js (画布操作)
- **文件上传**: axios + 进度条

## 项目结构

```
image_synthesis_system/
├── backend/                        # 后端服务
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI应用入口
│   │   ├── config.py               # 配置文件
│   │   ├── dependencies.py         # 依赖注入
│   │   │
│   │   ├── models/                 # 数据模型 (已提供)
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── project.py
│   │   │   ├── image.py
│   │   │   ├── synthesis_task.py
│   │   │   ├── dataset.py
│   │   │   └── user.py
│   │   │
│   │   ├── schemas/                # Pydantic数据模式
│   │   │   ├── __init__.py
│   │   │   ├── project.py
│   │   │   ├── image.py
│   │   │   ├── synthesis.py
│   │   │   └── dataset.py
│   │   │
│   │   ├── api/                    # API路由
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── projects.py
│   │   │   │   ├── images.py
│   │   │   │   ├── synthesis.py
│   │   │   │   └── datasets.py
│   │   │   └── deps.py
│   │   │
│   │   ├── services/               # 业务逻辑服务
│   │   │   ├── __init__.py
│   │   │   ├── project_service.py
│   │   │   ├── image_service.py
│   │   │   ├── matting_service.py
│   │   │   ├── synthesis_service.py
│   │   │   └── dataset_service.py
│   │   │
│   │   ├── core/                   # 核心功能
│   │   │   ├── __init__.py
│   │   │   ├── database.py         # 数据库连接
│   │   │   ├── redis.py            # Redis连接
│   │   │   ├── security.py         # 安全相关
│   │   │   └── exceptions.py       # 异常处理
│   │   │
│   │   ├── processors/             # 图像处理器
│   │   │   ├── __init__.py
│   │   │   ├── image_processor.py  # 图像基础处理
│   │   │   ├── matting_processor.py # 抠图处理
│   │   │   ├── synthesis_processor.py # 合成处理
│   │   │   └── dataset_generator.py # 数据集生成
│   │   │
│   │   ├── utils/                  # 工具类
│   │   │   ├── __init__.py
│   │   │   ├── file_utils.py
│   │   │   ├── image_utils.py
│   │   │   ├── async_utils.py
│   │   │   └── memory_utils.py
│   │   │
│   │   └── tasks/                  # 异步任务
│   │       ├── __init__.py
│   │       ├── image_tasks.py
│   │       ├── synthesis_tasks.py
│   │       └── dataset_tasks.py
│   │
│   ├── tests/                      # 测试
│   ├── requirements.txt            # 依赖包
│   ├── Dockerfile                  # Docker配置
│   └── docker-compose.yml
│
├── frontend/                       # 前端应用
│   ├── src/
│   │   ├── components/             # 组件
│   │   │   ├── project/
│   │   │   ├── image/
│   │   │   ├── synthesis/
│   │   │   └── dataset/
│   │   │
│   │   ├── views/                  # 页面
│   │   │   ├── ProjectView.vue
│   │   │   ├── ImageView.vue
│   │   │   ├── SynthesisView.vue
│   │   │   └── DatasetView.vue
│   │   │
│   │   ├── services/               # API服务
│   │   ├── stores/                 # 状态管理
│   │   ├── utils/                  # 工具类
│   │   └── types/                  # TypeScript类型
│   │
│   ├── public/
│   ├── package.json
│   └── Dockerfile
│
├── storage/                        # 存储目录
│   ├── projects/
│   ├── images/
│   ├── results/
│   └── datasets/
│
├── docs/                           # 文档
├── scripts/                        # 脚本
└── README.md
```

## 核心功能模块

### 1. 项目管理
- 创建/删除/编辑项目
- 项目配置管理
- 项目统计信息

### 2. 图像管理
- 多格式图像上传 (JPEG, PNG, TIFF, BMP, WEBP)
- 批量上传文件夹
- 图像预览和元数据提取
- 图像分类管理 (前景图/背景图)

### 3. 智能抠图
- 基于rembg的自动抠图
- 可指定抠图目标物体
- 抠图质量评估
- 抠图结果预览

### 4. 图像合成
- 前景图随机放置到背景图
- 手动控制放置位置
- 多个前景图合成到一个背景图
- 合成参数控制 (大小、旋转、透明度等)

### 5. 数据集生成
- 支持多种格式: TXT, JSON, XML
- 自动生成标注文件
- 数据集打包下载
- 自定义数据集格式

## 技术特性

### 异步处理
- FastAPI异步框架
- asyncio并发处理
- 异步文件I/O
- 流式文件上传

### 高并发
- 连接池管理
- 任务队列处理
- 并发限制控制
- 负载均衡

### 内存管理
- 图像内存池
- 批处理机制
- 内存使用监控
- 自动垃圾回收

### 性能优化
- 图像缓存机制
- 数据库连接池
- Redis缓存
- CDN支持

## 部署方案

### 开发环境
```bash
# 后端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# 前端
cd frontend
npm install
npm run dev
```

### 生产环境
```bash
# Docker部署
docker-compose up -d
```

## 监控和日志
- 应用性能监控
- 错误日志收集
- 业务指标统计
- 系统资源监控