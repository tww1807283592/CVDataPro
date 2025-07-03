# 数据合成项目完整文档

## 项目概述

### 项目简介
高性能前后端分离的计算机视觉数据合成系统，专注于智能抠图、多目标合成和多格式数据集生成。系统采用异步架构，支持高并发处理，具备完善的内存管理机制。

### 核心功能
- **智能抠图**: 支持多种深度学习模型的自动抠图和指定目标抠图
- **多目标合成**: 支持在单个背景图中放置多个前景目标
- **位置控制**: 支持随机放置和手动指定位置
- **多格式支持**: 支持所有常见图像格式（JPG, PNG, BMP, TIFF等）
- **批量处理**: 支持文件夹批量导入和处理
- **数据集生成**: 支持TXT, JSON, XML, COCO, YOLO等多种标注格式
- **项目隔离**: 每个项目的图像和数据完全独立

### 技术特点
- **异步架构**: 基于FastAPI和asyncio的高性能异步处理
- **多线程支持**: 智能的线程池管理和并发控制
- **高并发处理**: 支持大量并发请求和任务处理
- **内存优化**: 智能内存管理和垃圾回收机制
- **容器化部署**: 完整的Docker容器化方案

## 完整项目结构

```
data-synthesis-platform/
├── README.md                         # 项目说明文档
├── docker-compose.yml                # 整体Docker编排
├── .env.example                      # 环境变量示例
├── .gitignore                        # Git忽略规则
├── nginx/                            # Nginx配置
│   ├── nginx.conf                    # 主配置文件
│   ├── ssl/                          # SSL证书目录
│   └── logs/                         # Nginx日志
├── 
├── backend/                          # 后端服务目录
│   ├── app/                          # 主应用代码
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI主应用入口
│   │   │
│   │   ├── config/                   # 配置管理
│   │   │   ├── __init__.py
│   │   │   ├── settings.py           # 应用设置和环境变量
│   │   │   ├── database.py           # 数据库连接配置
│   │   │   ├── redis.py              # Redis连接配置
│   │   │   └── logging.py            # 日志配置
│   │   │
│   │   ├── core/                     # 核心功能模块
│   │   │   ├── __init__.py
│   │   │   ├── security.py           # 认证和授权
│   │   │   ├── middleware.py         # 中间件定义
│   │   │   ├── exceptions.py         # 异常处理器
│   │   │   ├── dependencies.py       # 依赖注入
│   │   │   └── events.py             # 应用事件处理
│   │   │
│   │   ├── models/                   # 数据模型定义
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # 基础模型类
│   │   │   ├── project.py            # 项目模型
│   │   │   ├── image.py              # 图像模型
│   │   │   ├── synthesis_task.py     # 合成任务模型
│   │   │   ├── dataset.py            # 数据集模型
│   │   │   └── user.py               # 用户模型
│   │   │
│   │   ├── schemas/                  # Pydantic数据校验模式
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # 基础Schema
│   │   │   ├── project.py            # 项目相关Schema
│   │   │   ├── image.py              # 图像相关Schema
│   │   │   ├── synthesis.py          # 合成相关Schema
│   │   │   ├── dataset.py            # 数据集相关Schema
│   │   │   └── response.py           # 响应模式定义
│   │   │
│   │   ├── api/                      # API路由定义
│   │   │   ├── __init__.py
│   │   │   ├── deps.py               # API依赖
│   │   │   └── v1/                   # API版本1
│   │   │       ├── __init__.py
│   │   │       ├── projects.py       # 项目管理API
│   │   │       ├── images.py         # 图像处理API
│   │   │       ├── matting.py        # 抠图功能API
│   │   │       ├── synthesis.py      # 合成功能API
│   │   │       ├── datasets.py       # 数据集API
│   │   │       ├── upload.py         # 文件上传API
│   │   │       └── download.py       # 文件下载API
│   │   │
│   │   ├── services/                 # 业务逻辑服务层
│   │   │   ├── __init__.py
│   │   │   ├── base_service.py       # 基础服务类
│   │   │   ├── project_service.py    # 项目管理服务
│   │   │   ├── image_service.py      # 图像处理服务
│   │   │   ├── matting_service.py    # 智能抠图服务
│   │   │   ├── synthesis_service.py  # 图像合成服务
│   │   │   ├── dataset_service.py    # 数据集生成服务
│   │   │   ├── upload_service.py     # 文件上传服务
│   │   │   └── storage_service.py    # 存储管理服务
│   │   │
│   │   ├── workers/                  # 异步任务工作器
│   │   │   ├── __init__.py
│   │   │   ├── base_worker.py        # 基础工作器
│   │   │   ├── matting_worker.py     # 抠图任务工作器
│   │   │   ├── synthesis_worker.py   # 合成任务工作器
│   │   │   ├── dataset_worker.py     # 数据集生成工作器
│   │   │   └── cleanup_worker.py     # 清理任务工作器
│   │   │
│   │   ├── utils/                    # 工具函数模块
│   │   │   ├── __init__.py
│   │   │   ├── image_utils.py        # 图像处理工具
│   │   │   ├── file_utils.py         # 文件操作工具
│   │   │   ├── memory_utils.py       # 内存管理工具
│   │   │   ├── async_utils.py        # 异步处理工具
│   │   │   ├── validation_utils.py   # 数据验证工具
│   │   │   ├── format_utils.py       # 格式转换工具
│   │   │   └── logger_utils.py       # 日志工具
│   │   │
│   │   ├── db/                       # 数据库相关
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # 数据库基础类
│   │   │   ├── session.py            # 会话管理
│   │   │   └── init_db.py            # 数据库初始化
│   │   │
│   │   ├── tasks/                    # Celery任务定义
│   │   │   ├── __init__.py
│   │   │   ├── matting_tasks.py      # 抠图任务
│   │   │   ├── synthesis_tasks.py    # 合成任务
│   │   │   ├── dataset_tasks.py      # 数据集任务
│   │   │   └── maintenance_tasks.py  # 维护任务
│   │   │
│   │   └── celery_app.py             # Celery应用配置
│   │
│   ├── alembic/                      # 数据库迁移
│   │   ├── env.py                    # Alembic环境配置
│   │   ├── script.py.mako            # 迁移脚本模板
│   │   ├── alembic.ini               # Alembic配置文件
│   │   └── versions/                 # 迁移版本文件
│   │
│   ├── tests/                        # 测试文件
│   │   ├── __init__.py
│   │   ├── conftest.py               # pytest配置
│   │   ├── test_api/                 # API测试
│   │   │   ├── __init__.py
│   │   │   ├── test_projects.py
│   │   │   ├── test_images.py
│   │   │   ├── test_synthesis.py
│   │   │   └── test_datasets.py
│   │   ├── test_services/            # 服务层测试
│   │   │   ├── __init__.py
│   │   │   ├── test_matting_service.py
│   │   │   ├── test_synthesis_service.py
│   │   │   └── test_dataset_service.py
│   │   └── test_utils/               # 工具函数测试
│   │       ├── __init__.py
│   │       ├── test_image_utils.py
│   │       └── test_memory_utils.py
│   │
│   ├── scripts/                      # 脚本文件
│   │   ├── init_db.py               # 数据库初始化脚本
│   │   ├── create_superuser.py      # 创建管理员脚本
│   │   └── migrate.py               # 数据迁移脚本
│   │
│   ├── requirements/                 # 依赖文件
│   │   ├── base.txt                 # 基础依赖
│   │   ├── dev.txt                  # 开发依赖
│   │   ├── prod.txt                 # 生产依赖
│   │   └── test.txt                 # 测试依赖
│   │
│   ├── Dockerfile                   # 后端Docker镜像
│   ├── Dockerfile.dev               # 开发环境Docker镜像
│   ├── .dockerignore               # Docker忽略文件
│   └── pyproject.toml              # Python项目配置
│
├── frontend/                        # 前端应用目录
│   ├── public/                      # 静态资源
│   │   ├── index.html              # 主页面模板
│   │   ├── favicon.ico             # 网站图标
│   │   └── assets/                 # 静态资源文件
│   │
│   ├── src/                        # 源代码
│   │   ├── main.tsx                # 应用入口文件
│   │   ├── App.tsx                 # 根组件
│   │   ├── index.css               # 全局样式
│   │   ├── vite-env.d.ts          # Vite类型定义
│   │   │
│   │   ├── components/             # 组件目录
│   │   │   ├── common/             # 通用组件
│   │   │   │   ├── Layout/         # 布局组件
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   ├── Header.tsx
│   │   │   │   │   ├── Sidebar.tsx
│   │   │   │   │   └── Footer.tsx
│   │   │   │   ├── Upload/         # 上传组件
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   ├── ImageUpload.tsx
│   │   │   │   │   └── FolderUpload.tsx
│   │   │   │   ├── Loading/        # 加载组件
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── Spinner.tsx
│   │   │   │   └── Modal/          # 模态框组件
│   │   │   │       ├── index.tsx
│   │   │   │       └── ConfirmModal.tsx
│   │   │   │
│   │   │   ├── project/            # 项目相关组件
│   │   │   │   ├── ProjectList/    # 项目列表
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── ProjectCard.tsx
│   │   │   │   ├── ProjectForm/    # 项目表单
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── ProjectSettings.tsx
│   │   │   │   └── ProjectDetail/  # 项目详情
│   │   │   │       ├── index.tsx
│   │   │   │       └── ProjectInfo.tsx
│   │   │   │
│   │   │   ├── image/              # 图像处理组件
│   │   │   │   ├── ImagePreview/   # 图像预览
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── ImageViewer.tsx
│   │   │   │   ├── ImageEditor/    # 图像编辑
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   ├── Canvas.tsx
│   │   │   │   │   └── Toolbar.tsx
│   │   │   │   └── ImageGallery/   # 图像画廊
│   │   │   │       ├── index.tsx
│   │   │   │       └── GalleryGrid.tsx
│   │   │   │
│   │   │   ├── matting/            # 抠图相关组件
│   │   │   │   ├── MattingPanel/   # 抠图面板
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   └── ModelSelector.tsx
│   │   │   │   └── MattingResult/  # 抠图结果
│   │   │   │       ├── index.tsx
│   │   │   │       └── ResultViewer.tsx
│   │   │   │
│   │   │   └── synthesis/          # 合成相关组件
│   │   │       ├── SynthesisStudio/# 合成工作室
│   │   │       │   ├── index.tsx
│   │   │       │   ├── Canvas.tsx
│   │   │       │   ├── ObjectPanel.tsx
│   │   │       │   └── ControlPanel.tsx
│   │   │       ├── ConfigPanel/    # 配置面板
│   │   │       │   ├── index.tsx
│   │   │       │   ├── PositionControl.tsx
│   │   │       │   └── StyleControl.tsx
│   │   │       └── BatchSynthesis/ # 批量合成
│   │   │           ├── index.tsx
│   │   │           └── BatchConfig.tsx
│   │   │
│   │   ├── pages/                  # 页面组件
│   │   │   ├── Home/               # 首页
│   │   │   │   └── index.tsx
│   │   │   ├── Project/            # 项目管理页面
│   │   │   │   ├── index.tsx
│   │   │   │   ├── List.tsx
│   │   │   │   └── Detail.tsx
│   │   │   ├── ImageProcess/       # 图像处理页面
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Upload.tsx
│   │   │   │   └── Matting.tsx
│   │   │   ├── Synthesis/          # 合成页面
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Studio.tsx
│   │   │   │   └── Batch.tsx
│   │   │   ├── Dataset/            # 数据集页面
│   │   │   │   ├── index.tsx
│   │   │   │   ├── Generate.tsx
│   │   │   │   └── Download.tsx
│   │   │   └── Settings/           # 设置页面
│   │   │       ├── index.tsx
│   │   │       └── Profile.tsx
│   │   │
│   │   ├── services/               # API服务层
│   │   │   ├── api.ts              # API基础配置
│   │   │   ├── projectService.ts   # 项目服务
│   │   │   ├── imageService.ts     # 图像服务
│   │   │   ├── mattingService.ts   # 抠图服务
│   │   │   ├── synthesisService.ts # 合成服务
│   │   │   ├── datasetService.ts   # 数据集服务
│   │   │   └── uploadService.ts    # 上传服务
│   │   │
│   │   ├── stores/                 # 状态管理
│   │   │   ├── index.ts            # Store入口
│   │   │   ├── useProjectStore.ts  # 项目状态
│   │   │   ├── useImageStore.ts    # 图像状态
│   │   │   ├── useSynthesisStore.ts# 合成状态
│   │   │   └── useGlobalStore.ts   # 全局状态
│   │   │
│   │   ├── hooks/                  # 自定义Hook
│   │   │   ├── useAsync.ts         # 异步处理Hook
│   │   │   ├── useUpload.ts        # 上传Hook
│   │   │   ├── useWebSocket.ts     # WebSocket Hook
│   │   │   └── useLocalStorage.ts  # 本地存储Hook
│   │   │
│   │   ├── utils/                  # 工具函数
│   │   │   ├── request.ts          # 请求工具
│   │   │   ├── image.ts            # 图像处理工具
│   │   │   ├── format.ts           # 格式化工具
│   │   │   ├── validation.ts       # 验证工具
│   │   │   └── constants.ts        # 常量定义
│   │   │
│   │   ├── types/                  # TypeScript类型定义
│   │   │   ├── index.ts            # 类型入口
│   │   │   ├── project.ts          # 项目类型
│   │   │   ├── image.ts            # 图像类型
│   │   │   ├── synthesis.ts        # 合成类型
│   │   │   ├── dataset.ts          # 数据集类型
│   │   │   └── common.ts           # 通用类型
│   │   │
│   │   └── assets/                 # 资源文件
│   │       ├── images/             # 图片资源
│   │       ├── icons/              # 图标资源
│   │       └── styles/             # 样式文件
│   │           ├── globals.css     # 全局样式
│   │           ├── variables.css   # CSS变量
│   │           └── components.css  # 组件样式
│   │
│   ├── package.json                # 项目依赖和脚本
│   ├── package-lock.json           # 依赖锁定文件
│   ├── vite.config.ts              # Vite配置
│   ├── tsconfig.json               # TypeScript配置
│   ├── tsconfig.node.json          # Node.js TypeScript配置
│   ├── tailwind.config.js          # Tailwind CSS配置
│   ├── postcss.config.js           # PostCSS配置
│   ├── Dockerfile                  # 前端Docker镜像
│   └── .env.example                # 环境变量示例
│
├── storage/                        # 存储目录
│   ├── uploads/                    # 上传文件存储
│   │   ├── projects/               # 按项目分类存储
│   │   │   ├── {project_id}/       # 项目专用目录
│   │   │   │   ├── foregrounds/    # 前景图
│   │   │   │   ├── backgrounds/    # 背景图
│   │   │   │   └── processed/      # 处理后图像
│   │   └── temp/                   # 临时文件
│   │
│   ├── datasets/                   # 生成的数据集
│   │   ├── {project_id}/           # 按项目存储数据集
│   │   │   ├── images/             # 合成图像
│   │   │   ├── annotations/        # 标注文件
│   │   │   └── exports/            # 导出文件
│   │
│   ├── models/                     # AI模型文件
│   │   ├── matting/                # 抠图模型
│   │   └── detection/              # 检测模型
│   │
│   └── cache/                      # 缓存文件
│       ├── thumbnails/             # 缩略图缓存
│       └── processed/              # 处理结果缓存
│
├── docs/                           # 项目文档
│   ├── api/                        # API文档
│   │   ├── openapi.json            # OpenAPI规范
│   │   └── postman/                # Postman集合
│   ├── deployment/                 # 部署文档
│   │   ├── docker.md               # Docker部署
│   │   ├── kubernetes.md           # K8s部署
│   │   └── production.md           # 生产环境配置
│   ├── development/                # 开发文档
│   │   ├── setup.md                # 环境搭建
│   │   ├── contributing.md         # 贡献指南
│   │   └── testing.md              # 测试指南
│   └── user/                       # 用户文档
│       ├── quickstart.md           # 快速开始
│       ├── tutorial.md             # 使用教程
│       └── faq.md                  # 常见问题
│
├── monitoring/                     # 监控配置
│   ├── prometheus/                 # Prometheus配置
│   │   └── prometheus.yml
│   ├── grafana/                    # Grafana配置
│   │   ├── dashboards/             # 仪表板
│   │   └── datasources/            # 数据源
│   └── alerts/                     # 告警规则
│
├── scripts/                        # 项目脚本
│   ├── setup.sh                    # 环境初始化脚本
│   ├── deploy.sh                   # 部署脚本
│   ├── backup.sh                   # 备份脚本
│   └── cleanup.sh                  # 清理脚本
│
└── k8s/                           # Kubernetes配置
    ├── namespace.yaml              # 命名空间
    ├── configmap.yaml              # 配置映射
    ├── secret.yaml                 # 密钥配置
    ├── backend/                    # 后端K8s配置
    │   ├── deployment.yaml         # 部署配置
    │   ├── service.yaml            # 服务配置
    │   └── hpa.yaml                # 自动扩缩容
    ├── frontend/                   # 前端K8s配置
    │   ├── deployment.yaml
    │   └── service.yaml
    ├── database/                   # 数据库K8s配置
    │   ├── postgres.yaml
    │   └── redis.yaml
    └── ingress.yaml                # 入口配置
```

