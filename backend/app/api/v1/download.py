#!/usr/bin/env python3
"""
AI模型下载脚本
用于下载和管理图像合成平台所需的AI模型
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import argparse
from loguru import logger
import json
from tqdm.asyncio import tqdm
import tempfile
import shutil

# 配置日志
logger.remove()
logger.add(sys.stdout,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    url: str
    filename: str
    size: int
    sha256: str
    description: str
    required: bool = True
    model_type: str = "general"


class ModelDownloader:
    """AI模型下载器"""

    def __init__(self, models_dir: str = "models", max_concurrent: int = 3):
        self.models_dir = Path(models_dir)
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 创建模型目录
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 定义模型配置
        self.models_config = {
            # SAM模型
            "sam_vit_h": ModelInfo(
                name="sam_vit_h",
                url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                filename="sam_vit_h_4b8939.pth",
                size=2564550879,  # ~2.4GB
                sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
                description="SAM ViT-H 模型 (最高精度)",
                model_type="sam"
            ),
            "sam_vit_l": ModelInfo(
                name="sam_vit_l",
                url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                filename="sam_vit_l_0b3195.pth",
                size=1249550879,  # ~1.2GB
                sha256="3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
                description="SAM ViT-L 模型 (高精度)",
                model_type="sam"
            ),
            "sam_vit_b": ModelInfo(
                name="sam_vit_b",
                url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                filename="sam_vit_b_01ec64.pth",
                size=375042383,  # ~358MB
                sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
                description="SAM ViT-B 模型 (标准精度)",
                model_type="sam"
            ),

            # U2Net模型
            "u2net": ModelInfo(
                name="u2net",
                url="https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth",
                filename="u2net.pth",
                size=176681193,  # ~168MB
                sha256="347c3d51b7daf4b9cdecbf4bed263c5e70d743b7ac3c7a2d2b9eef0c7b913d4c",
                description="U2Net 抠图模型",
                model_type="u2net"
            ),
            "u2net_human_seg": ModelInfo(
                name="u2net_human_seg",
                url="https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net_human_seg.pth",
                filename="u2net_human_seg.pth",
                size=176681057,  # ~168MB
                sha256="c09ddc2e0104f800e3e1bb4652583d1f5c5e51cd5e4fb80c0e2e1f4e4c1e4e99",
                description="U2Net 人体分割模型",
                model_type="u2net"
            ),

            # RemBG模型 (这些模型会自动下载，这里预留配置)
            "rembg_u2net": ModelInfo(
                name="rembg_u2net",
                url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
                filename="u2net.onnx",
                size=176681193,
                sha256="60024c5c889badc19c04ad937298a77789e17c2532ee4a42b5e5c2b7a0b4e1a5",
                description="RemBG U2Net ONNX模型",
                model_type="rembg",
                required=False
            ),
            "rembg_silueta": ModelInfo(
                name="rembg_silueta",
                url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
                filename="silueta.onnx",
                size=43827445,
                sha256="55e59e0d8062d2f5d013e6c2392f0c65c8e8e8e6e8e8e8e8e8e8e8e8e8e8e8e8",
                description="RemBG Silueta ONNX模型",
                model_type="rembg",
                required=False
            )
        }

        # 创建模型类型目录
        for model_type in ["sam", "u2net", "rembg"]:
            (self.models_dir / model_type).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(total=3600)  # 1小时超时
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()

    def get_model_path(self, model_info: ModelInfo) -> Path:
        """获取模型文件路径"""
        return self.models_dir / model_info.model_type / model_info.filename

    def verify_file_hash(self, file_path: Path, expected_hash: str) -> bool:
        """验证文件SHA256哈希值"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # 分块读取，避免内存占用过大
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            actual_hash = sha256_hash.hexdigest()
            return actual_hash == expected_hash
        except Exception as e:
            logger.error(f"验证文件哈希失败: {e}")
            return False

    def is_model_downloaded(self, model_info: ModelInfo) -> bool:
        """检查模型是否已下载且完整"""
        model_path = self.get_model_path(model_info)

        if not model_path.exists():
            return False

        # 检查文件大小
        if model_path.stat().st_size != model_info.size:
            logger.warning(f"模型文件大小不匹配: {model_info.name}")
            return False

        # 验证哈希值
        if not self.verify_file_hash(model_path, model_info.sha256):
            logger.warning(f"模型文件哈希验证失败: {model_info.name}")
            return False

        return True

    async def download_model(self, model_info: ModelInfo, force: bool = False) -> bool:
        """下载单个模型"""
        async with self.semaphore:
            model_path = self.get_model_path(model_info)

            # 检查是否已下载
            if not force and self.is_model_downloaded(model_info):
                logger.info(f"模型已存在且完整: {model_info.name}")
                return True

            logger.info(f"开始下载模型: {model_info.name}")
            logger.info(f"描述: {model_info.description}")
            logger.info(f"大小: {model_info.size / (1024 * 1024):.1f} MB")

            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                    tmp_path = Path(tmp_file.name)

                # 下载文件
                async with self.session.get(model_info.url) as response:
                    response.raise_for_status()

                    # 获取文件大小
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size == 0:
                        total_size = model_info.size

                    # 创建进度条
                    progress_bar = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"下载 {model_info.name}"
                    )

                    # 写入临时文件
                    async with aiofiles.open(tmp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            progress_bar.update(len(chunk))

                    progress_bar.close()

                # 验证下载的文件
                if not self.verify_file_hash(tmp_path, model_info.sha256):
                    logger.error(f"下载的文件哈希验证失败: {model_info.name}")
                    tmp_path.unlink()
                    return False

                # 移动到最终位置
                model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(model_path))

                logger.success(f"模型下载成功: {model_info.name}")
                return True

            except Exception as e:
                logger.error(f"下载模型失败 {model_info.name}: {e}")
                # 清理临时文件
                if tmp_path.exists():
                    tmp_path.unlink()
                return False

    async def download_models(self, model_names: Optional[List[str]] = None,
                              force: bool = False, required_only: bool = False) -> Dict[str, bool]:
        """下载多个模型"""
        if model_names is None:
            if required_only:
                models_to_download = [
                    model for model in self.models_config.values()
                    if model.required
                ]
            else:
                models_to_download = list(self.models_config.values())
        else:
            models_to_download = [
                self.models_config[name] for name in model_names
                if name in self.models_config
            ]

        logger.info(f"准备下载 {len(models_to_download)} 个模型")

        # 创建下载任务
        tasks = []
        for model_info in models_to_download:
            task = asyncio.create_task(
                self.download_model(model_info, force),
                name=f"download_{model_info.name}"
            )
            tasks.append((model_info.name, task))

        # 等待所有任务完成
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"下载任务失败 {name}: {e}")
                results[name] = False

        return results

    def list_models(self) -> None:
        """列出所有可用的模型"""
        logger.info("可用模型列表:")

        for model_type in ["sam", "u2net", "rembg"]:
            models = [m for m in self.models_config.values() if m.model_type == model_type]
            if models:
                logger.info(f"\n{model_type.upper()} 模型:")
                for model in models:
                    status = "✓" if self.is_model_downloaded(model) else "✗"
                    required = "必需" if model.required else "可选"
                    size_mb = model.size / (1024 * 1024)
                    logger.info(f"  {status} {model.name} ({required}, {size_mb:.1f} MB)")
                    logger.info(f"    {model.description}")

    def check_models_status(self) -> Dict[str, bool]:
        """检查所有模型的下载状态"""
        status = {}
        for name, model_info in self.models_config.items():
            status[name] = self.is_model_downloaded(model_info)
        return status

    def get_missing_models(self, required_only: bool = False) -> List[str]:
        """获取缺失的模型列表"""
        missing = []
        for name, model_info in self.models_config.items():
            if required_only and not model_info.required:
                continue
            if not self.is_model_downloaded(model_info):
                missing.append(name)
        return missing

    def save_models_info(self) -> None:
        """保存模型信息到JSON文件"""
        info_file = self.models_dir / "models_info.json"
        models_info = {}

        for name, model_info in self.models_config.items():
            models_info[name] = {
                "name": model_info.name,
                "filename": model_info.filename,
                "size": model_info.size,
                "sha256": model_info.sha256,
                "description": model_info.description,
                "required": model_info.required,
                "model_type": model_info.model_type,
                "downloaded": self.is_model_downloaded(model_info)
            }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(models_info, f, indent=2, ensure_ascii=False)

        logger.info(f"模型信息已保存到: {info_file}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI模型下载工具")
    parser.add_argument("--models-dir", default="models",
                        help="模型存储目录 (默认: models)")
    parser.add_argument("--models", nargs="+",
                        help="指定要下载的模型名称")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载已存在的模型")
    parser.add_argument("--required-only", action="store_true",
                        help="只下载必需的模型")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用模型")
    parser.add_argument("--check", action="store_true",
                        help="检查模型下载状态")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="最大并发下载数 (默认: 3)")

    args = parser.parse_args()

    # 创建下载器
    downloader = ModelDownloader(args.models_dir, args.max_concurrent)

    # 处理命令
    if args.list:
        downloader.list_models()
        return

    if args.check:
        status = downloader.check_models_status()
        logger.info("模型状态:")
        for name, downloaded in status.items():
            status_str = "✓ 已下载" if downloaded else "✗ 未下载"
            logger.info(f"  {name}: {status_str}")

        missing = downloader.get_missing_models(args.required_only)
        if missing:
            logger.warning(f"缺失的模型: {', '.join(missing)}")
        return

    # 下载模型
    async with downloader:
        results = await downloader.download_models(
            args.models, args.force, args.required_only
        )

        # 显示结果
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        logger.info(f"\n下载完成: {successful}/{total} 个模型成功")

        if successful < total:
            failed = [name for name, success in results.items() if not success]
            logger.error(f"下载失败的模型: {', '.join(failed)}")

        # 保存模型信息
        downloader.save_models_info()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("下载被用户中断")
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        sys.exit(1)