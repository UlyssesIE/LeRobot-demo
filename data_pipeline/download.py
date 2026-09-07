"""下载 lerobot/xarm_lift_medium 并执行 AV1 解码冒烟测试。

数据集 ~17.8MB（Parquet + AV1/MP4），下载秒级完成。

用法（项目根目录 D:\\Code demos\\LeRobot\\LeRobot-demo，conda 环境 lerobot）：
    python data_pipeline/download.py                 # 下载 + 冒烟测试
    python data_pipeline/download.py --smoke-only    # 仅冒烟测试（数据已存在）
    python data_pipeline/download.py --skip-smoke    # 仅下载
    python data_pipeline/download.py --video-backend pyav   # 指定视频解码后端

注意：
    xarm_lift_medium 是 VA 数据集（无语言指令字段），训练时需注入占位语言指令
    （见 PLACEHOLDER_TASK）激活语言通路；本模块只负责下载与解码验证，不做注入。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset

# import random

logger = logging.getLogger("data_pipeline.download")

# --------------------------------------------------------------------------- #
# 常量 / 可配置项
# --------------------------------------------------------------------------- #
DATASET_REPO_ID = "lerobot/xarm_lift_medium"

# 下载落盘根目录。注意 LeRobotDataset 的 root 语义是「数据集直接物化到该路径下」，
# 不会在 root 下再建 repo_id 子目录。
DATASET_DIR = Path("outputs/datasets") / "xarm_lift_medium"

# 占位语言指令（训练阶段注入；此处仅作常量记录）
PLACEHOLDER_TASK = "Lift the object"

# 期望元数据与 shape，用于断言，避免静默的错误
EXPECTED_FPS = 15
EXPECTED_NUM_EPISODES = 800
EXPECTED_NUM_FRAMES = 20_000
EXPECTED_IMAGE_SHAPE = (3, 84, 84)  # observation.image 解码后 (C, H, W)
EXPECTED_STATE_SHAPE = (4,)         # observation.state
EXPECTED_ACTION_SHAPE = (4,)        # action


# --------------------------------------------------------------------------- #
# 提交 2：下载部分
# --------------------------------------------------------------------------- #
def download_dataset(
    repo_id: str = DATASET_REPO_ID,
    root: Path | str = DATASET_DIR,
) -> LeRobotDataset:
    """下载（若不存在）并加载数据集到本地 root。

    返回已加载的 dataset 实例，可复用于冒烟测试，避免二次下载。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    logger.info("下载/加载数据集 %s -> %s（约 17.8MB）...", repo_id, root)
    dataset = LeRobotDataset(repo_id, root=root)

    logger.info(
        "完成：episodes=%d, frames=%d, fps=%d, root=%s",
        dataset.num_episodes,
        dataset.num_frames,
        dataset.fps,
        root,
    )
    return dataset


# --------------------------------------------------------------------------- #
# 提交 1：AV1 解码冒烟测试
# --------------------------------------------------------------------------- #
def _find_video_keys(dataset: LeRobotDataset) -> list[str]:
    """发现视频特征键（不硬编码 observation.image，兼容多机位）。"""
    meta_keys = getattr(dataset.meta, "video_keys", None)
    if meta_keys:
        return list(meta_keys)
    return [k for k, v in dataset.features.items() if v.get("dtype") == "video"]


def smoke_test_av1_decode(
    dataset: LeRobotDataset | None = None,
    n_samples: int = 3,
    video_backend: str | None = None,
) -> None:
    """AV1 解码冒烟测试：抽样解码若干帧，校验 shape / 数值范围 / 非全零。

    - 验证 observation.image（AV1 / yuv420p）能被后端正确解帧；
    - 验证 observation.state / action 维度与 info.json 一致；
    - 解码失败时给出切换 video_backend 的明确提示。
    """
    if dataset is None:
        dataset = LeRobotDataset(
            DATASET_REPO_ID, root=DATASET_DIR, video_backend=video_backend
        )

    logger.info("实际视频后端：%s", getattr(dataset, "_video_backend", "unknown"))

    # 元数据断言
    assert dataset.fps == EXPECTED_FPS, f"fps={dataset.fps} != {EXPECTED_FPS}"
    assert dataset.num_episodes == EXPECTED_NUM_EPISODES, (
        f"episodes={dataset.num_episodes} != {EXPECTED_NUM_EPISODES}"
    )
    assert dataset.num_frames == EXPECTED_NUM_FRAMES, (
        f"frames={dataset.num_frames} != {EXPECTED_NUM_FRAMES}"
    )

    video_keys = _find_video_keys(dataset)
    if not video_keys:
        raise RuntimeError("未发现任何 video 特征，无法执行 AV1 解码测试。")
    logger.info("视频特征键：%s", video_keys)

    # 采样首/中/尾帧，覆盖不同视频 shard
    total = dataset.num_frames
    indices = sorted({0, total // 2, total - 1} | {i % total for i in range(n_samples)})

    for idx in indices:
        try:
            sample = dataset[idx]
        except Exception as e:  # noqa: BLE001 - 冒烟测试需捕获一切解码异常
            logger.error("解码失败 @idx=%d: %s", idx, e)
            raise RuntimeError(
                "AV1 解码失败。torch 2.10 的 torchcodec 可能缺少 AV1 解码器；"
                "请尝试 --video-backend pyav（需 `pip install av`）。"
            ) from e

        for key in video_keys:
            image = sample[key]
            assert tuple(image.shape) == EXPECTED_IMAGE_SHAPE, (
                f"{key} shape={tuple(image.shape)} != {EXPECTED_IMAGE_SHAPE}"
            )
            lo, hi = float(image.min()), float(image.max())
            assert 0.0 <= lo and hi <= 255.0, f"{key} 数值越界 [{lo}, {hi}]"
            assert hi - lo > 0.0, f"{key} 为常量帧（疑似黑帧/绿帧，AV1 解码异常）"

        state = sample["observation.state"]
        action = sample["action"]
        assert tuple(state.shape) == EXPECTED_STATE_SHAPE
        assert state.dtype == torch.float32
        assert tuple(action.shape) == EXPECTED_ACTION_SHAPE
        assert action.dtype == torch.float32

        logger.info(
            "idx=%d ok: image%s dtype=%s state%s action%s ep=%s frame=%s",
            idx,
            tuple(sample[video_keys[0]].shape),
            sample[video_keys[0]].dtype,
            tuple(state.shape),
            tuple(action.shape),
            sample.get("episode_index"),
            sample.get("frame_index"),
        )

    logger.info("AV1 冒烟测试通过：%d 帧解码校验成功。", len(indices))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 + 冒烟测试 xarm_lift_medium")
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--root", default=str(DATASET_DIR))
    parser.add_argument("--skip-smoke", action="store_true", help="仅下载")
    parser.add_argument("--smoke-only", action="store_true", help="仅冒烟测试")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--video-backend",
        choices=["torchcodec", "pyav", "video_reader"],
        default=None,
        help="默认自动检测（torchcodec 可用则优先，否则 pyav）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    dataset = None
    if not args.smoke_only:
        dataset = download_dataset(args.repo_id, Path(args.root))
    if not args.skip_smoke:
        smoke_test_av1_decode(dataset, n_samples=args.n_samples, video_backend=args.video_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
