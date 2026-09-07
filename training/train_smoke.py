"""最小训练冒烟测试：验证 SmolVLA + LoRA 在 6GB 显存上能 forward/backward。

目标（fail-fast）：
    1. 验证 transformers 5.3.0 与 lerobot 0.5.1 的兼容性（SmolVLM 底层依赖 transformers）
    2. 验证 SmolVLA(450M) + LoRA + bf16 在 6GB 显存下可跑通 1 个 batch 的 forward/backward
    3. 验证数据链路：dataset → 注入占位指令 → preprocessor → forward

明确不做（留给后续 configs/train.py）：
    - 不做 config 系统、不做 wandb、不做评测、不做 checkpoint 保存
    - 不追求训练质量（归一化/超参/loss 数值仅作连通性验证）

用法（项目根目录，conda 环境 lerobot，需先设 HF 镜像下载 450M 权重）：
    $env:HF_ENDPOINT="https://hf-mirror.com"
    python training/train_smoke.py                       # 默认 bf16, batch=1, 3 步
    python training/train_smoke.py --steps 5 --dtype fp32
    python training/train_smoke.py --resize 256          # OOM 时降图像分辨率
    python training/train_smoke.py --chunk-size 50       # 还原 SmolVLA 默认 action 序列长度

注意：
    - xarm_lift_medium 是 VA 数据集（无语言指令），本脚本注入占位指令激活语言通路
    - xarm 每个 episode 仅 ~25 帧，SmolVLA 默认 chunk_size=50 会大量越界，故默认降到 10
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
# from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType


logger = logging.getLogger("training.train_smoke")

# --------------------------------------------------------------------------- #
# 常量 / 可配置项（与 data_pipeline/download.py 保持一致）
# --------------------------------------------------------------------------- #
DATASET_REPO_ID = "lerobot/xarm_lift_medium"
DATASET_DIR = Path("outputs/datasets") / "xarm_lift_medium"
PRETRAINED_PATH = "lerobot/smolvla_base"   # 450M 预训练 VLA，首次运行需联网下载
PLACEHOLDER_TASK = "Lift the object"       # 占位语言指令（注入以激活语言通路）


# --------------------------------------------------------------------------- #
# 诊断辅助
# --------------------------------------------------------------------------- #
def _mem_mb() -> str:
    """当前已分配显存 / 峰值显存（MB）。"""
    if not torch.cuda.is_available():
        return "n/a (CPU)"
    cur = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return f"alloc={cur:.0f}MB peak={peak:.0f}MB"


def _check_grads(policy) -> tuple[float, bool]:
    """返回 (梯度L2范数, 是否含NaN)。仅统计可训练参数。"""
    grad_norm, has_nan = 0.0, False
    for p in policy.parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                has_nan = True
            grad_norm += p.grad.norm().item() ** 2
    return grad_norm**0.5, has_nan


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_smoke(
    batch_size: int,
    steps: int,
    dtype_str: str,
    chunk_size: int,
    resize: int,
    peft_r: int,
    peft_alpha: int,
    video_backend: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 bf16，请改用 --dtype fp32（RTX 30 系支持 bf16，理论无碍）")

    # ---- 阶段 1：读元数据（本地，不联网），拿到 fps 供 delta_timestamps 使用 ----
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_DIR)
    fps = ds_meta.fps
    logger.info("元数据：fps=%d, episodes=%d, frames=%d", fps, ds_meta.total_episodes, ds_meta.total_frames)

    # ---- 阶段 2：加载预训练 SmolVLA（首次联网下载 450M 权重 + SmolVLM tokenizer）----
    # 这一步是「transformers 兼容性」的核心验证点：SmolVLM 底层走 transformers 加载。
    logger.info("加载 SmolVLA from_pretrained(%s)...（首次需下载权重）", PRETRAINED_PATH)
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)

    # 加载后调整训练期超参（均不改变任何层的参数 shape，安全）：
    #   chunk_size：action 序列长度。默认 50 对 25 帧/episode 的 xarm 会大量越界 padding
    #   resize：图像缩放尺寸。默认 512 吃显存，6GB 紧张时可降 256
    policy.config.chunk_size = chunk_size
    policy.config.n_action_steps = chunk_size
    policy.config.resize_imgs_with_padding = (resize, resize)
    cfg = policy.config  # 保存引用：wrap_with_peft 之后 policy.config 会变成 PeftConfig

    logger.info(
        "config: chunk_size=%d, resize=%s, max_state_dim=%d, max_action_dim=%d",
        cfg.chunk_size, cfg.resize_imgs_with_padding, cfg.max_state_dim, cfg.max_action_dim,
    )

    # ---- 阶段 3：构建 delta_timestamps 并加载数据集（本地，不联网）----
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action":
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        else:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]
    logger.info("delta_timestamps: %s", {k: len(v) for k, v in delta_timestamps.items()})

    dataset = LeRobotDataset(
        DATASET_REPO_ID, root=DATASET_DIR,
        delta_timestamps=delta_timestamps, video_backend=video_backend,
    )

    # ---- 阶段 4：预处理器（照搬官方 lerobot_train.py 的 override 逻辑）----
    # 关键：用 dataset.meta.stats 覆盖 normalizer，否则 SmolVLA base 的 32 维 stats
    # 会与 xarm 的 4 维 state/action shape 不匹配。
    # ---- 覆盖 input/output features（关键：用 policy.config，而非新建 config）----
    features = dataset_to_policy_features(ds_meta.features)

    policy.config.output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    policy.config.input_features = {
        k: ft for k, ft in features.items() if k not in policy.config.output_features
    }

    # ---- 自证：覆盖后应立即打印断言 ----
    print("input_features :", {k: v.shape for k, v in policy.config.input_features.items()})
    print("output_features:", {k: v.shape for k, v in policy.config.output_features.items()})
    assert "observation.image" in policy.config.input_features, "单相机特征未生效！"
    assert policy.config.output_features["action"].shape[0] == 4, "action 维度不是 4！"

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

    # ---- 阶段 5：LoRA 包装（冻结主干，只训 adapter）----
    policy.config.pretrained_path = "lerobot/smolvla_base"


    policy = policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "lora", "r": peft_r, "lora_alpha": peft_alpha}
    )
    policy = policy.to(device=device, dtype=dtype)
    policy.train()

    n_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logger.info("参数：可训练=%.2fM, 总=%.2fM (LoRA 冻结比 %.1f%%)",
                n_learnable / 1e6, n_total / 1e6, 100 * n_learnable / n_total)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4
    )

    # ---- 阶段 6：数据加载器 + 训练循环 ----
    # num_workers=0：规避 Windows 上 DataLoader 多进程(spawn) 的坑
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        t0 = time.perf_counter()
        batch = next(iter(dataloader))
        batch["task"] = [PLACEHOLDER_TASK] * batch_size  # 注入占位语言指令
        batch = preprocessor(batch)

        # batch = preprocessor(batch)

        with torch.autocast(device_type=device.type, dtype=dtype):
            loss, loss_dict = policy.forward(batch)


            loss.backward()

        grad_norm, has_nan = _check_grads(policy)
        optimizer.step()
        optimizer.zero_grad()

        dt = time.perf_counter() - t0
        logger.info(
            "step %d: loss=%.4f (finite=%s) grad_norm=%.4f (nan=%s) 显存[%s] 耗时=%.2fs",
            step, loss.item(), torch.isfinite(loss).item(), grad_norm, has_nan, _mem_mb(), dt,
        )

        if has_nan or not torch.isfinite(loss):
            raise RuntimeError("梯度或 loss 出现 NaN/Inf，请检查 bf16 是否可用或数据是否有异常")

    logger.info("✅ SmolVLA + LoRA 冒烟测试通过：%d 步 forward/backward 完成，兼容性验证 OK", steps)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SmolVLA + LoRA 最小训练冒烟测试")
    parser.add_argument("--steps", type=int, default=3, help="forward/backward 步数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--chunk-size", type=int, default=10, help="action 序列长度（默认 50，xarm 短 episode 建议 10）")
    parser.add_argument("--resize", type=int, default=512, help="图像缩放尺寸，OOM 时降到 256")
    parser.add_argument("--peft-r", type=int, default=64)
    parser.add_argument("--peft-alpha", type=int, default=64)
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav", "video_reader"], default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    run_smoke(
        batch_size=args.batch_size,
        steps=args.steps,
        dtype_str=args.dtype,
        chunk_size=args.chunk_size,
        resize=args.resize,
        peft_r=args.peft_r,
        peft_alpha=args.peft_alpha,
        video_backend=args.video_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
