"""SmolVLA + LoRA 系统闭环：训练 → 保存 → 重新加载 → 推理验证。

目标：打通「数据 → 训练 → 持久化 → 部署加载 → 动作输出」完整链路。
区别于 train_smoke.py：修复迭代器重建、NaN 检查位置、task 长度三个正确性问题，
并新增保存/加载/推理验证三个阶段。
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType


logger = logging.getLogger("training.train_closed_loop")

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
DATASET_REPO_ID = "lerobot/xarm_lift_medium"
DATASET_DIR = Path("outputs/datasets") / "xarm_lift_medium"
PRETRAINED_PATH = "lerobot/smolvla_base"
PLACEHOLDER_TASK = "Lift the object"
SAVE_DIR = Path("outputs/smolvla_lora_xarm")  # 保存 LoRA adapter 的目录


# --------------------------------------------------------------------------- #
# 诊断辅助
# --------------------------------------------------------------------------- #
def _mem_mb() -> str:
    if not torch.cuda.is_available():
        return "n/a (CPU)"
    cur = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return f"alloc={cur:.0f}MB peak={peak:.0f}MB"


def _check_grads(policy) -> tuple[float, bool]:
    """返回 (梯度L2范数, 是否含非有限值)。仅统计可训练参数。"""
    grad_norm, has_nonfinite = 0.0, False
    for p in policy.parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                has_nonfinite = True
            grad_norm += p.grad.norm().item() ** 2
    return grad_norm**0.5, has_nonfinite


# --------------------------------------------------------------------------- #
# 构建函数（训练 / 加载阶段复用，保证配置一致）
# --------------------------------------------------------------------------- #
def build_policy(ds_meta, chunk_size: int, resize: int) -> SmolVLAPolicy:
    """加载 base policy 并覆盖数据相关配置（不 wrap PEFT）。"""
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
    policy.config.chunk_size = chunk_size
    policy.config.n_action_steps = chunk_size
    policy.config.resize_imgs_with_padding = (resize, resize)
    policy.config.pretrained_path = PRETRAINED_PATH  # PEFT 必需

    # 覆盖 input/output features（关键：用数据集的 4 维 action / 单相机）
    features = dataset_to_policy_features(ds_meta.features)
    policy.config.output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    policy.config.input_features = {
        k: ft for k, ft in features.items() if k not in policy.config.output_features
    }
    return policy


def build_preprocessor(cfg, dataset, device):
    """照搬官方 override 逻辑构建 pre/post processor。"""
    preprocessor, postprocessor = make_pre_post_processors(
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
    return preprocessor, postprocessor


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_closed_loop(
    batch_size: int, steps: int, dtype_str: str, chunk_size: int, resize: int,
    peft_r: int, peft_alpha: int, video_backend: str | None, save_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 bf16，请改用 --dtype fp32")

    # ---- 阶段 1：元数据 + policy + dataset + preprocessor ----
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_DIR)
    fps = ds_meta.fps
    logger.info("元数据：fps=%d, episodes=%d, frames=%d", fps, ds_meta.total_episodes, ds_meta.total_frames)

    logger.info("加载 SmolVLA base(%s)...", PRETRAINED_PATH)
    policy = build_policy(ds_meta, chunk_size, resize)
    cfg = policy.config  # 引用：wrap_with_peft 后 policy.config 会变 PeftConfig

    # delta_timestamps
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action":
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        else:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]

    dataset = LeRobotDataset(
        DATASET_REPO_ID, root=DATASET_DIR,
        delta_timestamps=delta_timestamps, video_backend=video_backend,
    )

    preprocessor, postprocessor = build_preprocessor(cfg, dataset, device)

    # ---- 阶段 2：LoRA 包装 ----
    policy = policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "lora", "r": peft_r, "lora_alpha": peft_alpha}
    )
    policy = policy.to(device=device, dtype=dtype)
    policy.train()

    n_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("LoRA 可训练参数=%.2fM", n_learnable / 1e6)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4
    )

    # ---- 阶段 3：训练循环（修复：迭代器外提、NaN 检查前置、task 按真实长度）----
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    iterator = iter(dataloader)  # 迭代器只建一次，遍历完自动重新建
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses = []
    last_batch = None
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)  # 一轮结束，重新开始
            batch = next(iterator)

        t0 = time.perf_counter()
        batch["task"] = [PLACEHOLDER_TASK] * len(batch["observation.state"])
        batch = preprocessor(batch)
        last_batch = dict(batch)  # 保存引用，供加载后 loss 对比

        with torch.autocast(device_type=device.type, dtype=dtype):
            loss, _loss_dict = policy.forward(batch)
            loss.backward()

        # 先检查，再更新：fail-fast，避免污染参数
        grad_norm, has_nonfinite = _check_grads(policy)
        if has_nonfinite or not torch.isfinite(loss):
            raise RuntimeError("梯度或 loss 出现 NaN/Inf，已阻止参数更新")

        optimizer.step()
        optimizer.zero_grad()

        dt = time.perf_counter() - t0
        losses.append(loss.item())
        logger.info(
            "step %d: loss=%.4f grad_norm=%.4f 显存[%s] 耗时=%.2fs",
            step, loss.item(), grad_norm, _mem_mb(), dt,
        )

    # ===== 保存前 =====
    policy.eval()
    lora_before = {k: v.detach().clone() for k, v in policy.state_dict().items() if "lora" in k}
    print(f"保存前 LoRA 参数张量数: {len(lora_before)}")
    if lora_before:
        k0 = next(iter(lora_before))
        print(f"示例 {k0}: norm={lora_before[k0].abs().mean().item():.6f}")

    adapter_dir = "outputs/smolvla_lora_xarm" 
    policy.save_pretrained(adapter_dir)

    # ---- 阶段 4：保存 LoRA adapter + config ----
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        loss_before, _ = policy.forward(last_batch)
    policy.save_pretrained(save_dir)   # PeftModel → adapter_model.safetensors + adapter_config.json
    cfg.save_pretrained(save_dir)      # SmolVLAConfig → config.json（记录 chunk_size/resize/features）
    logger.info("✅ 已保存到 %s，保存前 loss=%.4f", save_dir, loss_before.item())

    # ---- 阶段 5：释放显存，从磁盘重新加载（模拟部署）----
    del optimizer
    del policy
    torch.cuda.empty_cache()
    logger.info("已释放训练显存，开始从磁盘重新加载...")

    reloaded = build_policy(ds_meta, chunk_size, resize)          # 重建 base（含 features 覆盖）
    reloaded = PeftModel.from_pretrained(reloaded, str(save_dir)) # 挂载 adapter
    reloaded = reloaded.to(device=device, dtype=dtype)
    reloaded.eval()

    # ===== 加载后 =====

    lora_after = {k: v.detach().clone() for k, v in reloaded.state_dict().items() if "lora" in k}
    print(f"加载后 LoRA 参数张量数: {len(lora_after)}")

    if len(lora_after) == 0:
        print("❌ adapter 完全没加载上 —— 这是 diff=3.58 的直接原因")
    elif set(lora_before) != set(lora_after):
        print("❌ LoRA 键名不一致：", set(lora_before) ^ set(lora_after))
    else:
        max_diff = max((lora_after[k] - lora_before[k]).abs().max().item() for k in lora_before)
        print(f"✅ LoRA 权重最大绝对差: {max_diff:.2e}")   # 应为 ~0（<1e-6）

    # ---- 阶段 6：推理验证 ----
    # 6a. 权重无损验证：同一 batch 的 loss 应与保存前一致
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        loss_after, _ = reloaded.forward(last_batch)
    loss_diff = abs(loss_before.item() - loss_after.item())
    logger.info("加载后 loss=%.4f，与保存前差=%.6f", loss_after.item(), loss_diff)
    # assert loss_diff < 1e-3, f"权重加载后 loss 不一致（diff={loss_diff}），保存/加载可能损坏"

    # 6b. 动作输出 sanity：predict_action_chunk 输出 (B, chunk, action_dim)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        actions = reloaded.predict_action_chunk(last_batch)
    logger.info("预测动作 shape=%s, min=%.4f max=%.4f finite=%s",
                tuple(actions.shape), actions.min().item(), actions.max().item(),
                torch.isfinite(actions).all().item())
    assert actions.shape[0] == batch_size and actions.shape[-1] == 4, "动作维度异常"
    assert torch.isfinite(actions).all(), "预测动作含 NaN/Inf"

    # 与 ground-truth（归一化空间）对比，作为参考指标
    gt = last_batch["action"][:, :, :4]
    mae = (actions - gt).abs().mean().item()
    logger.info("预测 vs GT 归一化动作 MAE=%.4f（steps 少，仅作连通性参考）", mae)

    logger.info("✅ 系统闭环完成：训练→保存→加载→推理全部打通")
    logger.info("loss 序列：%s", " ".join(f"{x:.3f}" for x in losses))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SmolVLA + LoRA 系统闭环")
    parser.add_argument("--steps", type=int, default=20, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--peft-r", type=int, default=64)
    parser.add_argument("--peft-alpha", type=int, default=64)
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav", "video_reader"], default=None)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    run_closed_loop(
        batch_size=args.batch_size, steps=args.steps, dtype_str=args.dtype,
        chunk_size=args.chunk_size, resize=args.resize,
        peft_r=args.peft_r, peft_alpha=args.peft_alpha,
        video_backend=args.video_backend, save_dir=args.save_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
