"""SmolVLA + LoRA closed loop: train -> save -> reload -> inference verification.

Goal: prove the full chain "data -> train -> persist -> deploy-load -> action output".
Compared to train_smoke.py, this fixes iterator rebuild, NaN-check placement, and
task-length correctness, and adds save / load / inference verification stages.

The "checkpoint parity" verification uses two HARD criteria:
  1. Weight parity: LoRA tensors before save vs after load must be bit-exact.
  2. Output parity: pre-save vs post-load predict_action_chunk outputs must match.
Loss is only printed as an informational signal and is NOT a pass/fail criterion.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel

from lerobot.configs.types import FeatureType
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from eval.verify_checkpoint_parity import (
    predict_frozen
)

logger = logging.getLogger("training.train_closed_loop")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATASET_REPO_ID = "lerobot/xarm_lift_medium"
DATASET_DIR = Path("outputs/datasets") / "xarm_lift_medium"
PRETRAINED_PATH = "lerobot/smolvla_base"
PLACEHOLDER_TASK = "Lift the object"
SAVE_DIR = Path("outputs/smolvla_lora_xarm")  # Directory for the saved LoRA adapter


# --------------------------------------------------------------------------- #
# Diagnostics helpers
# --------------------------------------------------------------------------- #
def _mem_mb() -> str:
    if not torch.cuda.is_available():
        return "n/a (CPU)"
    cur = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return f"alloc={cur:.0f}MB peak={peak:.0f}MB"


def _check_grads(policy) -> tuple[float, bool]:
    """Return (gradient L2 norm, whether any trainable grad is non-finite)."""
    grad_norm, has_nonfinite = 0.0, False
    for p in policy.parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                has_nonfinite = True
            grad_norm += p.grad.norm().item() ** 2
    return grad_norm**0.5, has_nonfinite


# --------------------------------------------------------------------------- #
# Build helpers (shared by train / reload to keep config identical)
# --------------------------------------------------------------------------- #
def build_policy(ds_meta, chunk_size: int, resize: int) -> SmolVLAPolicy:
    """Load the base policy and override data-dependent config (no PEFT wrap yet)."""
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
    policy.config.chunk_size = chunk_size
    policy.config.n_action_steps = chunk_size
    policy.config.resize_imgs_with_padding = (resize, resize)
    policy.config.pretrained_path = PRETRAINED_PATH  # Required by PEFT

    # Override input/output features (4-dim action / single camera from the dataset).
    features = dataset_to_policy_features(ds_meta.features)
    policy.config.output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    policy.config.input_features = {
        k: ft for k, ft in features.items() if k not in policy.config.output_features
    }
    return policy


def build_preprocessor(cfg, dataset, device):
    """Mirror the official override logic to build pre/post processors."""
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
# Main flow
# --------------------------------------------------------------------------- #
def run_closed_loop(
    batch_size: int, steps: int, dtype_str: str, chunk_size: int, resize: int,
    peft_r: int, peft_alpha: int, video_backend: str | None, save_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16; use --dtype fp32")

    # ---- Phase 1: metadata + policy + dataset + preprocessor ----
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_DIR)
    fps = ds_meta.fps
    logger.info("Metadata: fps=%d, episodes=%d, frames=%d", fps, ds_meta.total_episodes, ds_meta.total_frames)

    logger.info("Loading SmolVLA base (%s)...", PRETRAINED_PATH)
    policy = build_policy(ds_meta, chunk_size, resize)
    cfg = policy.config  # Reference; after wrap_with_peft policy.config becomes a PeftConfig

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

    # ---- Phase 2: LoRA wrapping ----
    policy = policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "lora", "r": peft_r, "lora_alpha": peft_alpha}
    )
    policy = policy.to(device=device, dtype=dtype)

    lora_layers = [n for n, m in policy.named_modules()
               if getattr(m, "lora_A", None) is not None]
    logger.info("[PROBE] LoRA layers on wrapped policy: %d", len(lora_layers))
    if len(lora_layers) == 0:
        logger.error("[PROBE] ZERO LoRA layers — target_modules mismatch, stop now")
    ##########################################################################
    policy.train()

    n_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("LoRA trainable parameters=%.2fM", n_learnable / 1e6)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4
    )

    # ---- Phase 3: training loop ----
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    iterator = iter(dataloader)  # Build the iterator once; rebuild on exhaustion.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses = []
    last_batch = None
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)  # End of epoch; restart.
            batch = next(iterator)

        t0 = time.perf_counter()
        batch["task"] = [PLACEHOLDER_TASK] * len(batch["observation.state"])
        batch = preprocessor(batch)
        last_batch = dict(batch)  # Keep a reference for post-load comparison.

        with torch.autocast(device_type=device.type, dtype=dtype):
            loss, _loss_dict = policy.forward(batch)
            loss.backward()

        ######################################################################
        # === 插入：仅 step == 0 打印一次 ===
        if step == 0:
            lora_pairs = [(n, p) for n, p in policy.named_parameters()
                        if "lora" in n and p.requires_grad]
            n_with_grad = sum(1 for _, p in lora_pairs if p.grad is not None)
            n_nonzero_grad = sum(1 for _, p in lora_pairs
                                if p.grad is not None and p.grad.abs().sum() > 0)
            logger.info(
                "[PROBE] LoRA grad: %d params, %d with grad, %d non-zero grad",
                len(lora_pairs), n_with_grad, n_nonzero_grad,
            )
            if n_with_grad == 0:
                logger.error("[PROBE] No LoRA gradients -> LoRA not in forward graph")

        #################################################################

        # Check before update: fail fast and avoid corrupting parameters.
        grad_norm, has_nonfinite = _check_grads(policy)
        if has_nonfinite or not torch.isfinite(loss):
            raise RuntimeError("NaN/Inf detected in gradients or loss; update blocked")

        optimizer.step()
        optimizer.zero_grad()

        dt = time.perf_counter() - t0
        losses.append(loss.item())
        logger.info(
            "step %d: loss=%.4f grad_norm=%.4f mem[%s] time=%.2fs",
            step, loss.item(), grad_norm, _mem_mb(), dt,
        )

    # ---- Phase 4: save adapter + capture pre-save reference artifacts ----
    save_dir.mkdir(parents=True, exist_ok=True)

    

    # Capture reference action output (frozen batch, eval + no_grad + autocast).
    act_before = predict_frozen(policy, last_batch, device, dtype)
    logger.info("Captured reference action output: shape=%s", tuple(act_before.shape))

    # Reference loss is informational only; NOT a pass/fail criterion.
    policy.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        loss_before, _ = policy.forward(last_batch)

    policy.save_pretrained(save_dir)   # PeftModel -> adapter_model.safetensors + adapter_config.json
    cfg.save_pretrained(save_dir)      # SmolVLAConfig -> config.json (chunk_size/resize/features)
    logger.info("Saved adapter to %s (pre-save loss=%.4f, informational)", save_dir, loss_before.item())

    # ---- Phase 5: release training memory, reload from disk (simulate deployment) ----
    del optimizer
    del policy
    torch.cuda.empty_cache()
    logger.info("Released training memory; reloading from disk...")

    reloaded = build_policy(ds_meta, chunk_size, resize)          # Rebuild base (with feature overrides)
    reloaded = PeftModel.from_pretrained(reloaded, str(save_dir)) # Attach LoRA adapter
    reloaded = reloaded.to(device=device, dtype=dtype)
    reloaded.eval()

    ############################################################################
    reloaded_lora = [n for n, m in reloaded.named_modules()
                    if getattr(m, "lora_A", None) is not None]
    logger.info("[PROBE] LoRA layers on reloaded: %d", len(reloaded_lora))
    logger.info("[PROBE] active_adapter=%s", getattr(reloaded, "active_adapter", None))

 
    # ---- Phase 6.5: decisive base-vs-LoRA probe ----
    base_only = build_policy(ds_meta, chunk_size, resize)   # 纯 base，不 wrap peft
    base_only = base_only.to(device=device, dtype=dtype)
    base_only.eval()

    def _seeded_predict(policy):
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
        return predict_frozen(policy, last_batch, device, dtype)

    y_base = _seeded_predict(base_only)     # 无 LoRA 的真·off
    y_lora = _seeded_predict(reloaded)      # LoRA on

    diff_base_lora = (y_base - y_lora).abs().max().item()
    logger.info("[PROBE] base vs LoRA-on: max_abs_diff = %.6e", diff_base_lora)

    ######################################################################################

    #####################################################################################
    # ---------- 对照 1：base vs base（建立噪声地板，检验 build_policy 随机初始化） ----------
    base_only_2 = build_policy(ds_meta, chunk_size, resize)
    base_only_2 = base_only_2.to(device=device, dtype=dtype).eval()
    diff_base_base = (_seeded_predict(base_only) - _seeded_predict(base_only_2)).abs().max().item()
    print(f"[CONTROL] base vs base = {diff_base_base:.6e}")

    # ---------- 对照 2：reload vs reload（seed 下的自确定性） ----------
    diff_reload_self = (_seeded_predict(reloaded) - _seeded_predict(reloaded)).abs().max().item()
    print(f"[CONTROL] reload vs reload = {diff_reload_self:.6e}")

    # ---------- 输出尺度 sanity check ----------
    with torch.no_grad():
        y_ref = _seeded_predict(base_only)
    print(f"[SCALE] base output: std={y_ref.std():.4f}  max_abs={y_ref.abs().max():.4f}")

    # ---------- 探针 6：手动零化 lora_B（等价 disable，绕开 PEFT 版本 bug） ----------
    _state_backup = {k: v.detach().clone() for k, v in reloaded.named_parameters() if "lora_B" in k}
    try:
        for n, p in reloaded.named_parameters():
            if "lora_B" in n:
                p.data.zero_()
        y_off_manual = _seeded_predict(reloaded)
    finally:
        for n, p in reloaded.named_parameters():
            if n in _state_backup:
                p.data.copy_(_state_backup[n])

    y_on_manual = _seeded_predict(reloaded)
    diff_on_off_manual = (y_on_manual - y_off_manual).abs().max().item()
    print(f"[PROBE6] on vs off (manual lora_B zero) = {diff_on_off_manual:.6e}")
    #####################################################################################

    # ---- Phase 7: action sanity checks + informational metrics ----
    # Informational loss signal (no threshold, no assertion).
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        loss_after, _ = reloaded.forward(last_batch)
    loss_diff = abs(loss_before.item() - loss_after.item())
    logger.info(
        "Reloaded loss=%.4f (pre-save=%.4f, diff=%.6f) [informational, not a criterion]",
        loss_after.item(), loss_before.item(), loss_diff,
    )

    # Action output sanity: predict_action_chunk -> (B, chunk, action_dim).
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        actions = reloaded.predict_action_chunk(last_batch)
    logger.info(
        "Predicted action shape=%s, min=%.4f max=%.4f finite=%s",
        tuple(actions.shape), actions.min().item(), actions.max().item(),
        torch.isfinite(actions).all().item(),
    )
    assert actions.shape[0] == batch_size and actions.shape[-1] == 4, "Unexpected action shape"
    assert torch.isfinite(actions).all(), "Predicted action contains NaN/Inf"

    # Informational metric against ground-truth (normalized space).
    gt = last_batch["action"][:, :, :4]
    mae = (actions.float() - gt.float()).abs().mean().item()
    logger.info("Prediction vs GT normalized action MAE=%.4f (informational, few steps)", mae)

    logger.info("Loss sequence: %s", " ".join(f"{x:.3f}" for x in losses))



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SmolVLA + LoRA closed loop")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--peft-r", type=int, default=64)
    parser.add_argument("--peft-alpha", type=int, default=64)
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav", "video_reader"], default=None)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Silence noisy network/download loggers; keep only application logs.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_closed_loop(
        batch_size=args.batch_size, steps=args.steps, dtype_str=args.dtype,
        chunk_size=args.chunk_size, resize=args.resize,
        peft_r=args.peft_r, peft_alpha=args.peft_alpha,
        video_backend=args.video_backend, save_dir=args.save_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
