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
    collect_lora_params, check_weight_parity,
    predict_frozen, check_output_parity, verify_checkpoint_parity,
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


# # --------------------------------------------------------------------------- #
# # Checkpoint parity verification (HARD criteria)
# # --------------------------------------------------------------------------- #
# def collect_lora_params(policy):
#     """Collect all LoRA weight tensors from a policy.

#     Args:
#         policy: The trained policy (SmolVLAPolicy or its PeftModel wrapper).

#     Returns:
#         dict[str, torch.Tensor]: parameter name -> detached clone.
#     """
#     state = policy.state_dict()
#     return {k: v.detach().clone() for k, v in state.items() if "lora" in k}


# def check_weight_parity(before_params, loaded_policy):
#     """Hard check: compare LoRA tensors before save vs after load.

#     Args:
#         before_params: Result of collect_lora_params() captured before saving.
#         loaded_policy: The reloaded model (PeftModel).

#     Returns:
#         tuple[int, float]: (number of compared tensors, max absolute difference).
#     """
#     loaded_params = collect_lora_params(loaded_policy)

#     if len(before_params) != len(loaded_params):
#         logger.warning(
#             "Weight parity FAILED: tensor count mismatch (before=%d, after=%d)",
#             len(before_params), len(loaded_params),
#         )
#         return len(loaded_params), float("inf")

#     max_abs_diff = 0.0
#     for name, before_tensor in before_params.items():
#         if name not in loaded_params:
#             logger.warning("Weight parity FAILED: missing tensor '%s' after load", name)
#             return len(loaded_params), float("inf")

#         after_tensor = loaded_params[name]
#         # Compare in the original dtype so bf16 stays bit-exact (no float-cast noise).
#         if not torch.equal(before_tensor, after_tensor):
#             diff = (before_tensor.float() - after_tensor.float()).abs().max().item()
#             max_abs_diff = max(max_abs_diff, diff)
#             logger.warning(
#                 "Weight parity MISMATCH: tensor '%s' max_abs_diff=%.3e", name, diff,
#             )

#     if max_abs_diff == 0.0:
#         logger.info(
#             "Weight parity PASSED: %d LoRA tensors are bit-exact (max_abs_diff=0.00e+00)",
#             len(loaded_params),
#         )
#     else:
#         logger.warning("Weight parity FAILED: max_abs_diff=%.3e", max_abs_diff)

#     return len(loaded_params), max_abs_diff


# def predict_frozen(policy, batch, device, amp_dtype):
#     """Run predict_action_chunk under eval + no_grad + autocast on a frozen batch.

#     Args:
#         policy: The model (set to eval mode here).
#         batch: The frozen input batch; each call deep-copies it to avoid in-place mutation.
#         device: torch.device used for autocast device_type.
#         amp_dtype: Mixed-precision dtype (e.g. torch.bfloat16).

#     Returns:
#         torch.Tensor: action chunk output, detached and cast to float32.
#     """
#     policy.eval()
#     with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
#         actions = policy.predict_action_chunk(copy.deepcopy(batch))
#     return actions.detach().float()


# def check_output_parity(act_before, act_after, threshold=1e-3):
#     """Hard check: compare pre-save vs post-load action outputs.

#     Args:
#         act_before: Reference output captured before saving.
#         act_after: Output produced by the reloaded model.
#         threshold: Pass threshold (bf16 numerical noise is far below this).

#     Returns:
#         float: Max absolute difference between the two outputs.
#     """
#     out_diff = (act_before - act_after).abs().max().item()
#     if out_diff < threshold:
#         logger.info("Output parity PASSED: max_abs_diff=%.3e (< %.1e)", out_diff, threshold)
#     else:
#         logger.warning(
#             "Output parity FAILED: max_abs_diff=%.3e exceeds %.1e; LoRA may not be activated",
#             out_diff, threshold,
#         )
#     return out_diff


# def verify_checkpoint_parity(before_params, act_before, batch, device, amp_dtype, loaded_policy):
#     """Run both hard parity checks against the reloaded model.

#     Args:
#         before_params: LoRA tensors collected before saving.
#         act_before: Reference action output captured before saving.
#         batch: Frozen batch used for the output comparison.
#         device: torch.device.
#         amp_dtype: Mixed-precision dtype.
#         loaded_policy: The reloaded model (PeftModel).

#     Returns:
#         bool: True if both checks pass.
#     """
#     _, weight_diff = check_weight_parity(before_params, loaded_policy)
#     act_after = predict_frozen(loaded_policy, batch, device, amp_dtype)
#     out_diff = check_output_parity(act_before, act_after)

#     weight_ok = weight_diff == 0.0
#     output_ok = out_diff < 1e-3
#     all_pass = weight_ok and output_ok

#     logger.info(
#         "Parity summary: weight=%s, output=%s, overall=%s",
#         "PASS" if weight_ok else "FAIL",
#         "PASS" if output_ok else "FAIL",
#         "PASS" if all_pass else "FAIL",
#     )
#     return all_pass


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
    ##########################################################################

    lora_snapshot_pre = collect_lora_params(policy)
    logger.info("[PROBE] LoRA snapshot before training: %d tensors", len(lora_snapshot_pre))
    ###########################################################################
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

    # Capture LoRA weights as the bit-exact reference BEFORE saving.
    lora_before = collect_lora_params(policy)
    logger.info("Captured %d LoRA tensors before save", len(lora_before))

    #################################################################################
    delta_train = sum(
        (lora_before[k].float() - lora_snapshot_pre[k].float()).abs().sum().item()
        for k in lora_snapshot_pre
    )
    logger.info("[PROBE] LoRA weight delta from training: %.6e", delta_train)
    if delta_train == 0.0:
        logger.error("[PROBE] LoRA weights unchanged — optimizer never updated them")

    ############################################################################

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

    ############################################################################

    # ---- Phase 6: verify checkpoint parity (HARD criteria) ----
    parity_ok = verify_checkpoint_parity(lora_before, act_before, last_batch, device, dtype, reloaded)

    ########################################################################################
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

    if parity_ok:
        logger.info("Closed loop complete: train -> save -> load -> inference (parity PASSED)")
    else:
        logger.error("Closed loop complete, but checkpoint parity FAILED; see checks above")


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
