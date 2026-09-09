"""
eval/verify_checkpoint_parity.py

Checkpoint 一致性验证：证明「训练后保存 → 加载 → 推理」链路中，
加载后的模型输出与保存前逐位一致（bit-exact），即 LoRA 权重确实被应用。

两个硬判据：
  1. 权重一致性：保存前 LoRA 张量 vs 加载后 LoRA 张量，torch.equal 逐一比对
  2. 输出一致性：同一冻结 batch 下，保存前 vs 加载后 predict_action_chunk 输出最大绝对差

loss 不作为硬判据，仅作参考打印。
"""

import copy
import logging
import torch


def predict_frozen(policy, batch, device, amp_dtype):
    """在 eval + no_grad + autocast 下对冻结 batch 做推理，返回 float 输出。

    参数:
        policy: 模型（eval 状态由调用方保证或此处强制）。
        batch: 冻结的输入 batch（每次调用内部 deepcopy，防就地修改）。
        device: 设备（用于 autocast 的 device_type）。
        amp_dtype: 混合精度 dtype（bf16）。

    返回:
        torch.Tensor: predict_action_chunk 的输出，已 detach 并转 float。
    """
    policy.eval()
    # 关键：固定 flow matching 的噪声采样，让 forward 可复现
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
        # 关键：每次调用前 fresh deepcopy，防止 predict 内部就地修改 batch
        action = policy.predict_action_chunk(copy.deepcopy(batch))
    return action.detach().float()
