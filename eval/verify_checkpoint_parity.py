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


def collect_lora_params(policy):
    """收集 policy 中所有 LoRA 权重张量（按参数名 -> 张量）。

    参数:
        policy: 训练完成、尚未卸载的 policy（SmolVLAPolicy 或其 PeftModel 包装）。

    返回:
        dict[str, torch.Tensor]: 参数名 -> 已 detach 的权重副本。
    """
    lora_params = {}
    for name, p in policy.named_parameters():
        if "lora" in name.lower():
            # detach + clone，避免后续 del 模型时引用失效
            lora_params[name] = p.detach().clone()
    return lora_params


def check_weight_parity(before_params, loaded_policy):
    """权重硬判据：逐一比对保存前与加载后的 LoRA 张量。

    参数:
        before_params: collect_lora_params 在保存前采集的结果。
        loaded_policy: 加载 adapter 后的模型（PeftModel）。

    返回:
        tuple[int, float]: (比对张量数, 最大绝对差)。
    """
    loaded_params = collect_lora_params(loaded_policy)

    if len(before_params) != len(loaded_params):
        print(
            f"[WEIGHT PARITY] FAILED: param count mismatch "
            f"(before={len(before_params)}, after={len(loaded_params)})"
        )
        return len(loaded_params), float("inf")

    max_abs_diff = 0.0
    for name, before_t in before_params.items():
        if name not in loaded_params:
            print(f"[WEIGHT PARITY] FAILED: missing param '{name}' after load")
            return len(loaded_params), float("inf")

        after_t = loaded_params[name]
        # 关键：bit-exact 比对必须在同一 dtype 下进行，
        # 否则 bf16 转 float 会引入转换噪声，破坏“逐位一致”结论。
        if not torch.equal(before_t, after_t):
            # 若 bit-exact 不成立，再量化最大绝对差，方便诊断差异量级
            diff = (before_t.float() - after_t.float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            print(
                f"[WEIGHT PARITY] MISMATCH: param '{name}' "
                f"max_abs_diff={diff:.3e}"
            )

    if max_abs_diff == 0.0:
        print(
            f"[WEIGHT PARITY] PASSED: {len(loaded_params)} LoRA tensors "
            f"are bit-exact (max_abs_diff=0.00e+00)"
        )
    else:
        print(f"[WEIGHT PARITY] FAILED: max_abs_diff={max_abs_diff:.3e}")

    return len(loaded_params), max_abs_diff


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


def check_output_parity(act_before, act_after, threshold=1e-3):
    """输出硬判据：比较保存前与加载后的推理输出。

    参数:
        act_before: 保存前 predict_action_chunk 输出。
        act_after: 加载后 predict_action_chunk 输出。
        threshold: 判定阈值，默认 1e-3（bf16 噪声量级远小于此）。

    返回:
        float: 输出最大绝对差。
    """
    out_diff = (act_before - act_after).abs().max().item()
    print(f"[OUTPUT PARITY] max_abs_diff = {out_diff:.3e}")

    if out_diff < threshold:
        print("[OUTPUT PARITY] PASSED: loaded model reproduces pre-save outputs")
    else:
        print(
            f"[OUTPUT PARITY] FAILED: out_diff={out_diff:.3e} exceeds "
            f"threshold={threshold:.1e}; LoRA may not be activated"
        )
    return out_diff


def verify_checkpoint_parity(before_params, act_before, batch, device, amp_dtype, loaded_policy):
    """仅对加载后模型做两级验证（保存前结果由调用方提前采好传入）。"""
    n_tensors, w_diff = check_weight_parity(before_params, loaded_policy)
    act_after = predict_frozen(loaded_policy, batch, device, amp_dtype)


    # ---- 判别实验：定位输出不齐的根因（假设 A/B/C） ----
    def _infer(policy):
        policy.eval()
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
            return policy.predict_action_chunk(copy.deepcopy(batch)).detach().float()

    y_on  = _infer(loaded_policy)     # adapter 激活
    y_on2 = _infer(loaded_policy)     # 自确定性（disable 之前测，两次都在激活态）
    loaded_policy.disable_adapter()
    y_off = _infer(loaded_policy)     # base-only
    # 不再 enable —— 后面直接打印，避免踩 enable_adapter 的坑


    print(f"[DIAG] self-determinism (on vs on2): {(y_on - y_on2).abs().max().item():.3e}")
    print(f"[DIAG] on vs off                  : {(y_on - y_off).abs().max().item():.3e}")
    print(f"[DIAG] act_before vs on           : {(act_before - y_on).abs().max().item():.3e}")
    print(f"[DIAG] act_before vs off          : {(act_before - y_off).abs().max().item():.3e}")
    print(f"[DIAG] act_after vs y_on (wrapper sanity): {(act_after - y_on).abs().max().item():.3e}")
    # ---- 判别实验结束 ----



    out_diff = check_output_parity(act_before, act_after)

    weight_ok = w_diff == 0.0
    output_ok = out_diff < 1e-3
    all_pass = weight_ok and output_ok

    print(
        f"[SUMMARY] weight_parity={'PASS' if weight_ok else 'FAIL'}, "
        f"output_parity={'PASS' if output_ok else 'FAIL'}, "
        f"overall={'PASS' if all_pass else 'FAIL'}"
    )
    return all_pass
