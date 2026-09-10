# LeRobot SmolVLA LoRA Fine-Tuning — Closed-Loop Verification Demo

> A verification demo for a **robotics VLA (Vision-Language-Action)** model.
> The goal is NOT to train a "production-ready" robot policy, but to prove with
> experimental evidence that the **LoRA fine-tuning pipeline of SmolVLA —
> train → save adapter → load adapter → inference — physically works end-to-end,
> and that the adapter genuinely contributes to the output.**

---

## What This Is

[SmolVLA](https://huggingface.co/lerobot/smolvla_base) is a lightweight VLA base
model from Hugging Face (~**450M** parameters), built on a SmolVLM2-500M
vision-language backbone plus a flow-matching action expert, designed for robotic
manipulation tasks and trainable/deployable on consumer GPUs.

This demo fine-tunes `lerobot/smolvla_base` with **LoRA** under the LeRobot
framework, and uses a series of probe experiments to verify that the adapter is
actually trained, saved, correctly loaded, and truly changes the output at
inference time.

## Why This Demo Exists

The most common pitfall in VLA fine-tuning is a **false positive** — logs show
`[OUTPUT PARITY] PASSED`, yet the LoRA adapter on/off switch has zero effect on
the output (`on vs off = 0.000e+00`).

This project uses deterministic probe experiments (fixed seeds, manual zeroing of
`lora_B`, etc.) to rule out random-initialization contamination and PEFT version
bugs, establishing the closed loop with hard evidence.

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows (development/verification environment) |
| GPU | NVIDIA RTX 3060 Laptop 6GB (or equivalent consumer GPU) |
| Python | 3.10+ |
| Deep learning | PyTorch + CUDA |
| Framework | [LeRobot](https://github.com/huggingface/lerobot) (with `[smolvla]` extras) + PEFT |
| VRAM | 6GB is sufficient (SmolVLA 450M + LoRA, ~1GB bf16 weights) |

> **Note**: 6GB is the *lower bound for "runnable"*, not a guarantee of training
> speed. The real bottlenecks are dataset size (official guidance suggests ~50
> episodes) and wall-clock time (20k steps ≈ 20+ hours), not VRAM.

## Quick Start

From the project root (`D:\Code demos\LeRobot\LeRobot-demo\`), only two commands:

```powershell
# 1. Set the Hugging Face mirror endpoint (accelerates model/dataset downloads)
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 2. Launch training (includes the full train → save → load → inference loop)
python -m training.train

Verification Results (Evidence Chain)
The following probes run under a fixed seed (torch.manual_seed(42) + CUDA seed)
to guarantee reproducibility and zero random contamination:

Probe	Result	Meaning
① LoRA layer count	37 layers	32 (q_proj/v_proj) + 5 projection matrices, matches official structure
② Gradient check	74 params with grad, 37 non-zero lora_B	Standard LoRA init, training is live
③ Training delta	1.73e+03	Weights are actually updated
④ Load check	37 layers, active_adapter=default	Adapter save/load works
⑤ base vs LoRA-on	5.214844e-01 (0.5215)	Adapter has real inference contribution (>0)
Control: base vs base	0.000000e+00	No random-init contamination; 0.5215 is clean LoRA contribution
Control: reload vs reload	0.000000e+00	Seed determinism holds across all random sources
Scale sanity check	std=0.2932, max_abs=1.8764	Output is O(1); 0.52/0.78 is substantive adaptation
⑥ on vs off (manual lora_B zeroing)	7.763672e-01 (0.7764)	LoRA switch truly changes output
Conclusion
The train → save → load → inference closed loop is verified successful, and
the adapter has substantive contribution to the output (not a no-op mount).