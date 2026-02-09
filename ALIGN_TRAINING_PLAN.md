# Align 阶段训练计划（DINOv3 + Qwen3VL TextModel，仅训练 Projector）

本文档是本仓库中 `align` 阶段训练的实用操作手册。

## 1. 目标

总目标保持不变：`stage=align` 只训练 `projector`，并冻结视觉骨干和 LLM 骨干。

本次方案改为：
- 视觉骨干：`DINOv3`
- 语言骨干：`Qwen3VL` 中抽取的 `TextModel`

关键参考：
- `prismatic/models/vlms/prismatic.py:129`
- `prismatic/models/vlms/prismatic.py:140`
- `prismatic/models/vlms/prismatic.py:142`
- `prismatic/models/vlms/prismatic.py:145`

## 2. 前置条件 ✅

1. CUDA 和 GPU 可用。
2. 当前环境中 `torchrun` 可正常运行。
3. 已准备 Hugging Face Token（如使用 gated/private 模型）。
4. 数据集路径有效。

脚本入口参考：
- `scripts/pretrain.py:18`
- `scripts/pretrain.py:117`

## 3. Hugging Face Token 配置 ✅

二选一：

1. 在仓库根目录放置 `.hf_token`（单行文本 token）。
2. 设置环境变量，并通过 `--hf_token` 传入变量名。

参考：
- `scripts/pretrain.py:73`
- `scripts/pretrain.py:74`
- `scripts/pretrain.py:128`

## 4. 模型适配改动清单（DINOv3 + Qwen3VL TextModel）

当前仓库默认注册的是 DINOv2 / CLIP / SigLIP 与 LLaMA2 / Mistral / Phi，
因此要先完成以下适配，才能启动新方案训练：

1. 视觉骨干接入（DINOv3）
- 新增或扩展视觉 backbone 封装（建议放在 `prismatic/models/backbones/vision/`）。
- 在 `prismatic/models/materialize.py` 的 `VISION_BACKBONES` 注册 `dinov3-*` 标识。

2. 语言骨干接入（Qwen3VL TextModel）
- 新增 LLM backbone 封装（建议放在 `prismatic/models/backbones/llm/`），只暴露 text model 作为解码骨干。
- 在 `prismatic/models/materialize.py` 的 `LLM_BACKBONES` 注册 `qwen3vl-text-*` 标识。

3. 组合模型配置
- 在 `prismatic/conf/models.py` 新增一个 model config（例如 `dinov3-qwen3vltext-align`）。
- 配置项至少包括：
  - `vision_backbone_id`
  - `llm_backbone_id`
  - `arch_specifier`
  - `align_*` 一组训练超参数
- 同时在 `ModelRegistry` 中注册该 model config，保证 `--model.type` 可识别。

4. 冻结行为核验
- 训练日志中应明确显示 `align` 阶段仅 projector 可训练。
- 核验入口：`vlm.freeze_backbones(cfg.stage)`（`scripts/pretrain.py:169`）。

## 5. 数据集配置检查

检查配置中的根目录和 align 阶段组件：
- `dataset_root_dir`
- `align_stage_components`

参考：
- `prismatic/conf/datasets.py:21`
- `prismatic/conf/datasets.py:27`
- `prismatic/conf/datasets.py:35`

如果你的数据集不在默认路径，可通过以下参数覆盖：
- `--dataset.dataset_root_dir /your/dataset/root`

## 6. Align 需要调的超参数

`align` 使用 `model.align_*` 字段：
- `align_epochs`
- `align_max_steps`
- `align_global_batch_size`
- `align_per_device_batch_size`
- `align_learning_rate`
- `align_weight_decay`
- `align_max_grad_norm`
- `align_lr_scheduler_type`
- `align_warmup_ratio`
- `align_train_strategy`

参考：
- `prismatic/conf/models.py:37`
- `prismatic/conf/models.py:49`
- `prismatic/conf/models.py:89`
- `prismatic/conf/models.py:100`

## 7. 冒烟测试（单卡、短跑）

先跑最小化验证（在新 model config 注册完成后执行）：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
  --stage align \
  --model.type dinov3-qwen3vltext-align \
  --dataset.type llava-v15 \
  --dataset.dataset_root_dir /your/dataset/root \
  --run_root_dir /home/max/openvla \
  --max_steps 20 \
  --trackers '["jsonl"]'
```

原因：
- `--stage align` 会触发 `freeze_backbones("align")`，仅训练 projector。

参考：
- `scripts/pretrain.py:168`
- `scripts/pretrain.py:169`
- `scripts/pretrain.py:225`

## 8. 正式训练（多卡）

冒烟测试通过后，按 GPU 数量扩展：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --stage align \
  --model.type dinov3-qwen3vltext-align \
  --dataset.type llava-v15 \
  --dataset.dataset_root_dir /your/dataset/root \
  --run_root_dir /your/runs/root \
  --trackers '["jsonl","wandb"]'
```

## 9. 训练中监控项

1. 训练 loss 应下降且整体稳定。
2. GPU 显存应处于可控范围。
3. 日志应显示 projector 可训练，vision 与 LLM 冻结。

参考：
- `prismatic/models/vlms/prismatic.py:151`
- `prismatic/models/vlms/prismatic.py:153`

## 10. Checkpoint 校验

align 阶段保存的模型状态应以可训练模块为主（重点是 projector）。

参考：
- `prismatic/training/strategies/ddp.py:37`
- `prismatic/training/strategies/ddp.py:40`
- `prismatic/training/strategies/fsdp.py:101`
- `prismatic/training/strategies/fsdp.py:110`

## 11. 进入 Finetune 阶段

align 结束后，使用 align checkpoint 进入 finetune 阶段。

参考：
- `scripts/pretrain.py:65`
- `scripts/pretrain.py:67`
- `scripts/pretrain.py:173`
