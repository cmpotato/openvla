from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText

MODEL_PATH = "/home/max/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"


def format_params(num_params: int) -> str:
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f} B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f} M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f} K"
    return str(num_params)


def module_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


print(f"正在从本地加载模型配置: {MODEL_PATH}")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

print("正在构建无内存占用的模型骨架...")
with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

print("\n--- 模型基础信息 (meta device) ---")
print(f"模型参数设备: {model.device}")
print(f"任意一个参数的设备: {next(model.parameters()).device}")

module_param_counts = []
for name, module in model.named_modules():
    num_params = module_param_count(module)
    if num_params > 0:
        module_param_counts.append((name, num_params))
module_param_counts.sort(key=lambda x: x[0])

print("\n--- 各模块参数量（完整路径）---")
print(f"{'模块名称':<80} {'参数量':>20}")
print(f"{'-'*80} {'-'*20}")
for name, count in module_param_counts:
    print(f"{name:<80} {format_params(count):>20}")

total_model_params = module_param_count(model)
print(f"{'-'*80} {'-'*20}")
print(f"{'总参数量':<80} {format_params(total_model_params):>20}")

visual_params = 0
language_model_params = 0
lm_head_params = 0
merger_params = 0

for name, module in model.named_modules():
    count = module_param_count(module)
    if name in {"model.visual", "visual"}:
        visual_params = count
    elif name in {"model.language_model", "language_model", "model.text_model", "text_model"}:
        language_model_params = count
    elif name == "lm_head":
        lm_head_params = count
    elif "merger" in name:
        merger_params += count

print("\n--- 主要模块参数量对比 ---")
print(f"{'视觉编码器(Visual)':<32} {format_params(visual_params):>15}")
print(f"{'Merger相关模块(总和)':<32} {format_params(merger_params):>15}")
print(f"{'语言模型主体(Text/LM)':<32} {format_params(language_model_params):>15}")
print(f"{'语言模型输出层(lm_head)':<32} {format_params(lm_head_params):>15}")
if language_model_params or lm_head_params:
    print(f"{'语言模型总计(主体+lm_head)':<32} {format_params(language_model_params + lm_head_params):>15}")

print("\n--- 一级子模块参数量（便于快速定位）---")
for child_name, child_module in model.named_children():
    print(f"{child_name:<32} {format_params(module_param_count(child_module)):>15}")
