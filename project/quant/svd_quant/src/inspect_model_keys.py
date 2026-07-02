from safetensors import safe_open
import json

model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"

# 加载索引
with open(f"{model_path}/model.safetensors.index.json", "r") as f:
    index_data = json.load(f)

weight_map = index_data["weight_map"]

# 打印一些示例权重名称
print("示例权重名称:")
for i, name in enumerate(list(weight_map.keys())[:50]):
    print(f"  {name}")

# 分析MoE相关的权重
print("\nMoE相关权重:")
moe_keys = [k for k in weight_map.keys() if "experts" in k or "moe" in k.lower()]
for key in moe_keys[:30]:
    print(f"  {key}")

# 分析共享专家
print("\n共享专家相关权重:")
shared_keys = [k for k in weight_map.keys() if "shared" in k]
for key in shared_keys[:20]:
    print(f"  {key}")

# 分析注意力层
print("\n注意力层相关权重:")
attn_keys = [k for k in weight_map.keys() if "self_attn" in k]
for key in attn_keys[:20]:
    print(f"  {key}")