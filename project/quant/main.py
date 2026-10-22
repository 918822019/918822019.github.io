from transformers import AutoModelForCausalLM

model_path = "../../data/models/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 打印模型树
def print_model_tree(model, indent=0):
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{'  ' * indent}{name}: {module.__class__.__name__} | params: {params:,}")
        if list(module.children()):
            print_model_tree(module, indent + 1)

print_model_tree(model)