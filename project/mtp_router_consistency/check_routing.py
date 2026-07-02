import torch
import sys
sys.path.insert(0, r'D:\918822019.github.io\project\mtp_router_consistency')
from model_utils import load_model_and_tokenizer
from config import Config

cfg = Config()
model, tokenizer, device = load_model_and_tokenizer(cfg.model_path, torch_dtype=torch.bfloat16)

prompt = 'def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)'
inp = tokenizer(prompt, return_tensors='pt')
input_ids = inp.input_ids.to(device)

outputs = model(input_ids=input_ids, output_router_logits=True, use_cache=False, return_dict=True)
router = outputs.router_logits[-2][0].squeeze(0)

ids = input_ids[0]
from collections import Counter
cnt = Counter(ids.tolist())
repeated = {k: v for k, v in cnt.items() if v >= 2 and k not in [tokenizer.bos_token_id, tokenizer.eos_token_id]}

print("=== Same token ID, different positions - routing consistency ===")
for tok_id in sorted(repeated, key=lambda x: -repeated[x])[:5]:
    tok = tokenizer.decode(tok_id)
    positions = [i for i, t in enumerate(ids) if t == tok_id]
    if len(positions) < 2:
        continue
    print(f"\ntoken '{tok}' (id={tok_id}) at positions {positions}")
    logits_at_pos = [router[p] for p in positions]
    cos_sum = 0
    iou_sum = 0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            cos = torch.nn.functional.cosine_similarity(logits_at_pos[i].float(), logits_at_pos[j].float(), dim=0)
            iou = len(set(logits_at_pos[i].float().topk(8).indices.tolist()) & 
                     set(logits_at_pos[j].float().topk(8).indices.tolist())) / 8
            cos_sum += cos
            iou_sum += iou
            print(f"    pos {positions[i]} vs {positions[j]}: cos={cos:.4f} top8_iou={iou:.3f}")
    n_pairs = len(positions) * (len(positions) - 1) // 2
    print(f"    avg cos={cos_sum/n_pairs:.4f} avg iou={iou_sum/n_pairs:.3f}")
