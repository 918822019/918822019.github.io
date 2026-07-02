import json
idx_path = "data/models/Mellum2-12B-A2.5B-Base-Pretrain/model.safetensors.index.json"
with open(idx_path) as f:
    idx = json.load(f)

all_keys = list(idx["weight_map"].keys())
print(f"Total keys: {len(all_keys)}")

mtp_keys = [k for k in all_keys if "mtp" in k.lower()]
print(f"MTP keys: {len(mtp_keys)}")
for k in mtp_keys:
    print(f"  {k} -> {idx['weight_map'][k]}")

print("\nLast 15 keys:")
for k in all_keys[-15:]:
    print(f"  {k}")
