"""
EdgeTransformer 文本生成推理脚本
"""
import argparse
import torch
import torch.nn.functional as F
from config import RunConfig
from model import EdgeTransformer
from data import BPETokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=64, temp=0.8, top_k=20, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = ids[:]

    for _ in range(max_len):
        logits = model(input_ids)[:, -1, :]
        logits = logits / temp
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1, None]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        if next_id == 3:  # EOS
            break

    return tokenizer.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/edge_transformer/last.pt")
    p.add_argument("--prompt", default="a cat sitting on")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_len", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load("output/bpe_tokenizer.pkl")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    model = EdgeTransformer(vocab_size=tok.vocab_size, dim=512, num_layers=4,
                            num_heads=8, head_dim=64, max_seq_len=512)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded step {ckpt.get('step', '?')}, loss {ckpt.get('loss', '?'):.4f}")

    prompts = [
        "a cat sitting on",
        "a group of people",
        "a close up of",
        "the view from",
        "a plate of",
    ]
    for prompt in prompts:
        out = generate(model, tok, prompt, args.max_len, args.temp, args.top_k, device)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {out}")


if __name__ == "__main__":
    main()
