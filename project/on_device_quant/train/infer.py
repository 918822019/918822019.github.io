"""
EdgeTransformer 文本/多模态生成推理脚本
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from config import RunConfig
from model import EdgeTransformer
from data import BPETokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=64, temp=0.8, top_k=20, device="cuda"):
    """
    自回归文本生成

    采样策略：temperature 缩放 + top-k 截断
      - temp < 1：更确定（趋向 argmax），temp > 1：更随机
      - top_k=20：只从概率最高的 20 个 token 中采样，过滤长尾噪声

    Args:
        model: 训练好的 EdgeTransformer
        tokenizer: BPE 分词器
        prompt: 文本提示
        max_len: 最大生成 token 数
        temp: 温度参数，控制采样随机性
        top_k: top-k 截断，0 表示不截断
        device: 推理设备
    Returns:
        生成的完整文本（prompt + 生成部分）
    """
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = ids[:]

    for _ in range(max_len):
        # 只取最后一个位置的 logits（因果模型，前面的 token 不需要重新预测）
        logits = model(input_ids)[:, -1, :]
        logits = logits / temp
        if top_k > 0:
            # 保留 top_k 个最大 logits，其余设为 -inf（softmax 后概率为 0）
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1, None]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        # 多项分布采样：按概率随机选一个 token
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        # 每步重新拼接完整序列（简单实现，未做 KV cache 优化）
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        if next_id == 3:  # EOS token，生成结束
            break

    return tokenizer.decode(generated)


@torch.no_grad()
def generate_multimodal(model, tokenizer, image, prompt, image_processor,
                        max_len=64, temp=0.8, top_k=20, device="cuda"):
    """
    图文条件文本生成（LLaVA 式）

    图像经 SigLIP 编码后作为前缀 token，文本 prompt 跟在后面，
    自回归生成描述文本。

    Args:
        model: 带视觉编码器的 EdgeTransformer
        tokenizer: BPE 分词器
        image: PIL Image
        prompt: 文本提示（如 "a photo of"）
        image_processor: SiglipImageProcessor
        max_len: 最大生成 token 数
        temp: 温度参数
        top_k: top-k 截断
        device: 推理设备
    """
    model.eval()
    # 编码图像
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)

    # 编码 prompt
    ids = [2] + tokenizer.encode(prompt)  # BOS + prompt
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = ids[:]

    for _ in range(max_len):
        logits = model(input_ids, images=pixel_values)
        # logits[:, -1, :] 是最后一个 text token 的预测（image tokens 在前缀）
        logits = logits[:, -1, :] / temp
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
    """加载 checkpoint，做文本或图文生成推理"""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/edge_transformer/last.pt")
    p.add_argument("--prompt", default="a cat sitting on")
    p.add_argument("--image", default="", help="图片路径（多模态推理用）")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_len", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load("output/bpe_tokenizer.pkl")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # 从 checkpoint 同目录读取 config.json，还原模型结构参数
    cfg_path = Path(args.ckpt).parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        m = cfg["model"]
    else:
        # 旧 checkpoint 没有 config.json，用默认参数回退
        m = {"dim": 1536, "num_layers": 20, "num_heads": 24, "head_dim": 64}

    # norm_type 从 config 读取，旧 checkpoint 默认用 rms
    norm_type = m.get("norm_type", "rms")
    vision_model = m.get("vision_model", "")
    model = EdgeTransformer(vocab_size=tok.vocab_size, dim=m["dim"], num_layers=m["num_layers"],
                            num_heads=m["num_heads"], head_dim=m["head_dim"], max_seq_len=512,
                            norm_type=norm_type, vision_model=vision_model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded step {ckpt.get('step', '?')}, loss {ckpt.get('loss', '?'):.4f}")

    if args.image and vision_model:
        # 多模态推理
        from data import get_image_processor
        from PIL import Image
        image_processor = get_image_processor(vision_model)
        image = Image.open(args.image).convert("RGB")
        out = generate_multimodal(model, tok, image, args.prompt, image_processor,
                                  args.max_len, args.temp, args.top_k, device)
        print(f"\nImage: {args.image}")
        print(f"Prompt: {args.prompt}")
        print(f"Output: {out}")
    else:
        # 纯文本推理
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
