import torch, torch.nn as nn, torch.nn.functional as F, math


class RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        t = torch.arange(offset, offset + x.shape[-2], device=x.device).type_as(self.inv_freq)
        freqs = t.unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        return torch.cat([freqs, freqs], -1)


def apply_rope(x, emb):
    s = x.shape[-1] // 2
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], s, 2).contiguous())
    ec = torch.view_as_complex(emb.reshape(*emb.shape[:-1], s, 2).contiguous())
    return torch.view_as_real(xc * ec).reshape(*x.shape[:-1], -1)


def linear_attn(q, k, v, decay=0.99, eps=1e-6):
    B, H, T, D = q.shape
    qf = F.relu(q).pow(2) + 0.1
    kf = F.relu(k).pow(2) + 0.1
    vf = F.relu(v).pow(2) + 0.1
    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    z = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    outs = []
    for i in range(T):
        S = decay * S + kf[:, :, i : i + 1].transpose(-1, -2) @ vf[:, :, i : i + 1]
        z = decay * z + kf[:, :, i]
        out = (qf[:, :, i : i + 1] @ S) / (qf[:, :, i : i + 1] @ z.unsqueeze(-1)).clamp(min=eps)
        outs.append(out)
    return torch.cat(outs, dim=-2)


class CSA(nn.Module):
    def __init__(self, dim, hd):
        super().__init__()
        self.q = nn.Linear(dim, hd, bias=False)
        self.k = nn.Linear(dim, hd, bias=False)
        self.v = nn.Linear(dim, hd, bias=False)
        self.o = nn.Linear(hd, dim, bias=False)
        self.decay = 0.99

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 1, -1).transpose(1, 2)
        k = self.k(x).view(B, T, 1, -1).transpose(1, 2)
        v = self.v(x).view(B, T, 1, -1).transpose(1, 2)
        out = linear_attn(q, k, v, self.decay).squeeze(1)
        return self.o(out)


class HCA(nn.Module):
    def __init__(self, dim, hd):
        super().__init__()
        self.q = nn.Linear(dim, hd, bias=False)
        self.k = nn.Linear(dim, hd, bias=False)
        self.v = nn.Linear(dim, hd, bias=False)
        self.o = nn.Linear(hd, dim, bias=False)
        self.decay = 0.999

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 1, -1).transpose(1, 2)
        k = self.k(x).view(B, T, 1, -1).transpose(1, 2)
        v = self.v(x).view(B, T, 1, -1).transpose(1, 2)
        out = linear_attn(q, k, v, self.decay).squeeze(1)
        return self.o(out)


class SWA(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(head_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        emb = self.rope(k)
        q = apply_rope(q, emb)
        k = apply_rope(k, emb)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.csa = CSA(dim, head_dim)
        self.hca = HCA(dim, head_dim)
        self.swa = SWA(dim, num_heads, head_dim)
        self.gate = nn.Parameter(torch.ones(3))
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        oc = self.csa(h)
        oh = self.hca(h)
        os = self.swa(h)
        g = F.softmax(self.gate, dim=0)
        x = x + self.drop(g[0] * oc + g[1] * oh + g[2] * os)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class EdgeTransformer(nn.Module):
    def __init__(self, vocab_size=4096, dim=512, num_layers=4, num_heads=8, head_dim=64,
                 ffn_mult=4, dropout=0.0, max_seq_len=512, weight_tie=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.layers = nn.ModuleList([
            HybridBlock(dim, num_heads, head_dim, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if weight_tie:
            self.head.weight = self.tok.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.tok(x) + self.pos[:, : x.size(1)]
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
