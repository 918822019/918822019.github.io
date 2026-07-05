# MTP Router Consistency Analysis

对 BailingMoeV2 模型中 MTP（Multi-Token Prediction）模块的路由预测与 Decoder 实际路由之间的一致性进行多维度定量分析。

## 项目结构

```
core/
  __init__.py           # 包入口，导出 Config, load_model_and_tokenizer
  pipeline.py           # 核心管线：加载模型 → 提取路由 → 对比指标 → 生成报告
docs/
  SYSTEM.md             # 完整系统文档（PCIe 计算、混合方案）
  FINDINGS.md           # 简洁发现汇总
notebooks/              # 5 个 .ipynb 分析笔记本（按主题合并）
  swap_analysis.ipynb   # 路由置换 + 分层交换 + 坍缩机制
  predict_analysis.ipynb# MTP 预测 + 跨 prompt + 置信度
  spec_decode.ipynb     # Spec decoding (draft + 跨 prompt + 模拟)
  full_test.ipynb       # 端到端综合测试
  analyze_routing.ipynb # Token→expert 映射分析
output/                 # 分析报告输出
  report.md             # 多维度分析报告
  results.json          # 完整分析数据
run_analysis.py         # 命令行入口
requirements.txt        # torch, transformers
README.md
```

## 用法

```bash
pip install -r requirements.txt

# 运行核心分析
py run_analysis.py

# 或直接运行 notebook（需 jupyter）
jupyter notebook notebooks/swap_analysis.ipynb
```

## 关键发现

| 维度 | 结果 |
|---|---|
| MTP路由 vs Decoder路由 Cos | ~0.97 |
| Expert 选择冗余度 | 不同 8-expert 子集 → MoE 输出 Cos ~0.92 |
| MTP hidden → gate 预测 (zero-shot) | 所有 18 层平均 Cos=0.81 |
| 深层 (L11-L18) 交换安全率 | 92.9% token 匹配 |
| 全部交换 → collapse | lm_head 映射到 token `!` |
| Spec decoding 加速 (code) | ~4.0x |
| PCIe 3.0 预取加速 | ~7.9x |
| 最终混合方案 | 浅层 L1-L10 token 查找 + 深层 L11-L18 MTP 预测 |

详见 `docs/SYSTEM.md`。
