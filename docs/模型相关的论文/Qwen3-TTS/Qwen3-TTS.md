# Qwen3-TTS技术报告

## 1. 模型概述

![](image.png)

Qwen3-TTS是阿里云Qwen团队开发的一系列强大的文本转语音模型，具备以下核心能力：

- **可控性**：支持通过自然语言指令控制语音风格、情感和语速
- **语音克隆与预设语音模版**：支持3秒快速声音克隆和49种高质量音色
- **自然度**：生成高度自然、类人的语音，支持自适应调整语速和韵律
- **多语言支持**：支持10种主要语言（中文、英文、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语）和9种方言
- **流式传输**：基于Dual-Track混合流式生成架构，端到端合成延迟低至97ms

模型为了实时输出，做了单码本->多码本的转换，将单码本的模型转换成多码本的模型，方便多语言支持。
以及模型在token解码上做了MTP（Multi-Token Prediction）的优化，同时把多个码本进行还原。

## 2. 模型结构

模型结构如下：
![](image1.png)

### 2.1 核心组件

首先Qwen-TTS-Tokenzier-12Hz，是用的Qwen2-Audio。
这里Qwen2-Audio是一个ASR模型，这里Qwen团队进行了两阶段训练。
第一阶段是继续做ASR的训练，第二阶段是基于卷积的梅尔谱图解码器对整个模型进行微调。

在第一阶段我们能看到模型插入了一个重采样层以及一个VQ层，这两个层的作用我们后面再介绍。

### 2.2 关键技术特性

1. **强大的语音表示能力**：基于自研的Qwen3-TTS-Tokenizer-12Hz，实现高效的声学压缩和高维语义建模
2. **通用端到端架构**：采用离散多码本语言模型架构，实现全信息端到端语音建模
3. **极低延迟流式生成**：基于Dual-Track混合流式生成架构，单模型同时支持流式和非流式生成

## 3. 详细介绍其中的训练方案

### 1. 为什么要重采样

方便把原始数据变成我们想要的频率特征数据，不过这一部分主要是应用在25HZ特征下的数据，阿里主要开源了12HZ的模型，所以我们这里不做过多讨论。

### 2. VQ层

VQ层是一个很经典的技术，是用于把连续的变量映射为离散的向量

这里我们重点介绍一下：

VQ层可以看作在一个连续的向量空间中，通过学习把连续向量空间划分为有限个区域。

1. 我们假设有一个码本，其中包含k个可以学习的嵌入向量
   dim为向量D的维度

给定输入编码器输出的连续向量，vq层执行映射，然后转换为离散表示

2. VQ数学公式

前向传播:k = \underset{j}{\operatorname{argmin}} \, \| z - e_j \|_2^2

### 3. 多码本架构

Qwen3-TTS采用离散多码本语言模型架构，相比传统的单码本架构有以下优势：

- **信息保留更完整**：多码本可以同时编码声学特征、语义信息和副语言信息
- **生成质量更高**：避免了信息瓶颈和级联错误
- **支持更丰富的语音表现力**：可以同时控制音色、情感、语速等多个维度

### 4. 流式生成机制

基于Dual-Track混合流式生成架构：

- **双轨并行处理**：一个轨道处理文本输入，另一个轨道生成语音输出
- **实时响应**：输入单个字符后即可输出首个音频包
- **低延迟**：端到端合成延迟低至97ms，满足实时交互场景需求

## 4. 模型变体

Qwen3-TTS提供多个模型变体，满足不同使用场景：

| 模型名称 | 参数量 | 主要功能 | 支持语言 |
|---------|--------|---------|---------|
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 基础模型，支持3秒快速声音克隆 | 10种语言 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | 支持9种优质音色和风格控制 | 10种语言 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1.7B | 基于自然语言描述的语音设计 | 10种语言 |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 轻量级基础模型 | 10种语言 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | 轻量级音色控制模型 | 10种语言 |

## 5. 使用示例

### 5.1 安装依赖

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
```

### 5.2 基础语音克隆

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 3秒快速声音克隆
wavs, sr = model.generate(
    text="你好，欢迎使用Qwen3-TTS语音合成系统。",
    language="Chinese",
    speaker="your_audio_reference.wav"  # 3秒参考音频
)
sf.write("output.wav", wavs[0], sr)
```

### 5.3 自定义音色生成

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 使用预设音色
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说"
)
sf.write("output_custom_voice.wav", wavs[0], sr)
```

### 5.4 语音设计

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 基于描述生成语音
wavs, sr = model.generate_voice_design(
    text="今天天气真好，我们去公园散步吧。",
    instruct="一个温柔的年轻女性声音，语气轻松愉快"
)
sf.write("output_voice_design.wav", wavs[0], sr)
```

## 6. 性能评测

### 6.1 多语言语音合成评测

| 语言 | Qwen3-TTS-12Hz | MiniMax | ElevenLabs | GPT-4o-Audio |
|------|----------------|---------|------------|--------------|
| 中文 | 2.156 | 3.241 | 4.123 | 3.567 |
| 英文 | 3.875 | 4.518 | 6.877 | 4.924 |
| 日语 | 3.875 | 4.518 | 6.877 | 4.924 |
| 韩语 | 2.202 | 2.274 | 3.053 | 2.763 |

*性能指标为WER（词错误率，越低越好）*

### 6.2 长语音生成评测

| 数据集 | Qwen3-TTS-25Hz | Qwen3-TTS-12Hz | Higgs-Audio-v2 |
|--------|----------------|----------------|----------------|
| long-zh | 1.517 | 2.356 | 5.505 |
| long-en | 1.225 | 2.812 | 6.917 |

## 7. 技术优势

1. **端到端架构**：避免了传统LM+DiT方案中的信息瓶颈和级联错误
2. **多码本设计**：完整保留副语言信息和声学环境特征
3. **低延迟流式**：满足实时交互场景的严格要求
4. **多语言支持**：覆盖全球主要语言和方言
5. **可控性强**：支持自然语言指令控制语音风格和情感

## 8. 应用场景

- **智能客服**：提供自然、人性化的语音交互体验
- **有声内容制作**：自动生成高质量的有声读物、播客等
- **语音助手**：支持实时流式语音交互
- **多语言翻译**：生成自然的多语言语音输出
- **个性化语音**：支持声音克隆和自定义语音设计

## 9. 相关资源

- [GitHub仓库](https://github.com/QwenLM/Qwen3-TTS)
- [HuggingFace模型](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [ModelScope模型](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [在线演示](https://huggingface.co/spaces/Qwen/Qwen3-TTS-Demo)

## 10. 引用

```bibtex
@misc{qwen3_tts_202601,
  author = {Qwen Team, Alibaba},
  title = {Qwen3-TTS Family is Now Open Sourced: Voice Design, Clone, and Generation!},
  year = {2026},
  url = {https://qwen.ai/blog?id=qwen3tts-0115},
  urldate = {2026-01-22}
}
```

---
*更新时间: 2026-06-23*
