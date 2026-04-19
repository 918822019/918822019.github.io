# Qwen2.5-Omni模型介绍

## 模型背景

Qwen2.5主体还是采用的Qwen2.5这一LLM模型。

不过比起Qwen2.5-vl来说，并没有改变Qwen2.5-vl的视觉编码器

![img.png](img.png)

从上面的模型结构图来看，也是可以能看出来，模型是拆为thinker和talker两部分，我们这里先不讨论talker部分，因为我们可以把talker部分看成一个TTS模块的拼接。

那么我们现在主要看thinker部分：

1. vision部分使用Qwen2.5-VL的视觉编码器（基于ViT）
2. 听觉编码器部分使用的Qwen2-Audio(这里插个题外话，Qwen3-TTS也是在Qwen2-Audio上改进训练出来的)
3. 文本部分还是用的Qwen2.5本体

## 训练

### 1. 文本处理

文本上其实还是使用Qwen2.5本身的tokenizer分词器，没有做什么改动

### 2.音频处理

音频这里Qwen2.5是把所有音频重采样为16kHz的频率

特征通过转换得到梅尔谱图，参数设定为25ms，步长10ms

时间对齐上每一个音频帧对应原始音频的40ms片段，这个40ms可能有的人看不懂。这个是因为步长的问题导致模型能感受到上一个窗口的最后15ms。25+15得到的40ms

```python
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def visualize_40ms_alignment():
    # 1. 生成一段模拟的“说话”音频 (1秒钟的正弦波，模拟元音)
    duration = 1.0  # 1秒
    sr = 16000      # 采样率 16kHz (符合你提到的预处理参数)
    t = np.linspace(0, duration, int(sr * duration))
    # 生成一个简单的波形：440Hz 的声音
    audio_waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 2. 定义关键参数
    window_size_ms = 40  # 模型的时间对齐粒度：40ms
    hop_size_ms = 40     # 步长也是 40ms (意味着没有重叠，紧密排列)
  
    # 将毫秒转换为样本数
    samples_per_frame = int(sr * (window_size_ms / 1000.0))
  
    # 3. 开始绘图
    plt.figure(figsize=(15, 6))
  
    # --- 绘制音频波形 ---
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_waveform, sr=sr, alpha=0.6, label="Audio Waveform")
    plt.title(f"Audio Waveform & 40ms Alignment (Model 'Hearing' Chunks)", fontsize=14)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    # --- 绘制“40ms”色块 ---
    # 我们在波形图上叠加矩形色块，展示模型是如何切分音频的
    current_time = 0
    frame_idx = 0
  
    while current_time < duration:
        # 计算当前帧的起止时间
        start_t = current_time
        end_t = current_time + (window_size_ms / 1000.0)
      
        # 在图上画一个半透明的红色矩形，代表这 40ms 被压缩成一个特征向量
        plt.axvspan(start_t, end_t, color='red', alpha=0.15)
      
        # 在中间写个编号
        plt.text(start_t + (window_size_ms/2000.0), 0.4, f"Frame {frame_idx}", 
                 rotation=90, va='center', ha='center', fontsize=8, color='darkred')
      
        current_time += (hop_size_ms / 1000.0)
        frame_idx += 1

    # --- 模拟视频帧对齐 (假设视频是 25fps) ---
    # 25fps 意味着每帧 40ms (1000ms / 25 = 40ms)
    # 这正是为什么选 40ms 的原因：它完美对应 25fps 的视频帧！
    video_fps = 25
    video_frame_interval = 1.0 / video_fps
  
    plt.subplot(2, 1, 2)
    plt.title(f"Video Frame Alignment (假设视频为 {video_fps} FPS)", fontsize=14)
    plt.xlim(0, duration)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Time (seconds)")

    # 画出视频帧的边界
    for i in range(int(duration * video_fps)):
        t_pos = i * video_frame_interval
        # 画竖线表示视频帧的分割
        plt.axvline(t_pos, color='blue', linestyle='--', alpha=0.5)
        plt.text(t_pos + video_frame_interval/2, 0.5, f"V-Frame {i}", 
                 rotation=90, va='center', ha='center', color='blue', fontsize=9)

    plt.tight_layout()
    plt.show()
  
    print(f"💡 关键点解释：")
    print(f"1. 红色色块 (40ms)：模型每 40ms 听一次声音，生成一个特征。")
    print(f"2. 蓝色虚线 (40ms)：如果是 25帧/秒 的视频，每帧也是 40ms。")
    print(f"3. 完美对齐：1个音频特征 = 1个视频画面，时间上严丝合缝。")

# 运行函数
visualize_40ms_alignment()
```

## 3.视觉处理

这里直接继承的Qwen2.5-VL的视觉编码器，输入的数据是通过混合训练策略来进行的。

这个混合是指数据的混合，因为视觉需要处理image和video，如果进行单一类型的数据训练很容易造成模态偏移。

所以这里是通过image和video数据混合去进行视觉模态的训练。

这里注意一下，为了工程上的方便实现，通过复制把图片扩展为视频去统一训练的。


## 4.TMROPE位置编码（核心创新）
