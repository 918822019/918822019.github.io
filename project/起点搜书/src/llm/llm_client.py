"""简易的 OpenAI 聊天客户端（项目内使用）。

用法：
    - 在环境变量中设置 OPENAI_API_KEY
    - 调用 `chat(messages)` 或直接以脚本运行进行快速测试

本模块使用与 OpenAI Chat Completions 兼容的请求格式。
"""

import os
from typing import List, Dict

try:
    import openai
except Exception:
    openai = None


def chat(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", **kwargs) -> str:
    """发送聊天消息并返回助手的文本回复。

    参数：
      - messages: 聊天消息列表，每项为 {"role": "system|user|assistant", "content": str}
      - model: 要使用的模型名称，默认为 "gpt-4o-mini"
      - kwargs: 额外传给 `openai.ChatCompletion.create` 的参数（如 `max_tokens`、`temperature` 等）

    返回：助手生成的文本内容（字符串）。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # 若未设置密钥，提示用户设置环境变量
        raise RuntimeError("请设置环境变量 OPENAI_API_KEY")

    if openai is None:
        # 如果未安装 openai 包，提示安装
        raise RuntimeError("未安装 openai 包；请运行: pip install openai")

    # 配置 API Key
    openai.api_key = api_key

    # 调用 ChatCompletion 接口
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    # 解析响应，兼容常见的 choices 结构
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"响应中未找到 choices: {resp}")

    # 返回第一个候选的 message.content
    return choices[0]["message"]["content"]


def main():
    example = [
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好，请介绍一下你自己"},
    ]
    try:
        text = chat(
            example,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=512,
            temperature=0.7,
        )
        print(text)
    except Exception as e:
        # 调用失败时打印错误信息，便于本地调试
        print("调用 OpenAI 出错:", e)


if __name__ == "__main__":
    main()
