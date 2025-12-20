import base64
from typing import List, Dict, Any, Tuple

import httpx
import os


DEFAULT_API_URL = "http://123.129.219.111:3000/v1"
DEFAULT_API_KEY = f"{os.getenv('OPENAI_API_KEY')}"


def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    读取本地图片并编码为 base64，同时返回 mime 类型（image/png 或 image/jpeg）
    """
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {ext}")
    return b64, mime


def build_image_message_content(prompt: str, image_path: str) -> List[Dict[str, Any]]:
    """
    构建 OpenAI 风格的 messages[].content 列表: [text, image_url(data:...)]
    """
    b64, mime = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        },
    ]


def is_vlm_model(model_name: str) -> bool:
    """
    根据模型名称的约定，粗略判断是否支持 vision。
    可按本地实际模型命名规则调整。
    """
    name = model_name.lower()
    return ("vision" in name) or ("vl" in name) or ("gpt-4o" in name) or ("gemini" in name)


def call_chat_completions(
    api_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """
    同步调用 OpenAI 兼容 /chat/completions，返回 message.content（可能为 str 或 list）的统一字符串形式。
    """
    url = api_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    # 兼容 content 可能是 list[{"type": "...", "text": "..."}]
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content
