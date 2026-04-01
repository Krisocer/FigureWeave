from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .config import (
    GEMINI_DEFAULT_IMAGE_SIZE,
    GEMINI_IMAGE_MAX_RETRIES,
    GEMINI_IMAGE_RETRY_BASE_DELAY,
    PROVIDER_CONFIGS,
    ProviderType,
)

def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的文本 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        reference_image: 参考图片（可选）
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    if provider == "bianxie":
        return _call_bianxie_text(prompt, api_key, model, base_url, max_tokens, temperature)
    if provider == "gemini":
        return _call_gemini_text(prompt, api_key, model, max_tokens, temperature)
    if provider == "openai":
        return _call_openai_text(prompt, api_key, model, max_tokens, temperature)
    if provider == "anthropic":
        return _call_anthropic_text(prompt, api_key, model, base_url, max_tokens, temperature)
    return _call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的多模态 LLM 调用接口

    Args:
        contents: 内容列表（字符串或 PIL Image）
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    if provider == "bianxie":
        return _call_bianxie_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    if provider == "gemini":
        return _call_gemini_multimodal(contents, api_key, model, max_tokens, temperature)
    if provider == "openai":
        return _call_openai_multimodal(contents, api_key, model, max_tokens, temperature)
    if provider == "anthropic":
        return _call_anthropic_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    return _call_openrouter_multimodal(contents, api_key, model, base_url, max_tokens, temperature)


def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    reference_image: Optional[Image.Image] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """
    统一的图像生成 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商

    Returns:
        生成的 PIL Image，失败返回 None
    """
    if provider == "bianxie":
        return _call_bianxie_image_generation(prompt, api_key, model, base_url, reference_image)
    if provider == "gemini":
        return _call_gemini_image_generation(
            prompt=prompt,
            api_key=api_key,
            model=model,
            reference_image=reference_image,
            image_size=image_size,
        )
    if provider == "openai":
        return _call_openai_image_generation(prompt, api_key, model, reference_image)
    if provider == "anthropic":
        raise ValueError("Anthropic Claude does not support image generation in this pipeline. Choose Gemini or OpenAI for the image stage.")
    return _call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)


def _build_openai_chat_content(contents: List[Any]) -> List[Dict[str, Any]]:
    message_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            message_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
    return message_content


# ============================================================================
# Bianxie Provider 实现 (使用 OpenAI SDK)
# ============================================================================

def _call_bianxie_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用 Bianxie 文本接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] API 调用失败: {e}")
        raise


def _call_bianxie_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用 Bianxie 多模态接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        message_content = _build_openai_chat_content(contents)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] 多模态 API 调用失败: {e}")
        raise


def _call_bianxie_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 OpenAI SDK 调用 Bianxie 图像生成接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        if reference_image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            buf = io.BytesIO()
            reference_image.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
            messages = [{"role": "user", "content": message_content}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = completion.choices[0].message.content if completion and completion.choices else None

        if not content:
            return None

        # Bianxie 返回 Markdown 格式的图片: ![text](data:image/png;base64,...)
        pattern = r'data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)'
        match = re.search(pattern, content)

        if match:
            image_base64 = match.group(2)
            image_data = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_data))

        return None
    except Exception as e:
        print(f"[Bianxie] 图像生成 API 调用失败: {e}")
        raise


# ============================================================================
def _call_openai_text(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Use the official OpenAI Chat Completions API for text generation."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[OpenAI] Text API call failed: {e}")
        raise


def _call_openai_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Use the official OpenAI Chat Completions API for image understanding + text output."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_openai_chat_content(contents)}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[OpenAI] Multimodal API call failed: {e}")
        raise


def _call_openai_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """
    Use the official OpenAI Responses API with the image_generation tool.
    This supports prompt-only generation and prompt + reference image generation.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response_content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]

        if reference_image is not None:
            buf = io.BytesIO()
            reference_image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            response_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                }
            )

        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": response_content}],
            tools=[{"type": "image_generation"}],
        )

        for output in getattr(response, "output", []) or []:
            if getattr(output, "type", None) != "image_generation_call":
                continue
            image_b64 = getattr(output, "result", None)
            if not image_b64:
                continue
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            image.load()
            return image

        return None
    except Exception as e:
        print(f"[OpenAI] Image generation API call failed: {e}")
        raise


def _build_anthropic_content(contents: List[Any]) -> List[Dict[str, Any]]:
    anthropic_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            anthropic_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            anthropic_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                }
            )
    return anthropic_content


def _extract_anthropic_text(result: dict) -> Optional[str]:
    parts = []
    for item in result.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    joined = "".join(parts).strip()
    return joined or None


def _call_anthropic_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    return _call_anthropic_multimodal(
        contents=[prompt],
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _call_anthropic_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call the official Anthropic Messages API with text and image inputs."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": _build_anthropic_content(contents),
            }
        ],
    }

    response = requests.post(base_url, headers=headers, json=payload, timeout=300)
    if response.status_code != 200:
        raise Exception(f"Anthropic API error: {response.status_code} - {response.text[:500]}")

    result = response.json()
    text = _extract_anthropic_text(result)
    if not text:
        raise Exception("Anthropic API returned no text content")
    return text


# OpenRouter Provider 实现 (使用 requests)
# ============================================================================

def _get_openrouter_headers(api_key: str) -> dict:
    """获取 OpenRouter 请求头"""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://localhost',
        'X-Title': 'MethodToSVG'
    }


def _get_openrouter_api_url(base_url: str) -> str:
    """获取 OpenRouter API URL"""
    if not base_url.endswith('/chat/completions'):
        if base_url.endswith('/'):
            return base_url + 'chat/completions'
        else:
            return base_url + '/chat/completions'
    return base_url


def _extract_openrouter_message_text(message: Any) -> Optional[str]:
    """尽可能从 OpenRouter message 中提取文本，兼容 string/list/object 多种 content 形态"""
    if not isinstance(message, dict):
        return None

    def _collect_from_part(part: Any, out: list[str]) -> None:
        if isinstance(part, str):
            text = part.strip()
            if text:
                out.append(text)
            return

        if not isinstance(part, dict):
            return

        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())

        nested = part.get("content")
        if isinstance(nested, list):
            for item in nested:
                _collect_from_part(item, out)

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, dict):
        chunks: list[str] = []
        _collect_from_part(content, chunks)
        if chunks:
            return "\n".join(chunks)

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            _collect_from_part(part, chunks)
        if chunks:
            return "\n".join(chunks)

    for key in ("output_text", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _summarize_openrouter_choice(choice: Any) -> str:
    """构造可读的 OpenRouter choice 摘要，便于定位空响应问题"""
    if not isinstance(choice, dict):
        return f"invalid choice type={type(choice).__name__}"

    message = choice.get("message")
    if not isinstance(message, dict):
        return (
            f"finish_reason={choice.get('finish_reason')}, "
            f"message_type={type(message).__name__}"
        )

    content = message.get("content")
    content_type = type(content).__name__
    if isinstance(content, str):
        content_size = len(content)
    elif isinstance(content, list):
        content_size = len(content)
    elif isinstance(content, dict):
        content_size = len(content.keys())
    else:
        content_size = 0

    refusal = message.get("refusal")
    refusal_preview = repr(refusal)
    if len(refusal_preview) > 220:
        refusal_preview = refusal_preview[:220] + "..."

    return (
        f"finish_reason={choice.get('finish_reason')}, "
        f"message_keys={sorted(message.keys())}, "
        f"content_type={content_type}, "
        f"content_size={content_size}, "
        f"refusal={refusal_preview}"
    )


def _call_openrouter_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 文本接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    message = choices[0].get('message', {})
    text = _extract_openrouter_message_text(message)
    if text:
        return text
    return None


def _call_openrouter_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 多模态接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    message_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            message_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': message_content}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    retry_env = os.environ.get("OPENROUTER_MULTIMODAL_RETRIES", "3")
    delay_env = os.environ.get("OPENROUTER_MULTIMODAL_RETRY_DELAY", "1.5")
    try:
        retry_count = max(1, int(retry_env))
    except ValueError:
        retry_count = 3
    try:
        retry_delay = max(0.0, float(delay_env))
    except ValueError:
        retry_delay = 1.5

    last_error: Optional[Exception] = None
    for attempt in range(1, retry_count + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=300)

            if response.status_code != 200:
                raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

            result = response.json()

            if 'error' in result:
                error_msg = result.get('error', {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get('message', str(error_msg))
                raise Exception(f'OpenRouter API 错误: {error_msg}')

            choices = result.get('choices', [])
            if not choices:
                raise RuntimeError("OpenRouter 返回 choices 为空")

            message = choices[0].get('message', {})
            text = _extract_openrouter_message_text(message)
            if text:
                return text

            choice_summary = _summarize_openrouter_choice(choices[0])
            raise RuntimeError(
                "OpenRouter 多模态响应没有可解析文本内容。"
                f" model={model}, summary={choice_summary}"
            )
        except Exception as e:
            last_error = e
            if attempt < retry_count:
                sleep_s = retry_delay * (2 ** (attempt - 1))
                print(
                    f"OpenRouter 多模态请求失败（尝试 {attempt}/{retry_count}）：{e}，"
                    f"{sleep_s:.1f}s 后重试..."
                )
                time.sleep(sleep_s)
                continue
            break

    if last_error is not None:
        raise last_error
    return None


def _call_openrouter_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 requests 调用 OpenRouter 图像生成接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    if reference_image is None:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        buf = io.BytesIO()
        reference_image.save(buf, format='PNG')
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        messages = [{'role': 'user', 'content': message_content}]

    payload = {
        'model': model,
        'messages': messages,
        # 对 OpenRouter 的 Gemini 图像模型，强制 image-only 可显著降低“返回纯文本无图片”的概率
        'modalities': ['image'],
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    def _extract_data_url_payload(data_url: str) -> Optional[str]:
        match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return re.sub(r"\s+", "", match.group(1))

    def _decode_base64_image(image_b64: str) -> Optional[Image.Image]:
        if not image_b64:
            return None
        try:
            b64 = re.sub(r"\s+", "", image_b64)
            padding = len(b64) % 4
            if padding:
                b64 += "=" * (4 - padding)
            image_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(image_data))
            image.load()
            return image
        except Exception:
            return None

    def _load_remote_image(image_url: str) -> Optional[Image.Image]:
        try:
            resp = requests.get(image_url, timeout=120)
            if resp.status_code != 200 or not resp.content:
                return None
            image = Image.open(io.BytesIO(resp.content))
            image.load()
            return image
        except Exception:
            return None

    def _extract_image_url(value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("url"), str):
                return value["url"]
            if "image_url" in value:
                return _extract_image_url(value.get("image_url"))
        return None

    def _try_parse_image_candidate(candidate: Any) -> Optional[Image.Image]:
        if isinstance(candidate, dict):
            # OpenAI/OpenRouter 常见图片字段
            for key in ("b64_json", "base64", "data"):
                raw = candidate.get(key)
                if isinstance(raw, str):
                    parsed = _decode_base64_image(raw)
                    if parsed is not None:
                        return parsed
            if "image_url" in candidate:
                parsed = _try_parse_image_candidate(candidate.get("image_url"))
                if parsed is not None:
                    return parsed
            if "url" in candidate:
                parsed = _try_parse_image_candidate(candidate.get("url"))
                if parsed is not None:
                    return parsed
            return None

        if not isinstance(candidate, str):
            return None

        candidate = candidate.strip()
        if not candidate:
            return None

        if candidate.startswith("data:image/"):
            b64_payload = _extract_data_url_payload(candidate)
            if b64_payload:
                return _decode_base64_image(b64_payload)
            return None

        if candidate.startswith("http://") or candidate.startswith("https://"):
            return _load_remote_image(candidate)

        # 极少数场景服务会直接返回纯 base64
        return _decode_base64_image(candidate)

    def _extract_markdown_image_urls(text: str) -> list[str]:
        urls: list[str] = []
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            urls.append(match.group(1).strip())
        for match in re.finditer(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", text, flags=re.IGNORECASE):
            urls.append(match.group(0).strip())
        return urls

    choices = result.get('choices', [])
    if not choices:
        raise RuntimeError("OpenRouter 返回中没有 choices，无法解析生图结果。")

    message = choices[0].get('message', {})
    candidates: list[Any] = []

    images = message.get("images")
    if isinstance(images, list):
        candidates.extend(images)
    elif images is not None:
        candidates.append(images)

    content = message.get("content")
    if isinstance(content, list):
        candidates.extend(content)
    elif isinstance(content, str):
        candidates.extend(_extract_markdown_image_urls(content))

    # 某些中间层会把图片放到顶层字段
    top_images = result.get("images")
    if isinstance(top_images, list):
        candidates.extend(top_images)

    for item in candidates:
        # 先尝试直接解析对象
        parsed = _try_parse_image_candidate(item)
        if parsed is not None:
            return parsed

        # 再尝试从对象中抽取 URL 字符串
        image_url = _extract_image_url(item)
        if image_url:
            parsed = _try_parse_image_candidate(image_url)
            if parsed is not None:
                return parsed

    content_preview = ""
    if isinstance(content, str):
        content_preview = content[:240].replace("\n", " ")

    refusal = message.get("refusal")
    message_keys = sorted(message.keys()) if isinstance(message, dict) else []
    images_count = len(images) if isinstance(images, list) else 0

    raise RuntimeError(
        "OpenRouter 响应成功但未包含可解析图片。"
        f" model={model}, message_keys={message_keys}, images_count={images_count}, "
        f"content_type={type(content).__name__}, refusal={refusal!r}, "
        f"content_preview={content_preview!r}"
    )


# ============================================================================
# Gemini Provider 实现 (Google 官方 SDK)
# ============================================================================

def _get_gemini_client(api_key: str):
    """获取 Gemini 客户端（延迟导入，避免非 Gemini 场景强依赖）"""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "未安装 google-genai，请执行: pip install google-genai"
        ) from e
    return genai.Client(api_key=api_key)


def _build_gemini_text_config(max_tokens: int, temperature: float):
    """构建 Gemini 文本生成配置"""
    from google.genai import types

    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def _extract_gemini_text(response: Any) -> Optional[str]:
    """从 Gemini 响应中提取文本"""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts = getattr(response, "parts", None) or []
    extracted: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str) and part_text.strip():
            extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    return None


def _extract_gemini_image(response: Any) -> Optional[Image.Image]:
    """从 Gemini 响应中提取图片（优先使用 part.as_image()）"""
    parts = getattr(response, "parts", None) or []
    for part in parts:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            if image is not None:
                return image

        inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline_data is None:
            continue
        data = getattr(inline_data, "data", None)
        if isinstance(data, bytes) and data:
            return Image.open(io.BytesIO(data))
        if isinstance(data, str) and data:
            return Image.open(io.BytesIO(base64.b64decode(data)))

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                image = as_image()
                if image is not None:
                    return image
    return None


def _call_gemini_text(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Gemini 文本接口"""
    try:
        client = _get_gemini_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] 文本 API 调用失败: {e}")
        raise


def _call_gemini_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Gemini 多模态接口"""
    try:
        client = _get_gemini_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] 多模态 API 调用失败: {e}")
        raise


def _call_gemini_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    reference_image: Optional[Image.Image] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """调用 Gemini 生图接口，默认 image_size=2K"""
    from google.genai import errors as genai_errors
    from google.genai import types

    client = _get_gemini_client(api_key)
    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(image_size=image_size),
    )

    if reference_image is None:
        contents: list[Any] = [prompt]
    else:
        # 参考图放在前面，提示语在后，遵循 Gemini 多模态输入习惯
        contents = [reference_image, prompt]

    last_error: Exception | None = None
    for attempt in range(1, GEMINI_IMAGE_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return _extract_gemini_image(response)
        except genai_errors.ServerError as e:
            last_error = e
            if getattr(e, "status_code", None) != 503 or attempt >= GEMINI_IMAGE_MAX_RETRIES:
                print(f"[Gemini] 图像生成 API 调用失败: {e}")
                raise

            delay = GEMINI_IMAGE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[Gemini] 生图服务暂时繁忙，{attempt}/{GEMINI_IMAGE_MAX_RETRIES} 次重试，"
                f"{delay:.0f} 秒后再试..."
            )
            time.sleep(delay)
        except Exception as e:
            print(f"[Gemini] 图像生成 API 调用失败: {e}")
            raise

    if last_error is not None:
        raise last_error
    return None


# ============================================================================
# 步骤一：调用 LLM 生成图片
# ============================================================================

