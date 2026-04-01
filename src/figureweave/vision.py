from __future__ import annotations

import base64
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .config import (
    BOXLIB_NO_ICON_MODE_KEY,
    FLOWCHART_STYLE_PROMPT,
    GEMINI_DEFAULT_IMAGE_SIZE,
    LOCAL_DETECTOR_MAX_BOX_AREA_RATIO,
    LOCAL_DETECTOR_MIN_SCORE,
    LOCAL_OPEN_VOCAB_DETECTOR_MODEL,
    PlaceholderMode,
    ProviderType,
    SAM3_API_TIMEOUT,
    SAM3_FAL_API_URL,
    SAM3_ROBOFLOW_API_URL,
    SVG_MAX_PLACEHOLDERS,
    SVG_MIN_BOX_AREA_RATIO,
)
from .llm import call_llm_image_generation

USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH: Optional[str] = None
_LOCAL_DETECTOR_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

def generate_figure_from_method(
    method_text: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    figure_caption: Optional[str] = None,
    use_reference_image: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> str:
    """
    使用 LLM 生成学术风格图片

    Args:
        method_text: Paper method 文本内容
        output_path: 输出图片路径
        api_key: API Key
        model: 生图模型名称
        base_url: API base URL
        provider: API 提供商
        use_reference_image: 是否使用参考图片（None 则使用全局设置）
        reference_image_path: 参考图片路径（None 则使用全局设置）

    Returns:
        生成的图片路径
    """
    print("=" * 60)
    print("步骤一：使用 LLM 生成学术风格图片")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    if provider == "gemini":
        print(f"分辨率: {image_size}")

    if use_reference_image is None:
        use_reference_image = USE_REFERENCE_IMAGE
    if reference_image_path is None:
        reference_image_path = REFERENCE_IMAGE_PATH
    if reference_image_path:
        use_reference_image = True

    reference_image = None
    if use_reference_image:
        if not reference_image_path:
            raise ValueError("启用参考图模式但未提供 reference_image_path")
        reference_image = Image.open(reference_image_path)
        print(f"参考图片: {reference_image_path}")

    if use_reference_image:
        prompt = f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

{FLOWCHART_STYLE_PROMPT}

Below is the method section of the paper:
\"\"\"
{method_text}
\"\"\""""
        if figure_caption:
            prompt += f"""

The target figure should also satisfy this figure caption / figure brief:
{figure_caption}"""
    else:
        prompt = f"""Generate a professional academic paper figure to visualize the method below.

{FLOWCHART_STYLE_PROMPT}

Additional preferences:
- Show only the essential modules and arrows.
- Favor module-level structure over small decorative subcomponents.
- Keep labels concise and readable.
- If the method sounds complicated, summarize it into three primary stages in the figure.

Below is the method section of this paper:

{method_text}
"""
        if figure_caption:
            prompt += f"""

The target figure caption / figure brief is:
{figure_caption}

Use the caption to decide the most important modules, labels, and flow directions to show.
Stay with a sparse flowchart layout and avoid extra icons unless they are necessary to convey the pipeline."""

    print(f"发送请求到: {base_url}")

    img = call_llm_image_generation(
        prompt=prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        reference_image=reference_image,
        image_size=image_size,
    )

    if img is None:
        raise Exception('API 响应中没有找到图片')

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 PNG 保存（Gemini 返回的图片对象 save() 可能不接受 format 参数）
    try:
        img.save(str(output_path), format='PNG')
    except TypeError:
        img.save(str(output_path))
        # 某些 SDK 对象会按自身默认编码写盘（如 JPEG），这里强制转存为真实 PNG
        with Image.open(str(output_path)) as normalized:
            normalized.save(str(output_path), format='PNG')
    print(f"图片已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤二：SAM3 分割 + Box合并 + 灰色填充+黑色边框+序号标记
# ============================================================================

def get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont:
    """
    根据 box 尺寸动态计算合适的字体大小

    Args:
        box_width: 矩形宽度
        box_height: 矩形高度

    Returns:
        PIL ImageFont 对象
    """
    # 字体大小为 box 短边的 1/4，最小 12，最大 48
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))

    # 尝试加载字体
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    # 回退到默认字体
    try:
        return ImageFont.load_default()
    except:
        return None


# ============================================================================
# Box 合并辅助函数
# ============================================================================

def calculate_overlap_ratio(box1: dict, box2: dict) -> float:
    """
    计算两个box的重叠比例

    Args:
        box1: 第一个box，包含 x1, y1, x2, y2
        box2: 第二个box，包含 x1, y1, x2, y2

    Returns:
        重叠比例 = 交集面积 / 较小box面积
    """
    # 计算交集区域
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    # 无交集
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算各自面积
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    if area1 == 0 or area2 == 0:
        return 0.0

    # 返回交集占较小box的比例
    return intersection / min(area1, area2)


def merge_two_boxes(box1: dict, box2: dict) -> dict:
    """
    合并两个box为最小包围矩形

    Args:
        box1: 第一个box
        box2: 第二个box

    Returns:
        合并后的box（最小包围矩形）
    """
    merged = {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "score": max(box1.get("score", 0), box2.get("score", 0)),  # 保留较高置信度
    }
    # 合并 prompt 字段（如果存在）
    prompt1 = box1.get("prompt", "")
    prompt2 = box2.get("prompt", "")
    if prompt1 and prompt2:
        if prompt1 == prompt2:
            merged["prompt"] = prompt1
        else:
            # 合并不同的 prompts，保留置信度更高的那个
            if box1.get("score", 0) >= box2.get("score", 0):
                merged["prompt"] = prompt1
            else:
                merged["prompt"] = prompt2
    elif prompt1:
        merged["prompt"] = prompt1
    elif prompt2:
        merged["prompt"] = prompt2
    return merged


def merge_overlapping_boxes(boxes: list, overlap_threshold: float = 0.9) -> list:
    """
    迭代合并重叠的boxes

    Args:
        boxes: box列表，每个box包含 x1, y1, x2, y2, score
        overlap_threshold: 重叠阈值，超过此值则合并（默认0.9）

    Returns:
        合并后的box列表，重新编号
    """
    if overlap_threshold <= 0 or len(boxes) <= 1:
        return boxes

    # 复制列表避免修改原数据
    working_boxes = [box.copy() for box in boxes]

    merged = True
    iteration = 0
    while merged:
        merged = False
        iteration += 1
        n = len(working_boxes)

        for i in range(n):
            if merged:
                break
            for j in range(i + 1, n):
                ratio = calculate_overlap_ratio(working_boxes[i], working_boxes[j])
                if ratio >= overlap_threshold:
                    # 合并 box_i 和 box_j
                    new_box = merge_two_boxes(working_boxes[i], working_boxes[j])
                    # 移除原有两个box，添加合并后的box
                    working_boxes = [
                        working_boxes[k] for k in range(n) if k != i and k != j
                    ]
                    working_boxes.append(new_box)
                    merged = True
                    print(f"    迭代 {iteration}: 合并 box {i} 和 box {j} (重叠比例: {ratio:.2f})")
                    break

    # 重新编号
    result = []
    for idx, box in enumerate(working_boxes):
        result_box = {
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "score": box.get("score", 0),
        }
        # 保留 prompt 字段（如果存在）
        if "prompt" in box:
            result_box["prompt"] = box["prompt"]
        result.append(result_box)

    return result


def _filter_boxes_for_svg_reconstruction(
    boxes: list[dict[str, Any]],
    image_size: tuple[int, int],
    min_area_ratio: float = SVG_MIN_BOX_AREA_RATIO,
    max_boxes: int = SVG_MAX_PLACEHOLDERS,
) -> list[dict[str, Any]]:
    """Keep only the most useful placeholders for SVG reconstruction.

    The SVG stage becomes unstable when SAM returns dozens of tiny fragments.
    For paper-style flowcharts, we prefer a small set of larger structural boxes.
    """
    if not boxes:
        return boxes

    width, height = image_size
    image_area = max(1, width * height)

    ranked_boxes: list[dict[str, Any]] = []
    for box in boxes:
        box_width = max(1, int(box["x2"]) - int(box["x1"]))
        box_height = max(1, int(box["y2"]) - int(box["y1"]))
        area = box_width * box_height
        area_ratio = area / image_area
        if area_ratio < min_area_ratio:
            continue
        ranked_boxes.append(
            {
                **box,
                "_area": area,
                "_area_ratio": area_ratio,
            }
        )

    if not ranked_boxes:
        # If the threshold was too aggressive, keep the single largest box as fallback.
        fallback = max(
            boxes,
            key=lambda box: max(1, int(box["x2"]) - int(box["x1"]))
            * max(1, int(box["y2"]) - int(box["y1"])),
        )
        ranked_boxes = [{**fallback, "_area": 1, "_area_ratio": 0.0}]

    ranked_boxes.sort(
        key=lambda box: (
            box["_area"],
            float(box.get("score", 0.0)),
        ),
        reverse=True,
    )
    ranked_boxes = ranked_boxes[:max_boxes]
    ranked_boxes.sort(key=lambda box: (int(box["y1"]) // 40, int(box["x1"])))

    result = []
    for idx, box in enumerate(ranked_boxes):
        filtered_box = {
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": int(box["x1"]),
            "y1": int(box["y1"]),
            "x2": int(box["x2"]),
            "y2": int(box["y2"]),
            "score": float(box.get("score", 0.0)),
        }
        if "prompt" in box:
            filtered_box["prompt"] = box["prompt"]
        result.append(filtered_box)
    return result


def _get_fal_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("SAM3 fal.ai API key missing: set --sam_api_key or FAL_KEY environment variable")
    return key


def _get_roboflow_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "SAM3 Roboflow API key missing: set --sam_api_key or ROBOFLOW_API_KEY/API_KEY environment variable"
        )
    return key


def _get_local_open_vocab_detector(device: str) -> tuple[Any, Any]:
    cache_key = (LOCAL_OPEN_VOCAB_DETECTOR_MODEL, device)
    cached = _LOCAL_DETECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"加载本地检测模型: {LOCAL_OPEN_VOCAB_DETECTOR_MODEL} (device={device})")
    processor = AutoProcessor.from_pretrained(LOCAL_OPEN_VOCAB_DETECTOR_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        LOCAL_OPEN_VOCAB_DETECTOR_MODEL
    )
    model = model.to(device)
    model.eval()
    _LOCAL_DETECTOR_CACHE[cache_key] = (processor, model)
    return processor, model


def _run_grounding_dino_local(
    image: Image.Image,
    prompt_list: list[str],
    min_score: float,
    device: str,
) -> list[dict[str, Any]]:
    processor, model = _get_local_open_vocab_detector(device)
    width, height = image.size
    threshold = max(min_score, LOCAL_DETECTOR_MIN_SCORE)
    image_area = width * height
    all_detected_boxes: list[dict[str, Any]] = []

    for prompt in prompt_list:
        print(f"\n  正在用 Grounding DINO 检测 '{prompt}'")
        inputs = processor(images=image, text=[[prompt]], return_tensors="pt")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=0.15,
            target_sizes=[(height, width)],
        )[0]

        prompt_count = 0
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            score_val = float(score)
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image_area)
            if area_ratio > LOCAL_DETECTOR_MAX_BOX_AREA_RATIO:
                print(
                    f"    跳过过大框: ({x1}, {y1}, {x2}, {y2}), "
                    f"score={score_val:.3f}, area_ratio={area_ratio:.3f}"
                )
                continue
            all_detected_boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": score_val,
                    "prompt": prompt,
                }
            )
            prompt_count += 1
            print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")

        print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")

    if device == "cuda":
        torch.cuda.empty_cache()
    return all_detected_boxes


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cxcywh_norm_to_xyxy(box: list | tuple, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _polygon_to_bbox(points: list, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    xs: list[float] = []
    ys: list[float] = []

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_sam3_api_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    metadata = response_json.get("metadata") if isinstance(response_json, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = item.get("score")
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )
        return detections

    boxes = response_json.get("boxes") if isinstance(response_json, dict) else None
    scores = response_json.get("scores") if isinstance(response_json, dict) else None
    if isinstance(boxes, list) and boxes:
        scores_list = scores if isinstance(scores, list) else []
        for idx, box in enumerate(boxes):
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = scores_list[idx] if idx < len(scores_list) else None
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )

    return detections


def _extract_roboflow_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections

    for prompt_result in prompt_results:
        if not isinstance(prompt_result, dict):
            continue
        predictions = prompt_result.get("predictions", [])
        if not isinstance(predictions, list):
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            confidence = prediction.get("confidence")
            masks = prediction.get("masks", [])
            if not isinstance(masks, list):
                continue
            for mask in masks:
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(
                        mask[0][0], (int, float)
                    ):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(
                                sub[0], (int, float)
                            ):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(
                                sub[0], (list, tuple)
                            ):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if not xyxy:
                    continue
                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "score": confidence,
                    }
                )

    return detections


def _call_sam3_api(
    image_data_uri: str,
    prompt: str,
    api_key: str,
    max_masks: int,
) -> dict:
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    response = requests.post(SAM3_FAL_API_URL, headers=headers, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 API 错误: {result.get('error')}")
    return result


def _call_sam3_roboflow_api(
    image_base64: str,
    prompt: str,
    api_key: str,
    min_score: float,
) -> dict:
    def _redact_secret(text: str) -> str:
        if not api_key:
            return text
        return text.replace(api_key, "***")

    payload = {
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    def _is_dns_error(exc: Exception) -> bool:
        msg = str(exc)
        patterns = [
            "NameResolutionError",
            "Temporary failure in name resolution",
            "getaddrinfo failed",
            "nodename nor servname provided",
            "gaierror",
        ]
        return any(p in msg for p in patterns)

    fallback_urls_env = os.environ.get("ROBOFLOW_API_FALLBACK_URLS", "")
    fallback_urls = [u.strip() for u in fallback_urls_env.split(",") if u.strip()]
    endpoint_urls = [SAM3_ROBOFLOW_API_URL] + [u for u in fallback_urls if u != SAM3_ROBOFLOW_API_URL]

    retry_count_env = os.environ.get("SAM3_API_RETRIES", "3")
    retry_delay_env = os.environ.get("SAM3_API_RETRY_DELAY", "1.5")
    try:
        retry_count = max(1, int(retry_count_env))
    except ValueError:
        retry_count = 3
    try:
        retry_delay = max(0.0, float(retry_delay_env))
    except ValueError:
        retry_delay = 1.5

    last_error: Optional[Exception] = None

    for endpoint in endpoint_urls:
        url = f"{endpoint}?api_key={api_key}"
        for attempt in range(1, retry_count + 1):
            try:
                response = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
                if response.status_code != 200:
                    raise Exception(
                        f"SAM3 Roboflow API 错误: {response.status_code} - {response.text[:500]}"
                    )
                result = response.json()
                if isinstance(result, dict) and "error" in result:
                    raise Exception(f"SAM3 Roboflow API 错误: {result.get('error')}")
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                # DNS/网络偶发问题时做指数退避重试
                if attempt < retry_count:
                    sleep_s = retry_delay * (2 ** (attempt - 1))
                    safe_error = _redact_secret(str(e))
                    print(
                        f"    Roboflow 请求失败（尝试 {attempt}/{retry_count}）：{safe_error}，"
                        f"{sleep_s:.1f}s 后重试..."
                    )
                    time.sleep(sleep_s)
                    continue
                # 当前 endpoint 的重试次数用尽，切到下一个 endpoint
                break
            except Exception as e:
                last_error = e
                break

    if last_error is not None and _is_dns_error(last_error):
        raise RuntimeError(
            "SAM3 Roboflow 域名解析失败（容器内 DNS 无法解析 serverless.roboflow.com）。\n"
            "可用修复：\n"
            "1) 在 docker-compose.yml 设置 dns（如 223.5.5.5 / 119.29.29.29）；\n"
            "2) 在 .env 里设置 ROBOFLOW_API_URL 或 ROBOFLOW_API_FALLBACK_URLS；\n"
            "3) 临时改用 --sam_backend fal（需 FAL_KEY）。"
        ) from last_error

    if last_error is not None:
        raise RuntimeError(f"SAM3 Roboflow 请求失败：{_redact_secret(str(last_error))}") from last_error

    raise RuntimeError("SAM3 Roboflow 请求失败：未知错误")


def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.5,
    merge_threshold: float = 0.9,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
) -> tuple[str, str, list]:
    """
    使用 SAM3 分割图片，用灰色填充+黑色边框+序号标记，生成 boxlib.json

    占位符样式：
    - 灰色填充 (#808080)
    - 黑色边框 (width=3)
    - 白色居中序号标签 (<AF>01, <AF>02, ...)

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        text_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: 最低置信度阈值
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）

    Returns:
        (samed_path, boxlib_path, valid_boxes)
    """
    print("\n" + "=" * 60)
    print("步骤二：SAM3 分割 + 灰色填充+黑色边框+序号标记")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    original_size = image.size
    print(f"原图尺寸: {original_size[0]} x {original_size[1]}")

    # 解析多个 prompts（支持逗号分隔）
    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"使用的 prompts: {prompt_list}")

    # 对每个 prompt 分别检测并收集结果
    all_detected_boxes = []
    total_detected = 0

    backend = sam_backend
    if backend == "api":
        backend = "fal"

    if backend == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import sam3

            sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, '__path__') else Path(sam3.__file__).parent
            bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                bpe_path = None
                print("警告: 未找到 bpe 文件，使用默认路径")

            model = build_sam3_image_model(
                device=device,
                bpe_path=str(bpe_path) if bpe_path else None,
            )
            processor = Sam3Processor(model, device=device)
            autocast_context = nullcontext()
            if device == "cuda":
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

            with autocast_context:
                inference_state = processor.set_image(image)

            for prompt in prompt_list:
                print(f"\n  正在检测: '{prompt}'")
                with autocast_context:
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

                boxes = output["boxes"]
                scores = output["scores"]

                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.detach().to(dtype=torch.float32).cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().to(dtype=torch.float32).cpu().numpy()

                prompt_count = 0
                for box, score in zip(boxes, scores):
                    if score >= min_score:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        all_detected_boxes.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "score": float(score),
                            "prompt": prompt
                        })
                        prompt_count += 1
                        print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score:.3f}")
                    else:
                        print(f"    跳过: score={score:.3f} < {min_score}")

                print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
                total_detected += prompt_count

            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"本地 SAM3 不可用，回退到本地检测模型: {exc}")
            all_detected_boxes = _run_grounding_dino_local(
                image=image,
                prompt_list=prompt_list,
                min_score=min_score,
                device=device,
            )
            total_detected = len(all_detected_boxes)

    elif backend == "fal":
        api_key = _get_fal_api_key(sam_api_key)
        max_masks = max(1, min(32, int(sam_max_masks)))
        image_data_uri = _image_to_data_uri(image)
        print(f"SAM3 fal.ai API 模式: max_masks={max_masks}")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_api(
                image_data_uri=image_data_uri,
                prompt=prompt,
                api_key=api_key,
                max_masks=max_masks,
            )
            detections = _extract_sam3_api_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt  # 记录来源 prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    elif backend == "roboflow":
        api_key = _get_roboflow_api_key(sam_api_key)
        image_base64 = _image_to_base64(image)
        print("SAM3 Roboflow API 模式: format=polygon")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_roboflow_api(
                image_base64=image_base64,
                prompt=prompt,
                api_key=api_key,
                min_score=min_score,
            )
            detections = _extract_roboflow_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    else:
        raise ValueError(f"未知 SAM3 后端: {sam_backend}")

    print(f"\n总计检测: {total_detected} 个对象 (来自 {len(prompt_list)} 个 prompts)")

    # 为所有检测到的 boxes 分配临时 id 和 label（用于合并）
    valid_boxes = []
    for i, box_data in enumerate(all_detected_boxes):
        valid_boxes.append({
            "id": i,
            "label": f"<AF>{i + 1:02d}",
            "x1": box_data["x1"],
            "y1": box_data["y1"],
            "x2": box_data["x2"],
            "y2": box_data["y2"],
            "score": box_data["score"],
            "prompt": box_data["prompt"]
        })

    # === 新增：合并重叠的boxes ===
    if merge_threshold > 0 and len(valid_boxes) > 1:
        print(f"\n  合并重叠的boxes (阈值: {merge_threshold})...")
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(valid_boxes, merge_threshold)
        merged_count = original_count - len(valid_boxes)
        if merged_count > 0:
            print(f"  合并完成: {original_count} -> {len(valid_boxes)} (合并了 {merged_count} 个)")
            # 打印合并后的box信息
            print(f"\n  合并后的boxes:")
            for box_info in valid_boxes:
                print(f"    {box_info['label']}: ({box_info['x1']}, {box_info['y1']}, {box_info['x2']}, {box_info['y2']})")
        else:
            print(f"  无需合并，所有boxes重叠比例均低于阈值")

    if len(valid_boxes) > 1:
        prefilter_count = len(valid_boxes)
        filtered_boxes = _filter_boxes_for_svg_reconstruction(
            valid_boxes,
            image_size=original_size,
        )
        removed_count = prefilter_count - len(filtered_boxes)
        if removed_count > 0:
            print(
                f"\n  为了让 SVG 重建更稳定，过滤过小/过碎的占位框: "
                f"{prefilter_count} -> {len(filtered_boxes)}"
            )
            valid_boxes = filtered_boxes
            for box_info in valid_boxes:
                print(
                    f"    保留 {box_info['label']}: "
                    f"({box_info['x1']}, {box_info['y1']}, {box_info['x2']}, {box_info['y2']})"
                )

    # 使用合并后的 valid_boxes 创建标记图片
    print(f"\n  绘制 samed.png (使用 {len(valid_boxes)} 个boxes)...")
    samed_image = image.copy()
    draw = ImageDraw.Draw(samed_image)

    for box_info in valid_boxes:
        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
        label = box_info["label"]

        # 灰色填充 + 黑色边框
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)

        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 获取合适大小的字体
        box_width = x2 - x1
        box_height = y2 - y1
        font = get_label_font(box_width, box_height)

        # 绘制白色居中序号标签
        if font:
            # 使用 anchor="mm" 居中绘制（如果支持）
            try:
                draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
            except TypeError:
                # 旧版本 PIL 不支持 anchor，手动计算位置
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            # 无字体时使用默认
            draw.text((cx, cy), label, fill="white")

    samed_path = output_dir / "samed.png"
    samed_image.save(str(samed_path))
    print(f"标记图片已保存: {samed_path}")

    boxlib_data = {
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "prompts_used": prompt_list,
        "boxes": valid_boxes,
        BOXLIB_NO_ICON_MODE_KEY: len(valid_boxes) == 0,
    }

    boxlib_path = output_dir / "boxlib.json"
    with open(boxlib_path, 'w', encoding='utf-8') as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)
    print(f"Box 信息已保存: {boxlib_path}")

    return str(samed_path), str(boxlib_path), valid_boxes


# ============================================================================
# 步骤三：裁切 + RMBG2 去背景
# ============================================================================

def _get_hf_token() -> Optional[str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not isinstance(token, str):
        return None
    token = token.strip()
    return token or None


def _has_rmbg2_cached_weights() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    snapshots_dir = hf_home / "hub" / "models--briaai--RMBG-2.0" / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snapshots_dir.glob("*/config.json"))


def _ensure_rmbg2_access_ready(rmbg_model_path: Optional[str]) -> None:
    if rmbg_model_path and Path(rmbg_model_path).exists():
        return
    if _get_hf_token() is not None:
        return
    if _has_rmbg2_cached_weights():
        return
    raise RuntimeError(
        "步骤三需要使用 briaai/RMBG-2.0，但当前未检测到可用访问凭据。\n"
        "请先完成：\n"
        "1) 申请访问 https://huggingface.co/briaai/RMBG-2.0\n"
        "2) 在 .env 设置 HF_TOKEN=你的Read权限token\n"
        "3) 重新运行 docker compose up -d --build"
    )


class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量背景抠图"""

    def __init__(self, model_path: Path | str | None = None, output_dir: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("./output/icons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_repo_id = "briaai/RMBG-2.0"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        hf_token = _get_hf_token()

        if self.model_path and self.model_path.exists():
            print(f"加载本地 RMBG 权重: {self.model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(self.model_path), trust_remote_code=True,
            ).eval().to(device)
        else:
            print("从 HuggingFace 加载 RMBG-2.0 模型...")
            if hf_token:
                print("检测到 HF_TOKEN，使用鉴权访问 gated 模型。")
            else:
                print("未检测到 HF_TOKEN，尝试匿名访问（gated 模型通常会失败）。")

            try:
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    self.model_repo_id,
                    trust_remote_code=True,
                    token=hf_token,
                ).eval().to(device)
            except Exception as e:
                msg = str(e).lower()
                is_gated = (
                    "gated repo" in msg
                    or "cannot access gated repo" in msg
                    or "access to model briaai/rmbg-2.0 is restricted" in msg
                    or "401 client error" in msg
                    or "you are trying to access a gated repo" in msg
                )
                if is_gated:
                    raise RuntimeError(
                        "无法下载 RMBG-2.0（HuggingFace gated 模型鉴权失败）。\n"
                        "请按以下步骤配置：\n"
                        "1) 登录并申请模型访问权限: https://huggingface.co/briaai/RMBG-2.0\n"
                        "2) 创建具有 Read 权限的 token\n"
                        "3) 在项目 .env 设置 HF_TOKEN=你的token\n"
                        "4) 重新执行: docker compose up -d --build"
                    ) from e
                raise

        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def remove_background(self, image: Image.Image, output_name: str) -> str:
        image_rgb = image.convert("RGB")
        input_tensor = self.transform_image(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_rgb.size)

        out = image_rgb.copy()
        out.putalpha(mask)

        out_path = self.output_dir / f"{output_name}_nobg.png"
        out.save(out_path)
        return str(out_path)


def crop_and_remove_background(
    image_path: str,
    boxlib_path: str,
    output_dir: str,
    rmbg_model_path: Optional[str] = None,
) -> list[dict]:
    """
    根据 boxlib.json 裁切图片并使用 RMBG2 去背景

    文件命名使用 label: icon_AF01.png, icon_AF01_nobg.png
    """
    print("\n" + "=" * 60)
    print("步骤三：裁切 + RMBG2 去背景")
    print("=" * 60)

    output_dir = Path(output_dir)
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib_data = json.load(f)

    boxes = boxlib_data["boxes"]

    if len(boxes) == 0:
        print("警告: 没有检测到有效的 box")
        return []

    remover = BriaRMBG2Remover(model_path=rmbg_model_path, output_dir=icons_dir)

    icon_infos = []
    for box_info in boxes:
        box_id = box_info["id"]
        label = box_info.get("label", f"<AF>{box_id + 1:02d}")
        # 将 <AF>01 转换为 AF01 用于文件名
        label_clean = label.replace("<", "").replace(">", "")

        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

        cropped = image.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path)

        nobg_path = remover.remove_background(cropped, f"icon_{label_clean}")

        icon_infos.append({
            "id": box_id,
            "label": label,
            "label_clean": label_clean,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "crop_path": str(crop_path),
            "nobg_path": nobg_path,
        })

        print(f"  {label}: 裁切并去背景完成 -> {nobg_path}")

    del remover
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return icon_infos


# ============================================================================
# 步骤四：多模态调用生成 SVG
# ============================================================================

