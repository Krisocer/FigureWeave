from __future__ import annotations

import io
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import FigureMode, PlaceholderMode, ProviderType
from .llm import call_llm_multimodal, call_llm_text


SVG_EDITABLE_TAGS = {
    "rect",
    "text",
    "tspan",
    "path",
    "line",
    "polyline",
    "polygon",
    "image",
    "circle",
    "ellipse",
    "g",
}


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _make_tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}" if ns else local


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_translate(transform: Optional[str]) -> tuple[float, float]:
    if not transform:
        return 0.0, 0.0
    match = re.search(r"translate\(\s*([-+]?\d*\.?\d+)(?:[\s,]+([-+]?\d*\.?\d+))?\s*\)", transform)
    if not match:
        return 0.0, 0.0
    tx = _parse_float(match.group(1)) or 0.0
    ty = _parse_float(match.group(2)) or 0.0
    return tx, ty


def _union_bbox(boxes: list[tuple[float, float, float, float]]) -> Optional[tuple[float, float, float, float]]:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def _apply_translate(
    bbox: Optional[tuple[float, float, float, float]],
    tx: float,
    ty: float,
) -> Optional[tuple[float, float, float, float]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return x1 + tx, y1 + ty, x2 + tx, y2 + ty


def _element_bbox(elem: ET.Element) -> Optional[tuple[float, float, float, float]]:
    tag = _local_name(elem.tag)
    tx, ty = _extract_translate(elem.get("transform"))

    if tag in {"rect", "image"}:
        x = _parse_float(elem.get("x"))
        y = _parse_float(elem.get("y"))
        w = _parse_float(elem.get("width"))
        h = _parse_float(elem.get("height"))
        if None in {x, y, w, h}:
            return None
        return _apply_translate((x, y, x + w, y + h), tx, ty)

    if tag == "line":
        x1 = _parse_float(elem.get("x1"))
        y1 = _parse_float(elem.get("y1"))
        x2 = _parse_float(elem.get("x2"))
        y2 = _parse_float(elem.get("y2"))
        if None in {x1, y1, x2, y2}:
            return None
        return _apply_translate((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), tx, ty)

    if tag in {"circle", "ellipse"}:
        cx = _parse_float(elem.get("cx"))
        cy = _parse_float(elem.get("cy"))
        if tag == "circle":
            r = _parse_float(elem.get("r"))
            if None in {cx, cy, r}:
                return None
            return _apply_translate((cx - r, cy - r, cx + r, cy + r), tx, ty)
        rx = _parse_float(elem.get("rx"))
        ry = _parse_float(elem.get("ry"))
        if None in {cx, cy, rx, ry}:
            return None
        return _apply_translate((cx - rx, cy - ry, cx + rx, cy + ry), tx, ty)

    if tag == "text":
        x = _parse_float(elem.get("x"))
        y = _parse_float(elem.get("y"))
        if None in {x, y}:
            return None
        font_size = _parse_float(elem.get("font-size")) or 32.0
        text_value = "".join(elem.itertext()).strip() or "X"
        width = max(font_size * 0.55 * len(text_value), font_size * 0.8)
        height = font_size * 1.2
        anchor = (elem.get("text-anchor") or "").strip().lower()
        if anchor == "middle":
            x1 = x - width / 2
            x2 = x + width / 2
        elif anchor == "end":
            x1 = x - width
            x2 = x
        else:
            x1 = x
            x2 = x + width
        return _apply_translate((x1, y - height, x2, y + height * 0.15), tx, ty)

    if tag in {"path", "polygon", "polyline"}:
        raw = elem.get("d") if tag == "path" else elem.get("points")
        if not raw:
            return None
        nums = [_parse_float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        nums = [v for v in nums if v is not None]
        if len(nums) < 2:
            return None
        xs = nums[0::2]
        ys = nums[1::2]
        if not xs or not ys:
            return None
        return _apply_translate((min(xs), min(ys), max(xs), max(ys)), tx, ty)

    if tag == "g":
        boxes = []
        for child in list(elem):
            child_box = _element_bbox(child)
            if child_box:
                boxes.append(child_box)
        return _apply_translate(_union_bbox(boxes), tx, ty)

    return None


def _bbox_area(bbox: Optional[tuple[float, float, float, float]]) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: Optional[tuple[float, float, float, float]]) -> Optional[tuple[float, float]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _expand_bbox(bbox: tuple[float, float, float, float], margin: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def _bbox_contains_point(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _get_class_tokens(elem: ET.Element) -> set[str]:
    return {token.strip() for token in (elem.get("class") or "").split() if token.strip()}


def _ensure_semantic_element_ids(root: ET.Element) -> None:
    counters: dict[str, int] = {}
    for elem in root.iter():
        tag = _local_name(elem.tag)
        if tag not in SVG_EDITABLE_TAGS:
            continue
        if elem.get("id"):
            continue
        counters[tag] = counters.get(tag, 0) + 1
        elem.set("id", f"fw-{tag}-{counters[tag]:03d}")


def _ensure_existing_semantic_groups(root: ET.Element) -> None:
    for elem in root.iter():
        if _local_name(elem.tag) != "g":
            continue
        elem_id = (elem.get("id") or "").strip().lower()
        if not elem_id:
            continue
        classes = _get_class_tokens(elem)
        if elem_id.startswith("module_") or elem_id.startswith("module-"):
            classes.update({"fw-module", "fw-editable"})
            elem.set("data-role", elem.get("data-role") or "module")
        elif elem_id.startswith("panel_") or elem_id.startswith("panel-"):
            classes.update({"fw-panel", "fw-editable"})
            elem.set("data-role", elem.get("data-role") or "panel")
        if classes:
            elem.set("class", " ".join(sorted(classes)))


def _ensure_cairo_runtime() -> None:
    gtk_bins = [
        Path(r"C:\Program Files\GTK3-Runtime Win64\bin"),
        Path(r"C:\Program Files (x86)\GTK3-Runtime Win64\bin"),
    ]
    for bin_dir in gtk_bins:
        if not bin_dir.is_dir():
            continue
        current_path = os.environ.get("PATH", "")
        if str(bin_dir) not in current_path:
            os.environ["PATH"] = str(bin_dir) + os.pathsep + current_path
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (FileNotFoundError, OSError):
                pass
        break


def _wrap_members_with_group(
    root: ET.Element,
    members: list[ET.Element],
    *,
    ns: str,
    group_id: str,
    class_name: str,
    role: str,
) -> ET.Element:
    children = list(root)
    insert_index = min(children.index(member) for member in members if member in children)
    group = ET.Element(
        _make_tag(ns, "g"),
        {
            "id": group_id,
            "class": class_name,
            "data-role": role,
        },
    )
    for member in members:
        if member in root:
            root.remove(member)
            group.append(member)
    root.insert(insert_index, group)
    return group


def semanticize_svg_for_editing(svg_code: str) -> str:
    """Wrap low-level SVG primitives into semantic groups for easier editing."""
    try:
        root = ET.fromstring(svg_code.encode("utf-8"))
    except Exception:
        return svg_code

    ns = root.tag.split("}", 1)[0].strip("{") if "}" in root.tag else ""
    _ensure_existing_semantic_groups(root)
    module_candidates: list[tuple[float, ET.Element, tuple[float, float, float, float]]] = []
    panel_candidates: list[tuple[float, ET.Element, tuple[float, float, float, float]]] = []

    for child in list(root):
        if _local_name(child.tag) != "rect":
            continue
        classes = _get_class_tokens(child)
        bbox = _element_bbox(child)
        if not bbox:
            continue
        area = _bbox_area(bbox)
        if "border-dashed" in classes:
            panel_candidates.append((area, child, bbox))
        elif any(cls.startswith("box-") for cls in classes):
            module_candidates.append((area, child, bbox))

    assigned: set[int] = set()
    module_index = 0
    for _, rect, bbox in sorted(module_candidates, key=lambda item: item[0]):
        if id(rect) in assigned or rect not in root:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        margin = max(18.0, min(width, height) * 0.18)
        expanded = _expand_bbox(bbox, margin)
        members: list[ET.Element] = []
        for child in list(root):
            if id(child) in assigned:
                continue
            child_tag = _local_name(child.tag)
            if child is rect:
                members.append(child)
                continue
            if child_tag in {"defs", "style"}:
                continue
            child_box = _element_bbox(child)
            if not child_box:
                continue
            center = _bbox_center(child_box)
            if center and _bbox_contains_point(expanded, center[0], center[1]):
                members.append(child)
            elif _bbox_intersects(expanded, child_box) and _bbox_area(child_box) < _bbox_area(expanded) * 0.7:
                members.append(child)
        if len(members) >= 2:
            module_index += 1
            group = _wrap_members_with_group(
                root,
                members,
                ns=ns,
                group_id=f"fw-module-{module_index:02d}",
                class_name="fw-module fw-editable",
                role="module",
            )
            assigned.update(id(member) for member in list(group))

    panel_index = 0
    for _, rect, bbox in sorted(panel_candidates, key=lambda item: item[0]):
        if rect not in root:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        margin = max(24.0, min(width, height) * 0.05)
        expanded = _expand_bbox(bbox, margin)
        members: list[ET.Element] = []
        for child in list(root):
            if _local_name(child.tag) in {"defs", "style"}:
                continue
            child_box = _element_bbox(child)
            if not child_box:
                continue
            center = _bbox_center(child_box)
            if center and _bbox_contains_point(expanded, center[0], center[1]):
                members.append(child)
        if len(members) >= 2:
            panel_index += 1
            _wrap_members_with_group(
                root,
                members,
                ns=ns,
                group_id=f"fw-panel-{panel_index:02d}",
                class_name="fw-panel fw-editable",
                role="panel",
            )

    root.set("data-figureweave-export", "semantic-svg")
    _ensure_semantic_element_ids(root)
    if ns:
        ET.register_namespace("", ns)
    return ET.tostring(root, encoding="unicode")

def generate_svg_template(
    figure_path: str,
    samed_path: str,
    boxlib_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    figure_mode: FigureMode = "simple_flowchart",
    figure_caption: Optional[str] = None,
    placeholder_mode: PlaceholderMode = "label",
    no_icon_mode: bool = False,
) -> str:
    """
    使用多模态 LLM 生成 SVG 代码

    Args:
        placeholder_mode: 占位符模式
            - "none": 无特殊样式
            - "box": 传入 boxlib 坐标
            - "label": 灰色填充+黑色边框+序号标签（推荐）
    """
    print("\n" + "=" * 60)
    print("步骤四：多模态调用生成 SVG")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"占位符模式: {placeholder_mode}")
    if no_icon_mode:
        print("无图标模式: 启用纯 SVG 复现回退")

    figure_img = Image.open(figure_path)
    samed_img = Image.open(samed_path)

    figure_width, figure_height = figure_img.size
    print(f"原图尺寸: {figure_width} x {figure_height}")
    caption_context = ""
    if figure_caption:
        caption_context = f"""

Figure caption / intent:
{figure_caption}
"""
    complex_context = ""
    if figure_mode == "complex_paper":
        complex_context = """

Complex paper figure guidance:
- Preserve nested panel structure, dashed grouping boxes, legends, and auxiliary branches when present.
- Reproduce formulas, brackets, small charts, heatmaps, and tiny labels as faithfully as possible.
- Prefer exact scientific layout fidelity over simplifying the figure."""

    if no_icon_mode:
        prompt_text = f"""编写 SVG 代码来尽可能像素级复现这张图片。

当前 SAM3 没有检测到任何有效图标，因此这是一个无图标回退模式任务：
- 不要添加任何灰色矩形占位符
- 不要添加任何 <AF>01 / <AF>02 标签
- 不要凭空生成图标框、占位组或额外装饰
- 所有可见内容都应直接用 SVG 元素复现
- 优先保持整体布局、文字、箭头、线条、边框和配色与原图一致

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG

Image reference notes:
- Image 1 is the original target figure.
- Image 2 is the SAM reference image. It does not contain any valid icon placeholder boxes for this run.
{caption_context}
{complex_context}

Editability requirements:
- Return semantically editable SVG, not just visually similar SVG.
- Wrap each logical module or block in a <g> element with a stable id such as module_encoder, module_xtgap, module_prediction, etc.
- Wrap each major panel or region in a <g> element with a stable id such as panel_input, panel_training, panel_evaluation, etc.
- Keep text as <text> elements, boxes as <rect> elements, and connectors as <line>/<path> whenever possible.
- Avoid converting the whole figure into arbitrary paths.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""
    else:
        # 基础 prompt
        base_prompt = f"""编写svg代码来实现像素级别的复现这张图片（除了图标用相同大小的矩形占位符填充之外其他文字和组件(尤其是箭头样式)都要保持一致（即灰色矩形覆盖的内容就是图标））

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG
"""
        if figure_caption:
            base_prompt += f"""

Figure caption / intent:
{figure_caption}
"""
        if complex_context:
            base_prompt += f"\n{complex_context}\n"
        base_prompt += """

EDITABILITY REQUIREMENTS:
- Return semantically editable SVG, not just a visually similar one.
- Wrap each logical module in a <g> element with a stable id, for example module_encoder, module_xtgap, module_fusion, module_prediction.
- Wrap each major panel or region in a <g> element with a stable id, for example panel_input, panel_training, panel_evaluation.
- Keep text as <text> elements, boxes as <rect> elements, and connectors as <line>/<path> whenever possible.
- Do not flatten the whole figure into arbitrary paths unless absolutely unavoidable.
"""

    if not no_icon_mode and placeholder_mode == "box":
        # box 模式：传入 boxlib 坐标
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_content = f.read()

        prompt_text = base_prompt + f"""
ICON COORDINATES FROM boxlib.json:
The following JSON contains precise icon coordinates detected by SAM3:
{boxlib_content}
Use these coordinates to accurately position your icon placeholders in the SVG.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif not no_icon_mode and placeholder_mode == "label":
        # label 模式：要求占位符样式与 samed.png 一致
        prompt_text = base_prompt + """
PLACEHOLDER STYLE REQUIREMENT:
Look at the second image (samed.png) - each icon area is marked with a gray rectangle (#808080), black border, and a centered label like <AF>01, <AF>02, etc.

Your SVG placeholders MUST match this exact style:
- Rectangle with fill="#808080" and stroke="black" stroke-width="2"
- Centered white text showing the same label (<AF>01, <AF>02, etc.)
- Wrap each placeholder in a <g> element with id matching the label (e.g., id="AF01")

Example placeholder structure:
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="14">&lt;AF&gt;01</text>
</g>

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif not no_icon_mode:  # none 模式
        prompt_text = base_prompt + """
Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    contents = [prompt_text, figure_img, samed_img]

    print(f"发送多模态请求到: {base_url}")

    content = call_llm_multimodal(
        contents=contents,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_tokens=50000,
    )

    if not content:
        raise Exception(
            f"API 响应中没有内容（provider={provider}, model={model}）。"
            "如果是 OpenRouter，可尝试增大 OPENROUTER_MULTIMODAL_RETRIES 后重试。"
        )

    svg_code = extract_svg_code(content)

    if not svg_code:
        raise Exception('无法从响应中提取 SVG 代码')

    # 步骤 4.5：SVG 语法验证和修复
    svg_code = check_and_fix_svg(
        svg_code=svg_code,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
    )
    svg_code = semanticize_svg_for_editing(svg_code)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"SVG 模板已保存: {output_path}")
    return str(output_path)


def extract_svg_code(content: str) -> Optional[str]:
    """从响应内容中提取 SVG 代码"""
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)

    pattern = r'```(?:svg|xml)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    if match:
        code = match.group(1).strip()
        if code.startswith('<svg'):
            return code

    if content.strip().startswith('<svg'):
        return content.strip()

    return None


# ============================================================================
# 步骤 4.5：SVG 语法验证和修复
# ============================================================================

def validate_svg_syntax(svg_code: str) -> tuple[bool, list[str]]:
    """使用 lxml 解析验证 SVG 语法"""
    try:
        from lxml import etree
        etree.fromstring(svg_code.encode('utf-8'))
        return True, []
    except ImportError:
        print("  警告: lxml 未安装，使用内置 xml.etree 进行验证")
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(svg_code)
            return True, []
        except ET.ParseError as e:
            return False, [f"XML 解析错误: {str(e)}"]
    except Exception as e:
        from lxml import etree
        if isinstance(e, etree.XMLSyntaxError):
            errors = []
            error_log = e.error_log
            for error in error_log:
                errors.append(f"行 {error.line}, 列 {error.column}: {error.message}")
            if not errors:
                errors.append(f"行 {e.lineno}, 列 {e.offset}: {e.msg}")
            return False, errors
        else:
            return False, [f"解析错误: {str(e)}"]


def fix_svg_with_llm(
    svg_code: str,
    errors: list[str],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_retries: int = 3,
) -> str:
    """使用 LLM 修复 SVG 语法错误"""
    print("\n  " + "-" * 50)
    print("  检测到 SVG 语法错误，调用 LLM 修复...")
    print("  " + "-" * 50)
    for err in errors:
        print(f"    {err}")

    current_svg = svg_code
    current_errors = errors

    for attempt in range(max_retries):
        print(f"\n  修复尝试 {attempt + 1}/{max_retries}...")

        error_list = "\n".join([f"  - {err}" for err in current_errors])
        prompt = f"""The following SVG code has XML syntax errors detected by an XML parser. Please fix ALL the errors and return valid SVG code.

SYNTAX ERRORS DETECTED:
{error_list}

ORIGINAL SVG CODE:
```xml
{current_svg}
```

IMPORTANT INSTRUCTIONS:
1. Fix all XML syntax errors (unclosed tags, invalid attributes, unescaped characters, etc.)
2. Ensure the output is valid XML that can be parsed by lxml
3. Keep all the visual elements and structure intact
4. Return ONLY the fixed SVG code, starting with <svg and ending with </svg>
5. Do NOT include any markdown formatting, explanation, or code blocks - just the raw SVG code"""

        try:
            content = call_llm_text(
                prompt=prompt,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=16000,
                temperature=0.3,
            )

            if not content:
                print("    响应为空")
                continue

            fixed_svg = extract_svg_code(content)

            if not fixed_svg:
                print("    无法从响应中提取 SVG 代码")
                continue

            is_valid, new_errors = validate_svg_syntax(fixed_svg)

            if is_valid:
                print("    修复成功！SVG 语法验证通过")
                return fixed_svg
            else:
                print(f"    修复后仍有 {len(new_errors)} 个错误:")
                for err in new_errors[:3]:
                    print(f"      {err}")
                if len(new_errors) > 3:
                    print(f"      ... 还有 {len(new_errors) - 3} 个错误")
                current_svg = fixed_svg
                current_errors = new_errors

        except Exception as e:
            print(f"    修复过程出错: {e}")
            continue

    print(f"  警告: 达到最大重试次数 ({max_retries})，返回最后一次的 SVG 代码")
    return current_svg


def check_and_fix_svg(
    svg_code: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
) -> str:
    """检查 SVG 语法并在需要时调用 LLM 修复"""
    print("\n" + "-" * 50)
    print("步骤 4.5：SVG 语法验证（使用 lxml XML 解析器）")
    print("-" * 50)

    is_valid, errors = validate_svg_syntax(svg_code)

    if is_valid:
        print("  SVG 语法验证通过！")
        return svg_code
    else:
        print(f"  发现 {len(errors)} 个语法错误")
        fixed_svg = fix_svg_with_llm(
            svg_code=svg_code,
            errors=errors,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
        )
        return fixed_svg


# ============================================================================
# 步骤 4.7：坐标系对齐
# ============================================================================

def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
    """从 SVG 代码中提取坐标系尺寸"""
    viewbox_pattern = r'viewBox=["\']([^"\']+)["\']'
    viewbox_match = re.search(viewbox_pattern, svg_code, re.IGNORECASE)

    if viewbox_match:
        viewbox_value = viewbox_match.group(1).strip()
        parts = viewbox_value.split()
        if len(parts) >= 4:
            try:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
                return vb_width, vb_height
            except ValueError:
                pass

    def parse_dimension(attr_name: str) -> Optional[float]:
        pattern = rf'{attr_name}=["\']([^"\']+)["\']'
        match = re.search(pattern, svg_code, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            numeric_match = re.match(r'([\d.]+)', value)
            if numeric_match:
                try:
                    return float(numeric_match.group(1))
                except ValueError:
                    pass
        return None

    width = parse_dimension('width')
    height = parse_dimension('height')

    if width and height:
        return width, height

    return None, None


def calculate_scale_factors(
    figure_width: int,
    figure_height: int,
    svg_width: float,
    svg_height: float,
) -> tuple[float, float]:
    """计算从 figure.png 像素坐标到 SVG 坐标的缩放因子"""
    scale_x = svg_width / figure_width
    scale_y = svg_height / figure_height
    return scale_x, scale_y


# ============================================================================
# 步骤五：图标替换到 SVG（支持序号匹配）
# ============================================================================

def replace_icons_in_svg(
    template_svg_path: str,
    icon_infos: list[dict],
    output_path: str,
    scale_factors: tuple[float, float] = (1.0, 1.0),
    match_by_label: bool = True,
) -> str:
    """
    将透明背景图标替换到 SVG 中的占位符

    Args:
        template_svg_path: SVG 模板路径
        icon_infos: 图标信息列表
        output_path: 输出路径
        scale_factors: 坐标缩放因子
        match_by_label: 是否使用序号匹配（label 模式）
    """
    print("\n" + "=" * 60)
    print("步骤五：图标替换到 SVG")
    print("=" * 60)
    print(f"匹配模式: {'序号匹配' if match_by_label else '坐标匹配'}")

    scale_x, scale_y = scale_factors
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"应用坐标缩放: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    with open(template_svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    for icon_info in icon_infos:
        label = icon_info.get("label", "")
        label_clean = icon_info.get("label_clean", label.replace("<", "").replace(">", ""))
        nobg_path = icon_info["nobg_path"]

        # 读取图标并转为 base64
        icon_img = Image.open(nobg_path)
        buf = io.BytesIO()
        icon_img.save(buf, format="PNG")
        icon_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        replaced = False

        if match_by_label and label:
            # 方式1：查找 id="AF01" 的 <g> 元素
            g_pattern = rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</g>'
            g_match = re.search(g_pattern, svg_content, re.IGNORECASE)

            if g_match:
                g_content = g_match.group(0)

                # 提取 <g> 元素的 transform="translate(x, y)" （如果存在）
                # 这处理 LLM 生成 <g id="AF01" transform="translate(100, 50)"><rect x="0" y="0" ...> 的情况
                g_tag_match = re.match(r'<g[^>]*>', g_content, re.IGNORECASE)
                translate_x, translate_y = 0.0, 0.0
                if g_tag_match:
                    g_tag = g_tag_match.group(0)
                    # 匹配 transform="translate(100, 50)" 或 transform="translate(100 50)"
                    transform_pattern = r'transform=["\'][^"\']*translate\s*\(\s*([\d.-]+)[\s,]+([\d.-]+)\s*\)'
                    transform_match = re.search(transform_pattern, g_tag, re.IGNORECASE)
                    if transform_match:
                        translate_x = float(transform_match.group(1))
                        translate_y = float(transform_match.group(2))

                # 从 <g> 中提取 <rect> 的尺寸
                rect_patterns = [
                    # x="100" y="50" width="80" height="80"
                    r'<rect[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?',
                    # width="80" height="80" x="100" y="50" (属性顺序不同)
                    r'<rect[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?',
                ]

                rect_info = None
                for rp in rect_patterns:
                    rect_match = re.search(rp, g_content, re.IGNORECASE)
                    if rect_match:
                        groups = rect_match.groups()
                        if len(groups) == 4:
                            if 'width' in rp[:50]:  # 第二种模式
                                width, height, x, y = groups
                            else:
                                x, y, width, height = groups
                            rect_info = {
                                'x': float(x),
                                'y': float(y),
                                'width': float(width),
                                'height': float(height)
                            }
                            break

                if rect_info:
                    # 将 <g> 的 transform translate 值加到 rect 坐标上
                    x = rect_info['x'] + translate_x
                    y = rect_info['y'] + translate_y
                    width, height = rect_info['width'], rect_info['height']

                    # 如果应用了 transform，输出提示
                    if translate_x != 0 or translate_y != 0:
                        print(f"  {label}: 检测到 <g> transform: translate({translate_x}, {translate_y})")

                    # 创建 image 标签替换整个 <g>
                    image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
                    svg_content = svg_content.replace(g_content, image_tag)
                    print(f"  {label}: 替换成功 (序号匹配 <g>) at ({x}, {y}) size {width}x{height}")
                    replaced = True

            # 方式2：查找包含 label 文本的 <text> 元素附近的 <rect>
            if not replaced:
                # 查找包含 <AF>01 或 &lt;AF&gt;01 的文本
                text_patterns = [
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ]

                for tp in text_patterns:
                    text_match = re.search(tp, svg_content, re.IGNORECASE)
                    if text_match:
                        # 找到文本，向前查找最近的 <rect>
                        text_pos = text_match.start()
                        preceding_svg = svg_content[:text_pos]

                        # 查找最后一个 <rect>
                        rect_matches = list(re.finditer(r'<rect[^>]*/?\s*>', preceding_svg, re.IGNORECASE))
                        if rect_matches:
                            last_rect = rect_matches[-1]
                            rect_content = last_rect.group(0)

                            # 提取 rect 的属性
                            x_match = re.search(r'\bx=["\']?([\d.]+)', rect_content)
                            y_match = re.search(r'\by=["\']?([\d.]+)', rect_content)
                            w_match = re.search(r'\bwidth=["\']?([\d.]+)', rect_content)
                            h_match = re.search(r'\bheight=["\']?([\d.]+)', rect_content)

                            if all([x_match, y_match, w_match, h_match]):
                                x = float(x_match.group(1))
                                y = float(y_match.group(1))
                                width = float(w_match.group(1))
                                height = float(h_match.group(1))

                                # 替换 rect 和 text
                                image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

                                # 删除 text
                                svg_content = svg_content.replace(text_match.group(0), '')
                                # 替换 rect
                                svg_content = svg_content.replace(rect_content, image_tag, 1)

                                print(f"  {label}: 替换成功 (序号匹配 <text>) at ({x}, {y}) size {width}x{height}")
                                replaced = True
                                break

        # 回退：使用坐标匹配
        if not replaced:
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]

            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

            x1_int, y1_int = int(round(x1)), int(round(y1))

            # 精确匹配
            rect_pattern = rf'<rect[^>]*x=["\']?{x1_int}(?:\.0)?["\']?[^>]*y=["\']?{y1_int}(?:\.0)?["\']?[^>]*/?\s*>'
            if re.search(rect_pattern, svg_content):
                svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1)
                print(f"  {label}: 替换成功 (坐标精确匹配) at ({x1:.1f}, {y1:.1f})")
                replaced = True
            else:
                # 近似匹配
                tolerance = 10
                found = False
                for dx in range(-tolerance, tolerance+1, 2):
                    for dy in range(-tolerance, tolerance+1, 2):
                        search_x = x1_int + dx
                        search_y = y1_int + dy
                        rect_pattern = rf'<rect[^>]*x=["\']?{search_x}(?:\.0)?["\']?[^>]*y=["\']?{search_y}(?:\.0)?["\']?[^>]*(?:fill=["\']?(?:#[0-9A-Fa-f]{{3,6}}|gray|grey)["\']?|stroke=["\']?(?:black|#000|#000000)["\']?)[^>]*/?\s*>'
                        if re.search(rect_pattern, svg_content, re.IGNORECASE):
                            svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1, flags=re.IGNORECASE)
                            print(f"  {label}: 替换成功 (坐标近似匹配) at ({x1:.1f}, {y1:.1f})")
                            found = True
                            replaced = True
                            break
                    if found:
                        break

        if not replaced:
            # 追加到 SVG 末尾
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]
            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
            svg_content = svg_content.replace('</svg>', f'  {image_tag}\n</svg>')
            print(f"  {label}: 追加到 SVG at ({x1:.1f}, {y1:.1f}) (未找到匹配的占位符)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"最终 SVG 已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤 4.6：LLM 优化 SVG
# ============================================================================

def count_base64_images(svg_code: str) -> int:
    """统计 SVG 中嵌入的 base64 图片数量"""
    pattern = r'(?:href|xlink:href)=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    matches = re.findall(pattern, svg_code)
    return len(matches)


def validate_base64_images(svg_code: str, expected_count: int) -> tuple[bool, str]:
    """验证 SVG 中的 base64 图片是否完整"""
    actual_count = count_base64_images(svg_code)

    if actual_count < expected_count:
        return False, f"base64 图片数量不足: 期望 {expected_count}, 实际 {actual_count}"

    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    for match in re.finditer(pattern, svg_code):
        b64_data = match.group(1)
        if len(b64_data) % 4 != 0:
            return False, f"发现截断的 base64 数据（长度 {len(b64_data)} 不是 4 的倍数）"
        if len(b64_data) < 100:
            return False, f"发现过短的 base64 数据（长度 {len(b64_data)}），可能被截断"

    return True, f"base64 图片验证通过: {actual_count} 张图片"


def svg_to_png(svg_path: str, output_path: str, scale: float = 1.0) -> Optional[str]:
    """将 SVG 转换为 PNG"""
    try:
        _ensure_cairo_runtime()
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=output_path, scale=scale)
        return output_path
    except ImportError:
        print("  警告: cairosvg 未安装，尝试使用其他方法")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            return output_path
        except ImportError:
            print("  警告: svglib 也未安装，无法转换 SVG 到 PNG")
            return None
        except Exception as e:
            print(f"  警告: svglib 转换失败: {e}")
            return None
    except Exception as e:
        print(f"  警告: cairosvg 转换失败: {e}")
        return None


def optimize_svg_with_llm(
    figure_path: str,
    samed_path: str,
    final_svg_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_iterations: int = 2,
    skip_base64_validation: bool = False,
    no_icon_mode: bool = False,
    figure_mode: FigureMode = "simple_flowchart",
) -> str:
    """
    使用 LLM 优化 SVG，使其与原图更加对齐

    Args:
        figure_path: 原图路径
        samed_path: 标记图路径
        final_svg_path: 输入 SVG 路径
        output_path: 输出 SVG 路径
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_iterations: 最大迭代次数（0 表示跳过优化）
        skip_base64_validation: 是否跳过 base64 图片验证

    Returns:
        优化后的 SVG 路径
    """
    print("\n" + "=" * 60)
    print("步骤 4.6：LLM 优化 SVG（位置和样式对齐）")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"最大迭代次数: {max_iterations}")
    if no_icon_mode:
        print("无图标模式: 优化时禁止引入占位框")

    # 如果迭代次数为 0，直接复制文件并跳过优化
    if max_iterations == 0:
        print("  迭代次数为 0，跳过 LLM 优化")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_svg_path, output_path)
        print(f"  直接复制模板: {final_svg_path} -> {output_path}")
        return str(output_path)

    with open(final_svg_path, 'r', encoding='utf-8') as f:
        current_svg = f.read()

    output_dir = Path(final_svg_path).parent

    original_image_count = 0
    if not skip_base64_validation:
        original_image_count = count_base64_images(current_svg)
        print(f"原始 SVG 包含 {original_image_count} 张嵌入图片")
    else:
        print("跳过 base64 图片验证（模板 SVG）")

    for iteration in range(max_iterations):
        print(f"\n  优化迭代 {iteration + 1}/{max_iterations}")
        print("  " + "-" * 50)

        current_svg_path = output_dir / f"temp_svg_iter_{iteration}.svg"
        current_png_path = output_dir / f"temp_png_iter_{iteration}.png"

        with open(current_svg_path, 'w', encoding='utf-8') as f:
            f.write(current_svg)

        png_result = svg_to_png(str(current_svg_path), str(current_png_path))

        if png_result is None:
            print("  无法将 SVG 转换为 PNG，跳过优化")
            break

        figure_img = Image.open(figure_path)
        samed_img = Image.open(samed_path)
        current_png_img = Image.open(str(current_png_path))

        if no_icon_mode:
            prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The SAM reference image for this run. No valid icon boxes were detected.
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and optimize:
1. Overall layout and spatial alignment
2. Text positions, font sizes, and colors
3. Arrows, connectors, borders, and strokes
4. Shapes, grouping, and visual hierarchy

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- No valid icon placeholders exist for this figure
- Do NOT add gray rectangles, AF labels, placeholder groups, or synthetic icon boxes
- Focus on position and style corrections"""
            if figure_mode == "complex_paper":
                prompt += """
- Treat this as a complex paper figure: preserve nested panels, dashed boxes, formulas, brackets, legends, and small plots whenever they appear in the original."""
        else:
            prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The same figure with icon positions marked as gray rectangles with labels (<AF>01, <AF>02, etc.)
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and check the following **TWO MAJOR ASPECTS with EIGHT KEY POINTS**:

## ASPECT 1: POSITION (位置)
1. **Icons (图标)**: Are icon placeholder positions matching the original?
2. **Text (文字)**: Are text elements positioned correctly?
3. **Arrows (箭头)**: Are arrows starting/ending at correct positions?
4. **Lines/Borders (线条)**: Are lines and borders aligned properly?

## ASPECT 2: STYLE (样式)
5. **Icons (图标)**: Icon placeholder sizes, proportions (must have gray fill #808080, black border, and centered label)
6. **Text (文字)**: Font sizes, colors, weights
7. **Arrows (箭头)**: Arrow styles, thicknesses, colors
8. **Lines/Borders (线条)**: Line styles, colors, stroke widths

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- Keep all icon placeholder structures intact (the <g> elements with id like "AF01")
- Focus on position and style corrections"""

        contents = [prompt, figure_img, samed_img, current_png_img]

        try:
            print("  发送优化请求...")
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
                temperature=0.3,
            )

            if not content:
                print("  响应为空")
                continue

            optimized_svg = extract_svg_code(content)

            if not optimized_svg:
                print("  无法从响应中提取 SVG 代码")
                continue

            is_valid, errors = validate_svg_syntax(optimized_svg)

            if not is_valid:
                print(f"  优化后的 SVG 有语法错误，尝试修复...")
                optimized_svg = fix_svg_with_llm(
                    svg_code=optimized_svg,
                    errors=errors,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    provider=provider,
                )

            if not skip_base64_validation:
                images_valid, images_msg = validate_base64_images(optimized_svg, original_image_count)
                if not images_valid:
                    print(f"  警告: {images_msg}")
                    print("  拒绝此次优化，保留上一版本 SVG")
                    continue
                print(f"  {images_msg}")

            current_svg = optimized_svg
            print("  优化迭代完成")

        except Exception as e:
            print(f"  优化过程出错: {e}")
            continue

        try:
            current_svg_path.unlink(missing_ok=True)
            current_png_path.unlink(missing_ok=True)
        except:
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_svg = semanticize_svg_for_editing(current_svg)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(current_svg)

    final_png_path = output_path.with_suffix('.png')
    svg_to_png(str(output_path), str(final_png_path))
    print(f"\n  优化后的 SVG 已保存: {output_path}")
    print(f"  PNG 预览已保存: {final_png_path}")

    return str(output_path)


# ============================================================================
# 主函数：完整流程
# ============================================================================

