from __future__ import annotations

import json
from html import unescape
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from .svg_ops import _element_bbox, _local_name, semanticize_svg_for_editing


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_svg_style_classes(root: ET.Element) -> dict[str, dict[str, str]]:
    style_map: dict[str, dict[str, str]] = {}
    css_texts: list[str] = []
    for elem in root.iter():
        if _local_name(elem.tag) == "style" and elem.text:
            css_texts.append(elem.text)

    rule_pattern = re.compile(r"\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}")
    decl_pattern = re.compile(r"([A-Za-z-]+)\s*:\s*([^;]+)")
    for css_text in css_texts:
        for class_name, body in rule_pattern.findall(css_text):
            props: dict[str, str] = {}
            for key, value in decl_pattern.findall(body):
                props[key.strip()] = value.strip()
            style_map[class_name] = props
    return style_map


def _classes(elem: ET.Element) -> list[str]:
    return [token.strip() for token in (elem.get("class") or "").split() if token.strip()]


def _resolve_style(elem: ET.Element, style_map: dict[str, dict[str, str]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for cls in _classes(elem):
        resolved.update(style_map.get(cls, {}))
    if elem.get("fill"):
        resolved["fill"] = elem.get("fill", "")
    if elem.get("stroke"):
        resolved["stroke"] = elem.get("stroke", "")
    if elem.get("stroke-width"):
        resolved["stroke-width"] = elem.get("stroke-width", "")
    return resolved


def _collect_text(elem: ET.Element, skip_group_classes: set[str] | None = None) -> list[tuple[float, str]]:
    items: list[tuple[float, str]] = []
    skip_group_classes = skip_group_classes or set()
    for child in elem.iter():
        if child is not elem and _local_name(child.tag) == "g":
            if skip_group_classes.intersection(_classes(child)):
                continue
        if _local_name(child.tag) != "text":
            continue
        text = " ".join(part.strip() for part in child.itertext() if part and part.strip()).strip()
        if not text:
            continue
        y = 0.0
        bbox = _element_bbox(child)
        if bbox:
            y = bbox[1]
        items.append((y, text))
    items.sort(key=lambda item: item[0])
    return items


def _label_from_group(
    group: ET.Element,
    *,
    role: str,
    skip_group_classes: set[str] | None = None,
) -> str:
    texts = _collect_text(group, skip_group_classes=skip_group_classes)
    if not texts:
        return role.replace("_", " ").title()
    lines: list[str] = []
    for _, text in texts:
        if text in lines:
            continue
        lines.append(text)
        if len(lines) >= (2 if role == "panel" else 3):
            break
    return "<br>".join(_escape_html(line) for line in lines)


def _find_primary_rect(group: ET.Element) -> Optional[ET.Element]:
    rects: list[tuple[float, ET.Element]] = []
    for elem in group.iter():
        if _local_name(elem.tag) != "rect":
            continue
        bbox = _element_bbox(elem)
        if not bbox:
            continue
        rects.append(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), elem))
    if not rects:
        return None
    rects.sort(key=lambda item: item[0], reverse=True)
    return rects[0][1]


def _drawio_style_for_group(
    group: ET.Element,
    style_map: dict[str, dict[str, str]],
    role: str,
) -> str:
    primary_rect = _find_primary_rect(group)
    fill = "#ffffff"
    stroke = "#000000"
    stroke_width = "2"
    dashed = "0"
    rounded = "1"
    if primary_rect is not None:
        props = _resolve_style(primary_rect, style_map)
        fill = props.get("fill", fill)
        stroke = props.get("stroke", stroke)
        stroke_width = re.sub(r"[^0-9.]", "", props.get("stroke-width", stroke_width)) or stroke_width
        if "dasharray" in props.get("stroke-dasharray", "") or "dashed" in props.get("stroke-dasharray", ""):
            dashed = "1"
        if "border-dashed" in _classes(primary_rect):
            dashed = "1"
            fill = "none"

    if role == "panel":
        return (
            "rounded=1;whiteSpace=wrap;html=1;container=1;collapsible=0;"
            f"fillColor={fill if fill != 'none' else 'none'};"
            f"strokeColor={stroke};strokeWidth={stroke_width};dashed={dashed};"
            "fontStyle=1;verticalAlign=top;align=center;spacingTop=8;"
        )

    return (
        "rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill if fill != 'none' else '#ffffff'};"
        f"strokeColor={stroke};strokeWidth={stroke_width};dashed={dashed};"
        "verticalAlign=middle;align=center;spacing=6;"
    )


def _iter_groups(root: ET.Element, class_token: str) -> list[ET.Element]:
    groups: list[ET.Element] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "g":
            continue
        if class_token in _classes(elem):
            groups.append(elem)
    return groups


def _line_endpoints(elem: ET.Element) -> Optional[tuple[float, float, float, float]]:
    if _local_name(elem.tag) != "line":
        return None
    attrs = [elem.get("x1"), elem.get("y1"), elem.get("x2"), elem.get("y2")]
    if any(v is None for v in attrs):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in attrs]
    except ValueError:
        return None
    return x1, y1, x2, y2


def _style_value(elem: ET.Element, key: str) -> Optional[str]:
    style = elem.get("style") or ""
    match = re.search(rf"{re.escape(key)}\s*:\s*([^;]+)", style)
    if not match:
        return None
    return match.group(1).strip()


def _has_marker(elem: ET.Element, key: str) -> bool:
    direct = elem.get(key)
    if direct:
        return True
    style_value = _style_value(elem, key)
    return bool(style_value and style_value.lower() != "none")


def _polyline_points(elem: ET.Element) -> list[tuple[float, float]]:
    raw = elem.get("points") or ""
    nums = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if len(nums) < 4:
        return []
    points: list[tuple[float, float]] = []
    for idx in range(0, len(nums) - 1, 2):
        try:
            points.append((float(nums[idx]), float(nums[idx + 1])))
        except ValueError:
            continue
    return points


def _path_points_from_d(path_d: str) -> list[tuple[float, float]]:
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", path_d or "")
    if not tokens:
        return []
    points: list[tuple[float, float]] = []
    idx = 0
    cmd = None
    cur_x = 0.0
    cur_y = 0.0
    start_x = 0.0
    start_y = 0.0

    def append_point(x: float, y: float) -> None:
        nonlocal cur_x, cur_y
        cur_x, cur_y = x, y
        points.append((x, y))

    while idx < len(tokens):
        token = tokens[idx]
        if re.fullmatch(r"[A-Za-z]", token):
            cmd = token
            idx += 1
            if cmd in {"Z", "z"}:
                append_point(start_x, start_y)
            continue
        if cmd is None:
            idx += 1
            continue

        def read_number() -> Optional[float]:
            nonlocal idx
            if idx >= len(tokens) or re.fullmatch(r"[A-Za-z]", tokens[idx]):
                return None
            try:
                value = float(tokens[idx])
            except ValueError:
                return None
            idx += 1
            return value

        if cmd in {"M", "m"}:
            x = read_number()
            y = read_number()
            if x is None or y is None:
                break
            if cmd == "m":
                x += cur_x
                y += cur_y
            append_point(x, y)
            start_x, start_y = x, y
            cmd = "L" if cmd == "M" else "l"
            continue

        if cmd in {"L", "l"}:
            x = read_number()
            y = read_number()
            if x is None or y is None:
                break
            if cmd == "l":
                x += cur_x
                y += cur_y
            append_point(x, y)
            continue

        if cmd in {"H", "h"}:
            x = read_number()
            if x is None:
                break
            if cmd == "h":
                x += cur_x
            append_point(x, cur_y)
            continue

        if cmd in {"V", "v"}:
            y = read_number()
            if y is None:
                break
            if cmd == "v":
                y += cur_y
            append_point(cur_x, y)
            continue

        if cmd in {"C", "c"}:
            numbers = [read_number() for _ in range(6)]
            if any(value is None for value in numbers):
                break
            x = numbers[4]
            y = numbers[5]
            if cmd == "c":
                x += cur_x
                y += cur_y
            append_point(x, y)
            continue

        if cmd in {"S", "s", "Q", "q"}:
            numbers = [read_number() for _ in range(4)]
            if any(value is None for value in numbers):
                break
            x = numbers[2]
            y = numbers[3]
            if cmd in {"s", "q"}:
                x += cur_x
                y += cur_y
            append_point(x, y)
            continue

        if cmd in {"T", "t"}:
            x = read_number()
            y = read_number()
            if x is None or y is None:
                break
            if cmd == "t":
                x += cur_x
                y += cur_y
            append_point(x, y)
            continue

        if cmd in {"A", "a"}:
            numbers = [read_number() for _ in range(7)]
            if any(value is None for value in numbers):
                break
            x = numbers[5]
            y = numbers[6]
            if cmd == "a":
                x += cur_x
                y += cur_y
            append_point(x, y)
            continue

        idx += 1

    return points


def _connector_points(elem: ET.Element) -> list[tuple[float, float]]:
    tag = _local_name(elem.tag)
    if tag == "line":
        endpoints = _line_endpoints(elem)
        if not endpoints:
            return []
        x1, y1, x2, y2 = endpoints
        return [(x1, y1), (x2, y2)]
    if tag in {"polyline", "polygon"}:
        return _polyline_points(elem)
    if tag == "path":
        return _path_points_from_d(elem.get("d") or "")
    return []


def _distance_point_to_bbox_sq(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = 0.0
    dy = 0.0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    return dx * dx + dy * dy


def _distance_sq(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def _nearest_group_id(
    point: tuple[float, float],
    groups: list[tuple[str, tuple[float, float, float, float]]],
    tolerance: float = 180.0,
) -> Optional[str]:
    best_id = None
    best_dist = tolerance * tolerance
    for group_id, bbox in groups:
        dist = _distance_point_to_bbox_sq(point, bbox)
        if dist < best_dist:
            best_dist = dist
            best_id = group_id
    return best_id


def _overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    inter = min(a2, b2) - max(a1, b1)
    if inter <= 0:
        return 0.0
    denom = max(min(a2 - a1, b2 - b1), 1.0)
    return inter / denom


def _infer_adjacency_side(
    source_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float],
) -> tuple[str, str] | None:
    sx1, sy1, sx2, sy2 = source_bbox
    tx1, ty1, tx2, ty2 = target_bbox
    horizontal_overlap = _overlap_ratio(sy1, sy2, ty1, ty2)
    vertical_overlap = _overlap_ratio(sx1, sx2, tx1, tx2)

    if horizontal_overlap >= 0.35:
        if sx2 <= tx1:
            return "right", "left"
        if tx2 <= sx1:
            return "left", "right"
    if vertical_overlap >= 0.35:
        if sy2 <= ty1:
            return "bottom", "top"
        if ty2 <= sy1:
            return "top", "bottom"
    return None


def _anchor_point_for_side(
    bbox: tuple[float, float, float, float],
    side: str,
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    if side == "left":
        return x1, (y1 + y2) / 2
    if side == "right":
        return x2, (y1 + y2) / 2
    if side == "top":
        return (x1 + x2) / 2, y1
    return (x1 + x2) / 2, y2


def _has_blocking_module_between(
    source_id: str,
    target_id: str,
    orientation: str,
    module_cells: list[tuple[str, tuple[float, float, float, float]]],
) -> bool:
    bbox_map = {module_id: bbox for module_id, bbox in module_cells}
    source_bbox = bbox_map[source_id]
    target_bbox = bbox_map[target_id]
    sx1, sy1, sx2, sy2 = source_bbox
    tx1, ty1, tx2, ty2 = target_bbox

    if orientation == "horizontal":
        x_left = min(sx2, tx2)
        x_right = max(sx1, tx1)
        y_band_1 = max(sy1, ty1) - 18
        y_band_2 = min(sy2, ty2) + 18
        for other_id, other_bbox in module_cells:
            if other_id in {source_id, target_id}:
                continue
            cx = (other_bbox[0] + other_bbox[2]) / 2
            cy = (other_bbox[1] + other_bbox[3]) / 2
            if x_left < cx < x_right and y_band_1 <= cy <= y_band_2:
                return True
        return False

    y_top = min(sy2, ty2)
    y_bottom = max(sy1, ty1)
    x_band_1 = max(sx1, tx1) - 18
    x_band_2 = min(sx2, tx2) + 18
    for other_id, other_bbox in module_cells:
        if other_id in {source_id, target_id}:
            continue
        cx = (other_bbox[0] + other_bbox[2]) / 2
        cy = (other_bbox[1] + other_bbox[3]) / 2
        if y_top < cy < y_bottom and x_band_1 <= cx <= x_band_2:
            return True
    return False


def _append_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    source_id: str,
    target_id: str,
    source_point: tuple[float, float],
    target_point: tuple[float, float],
    module_bbox_lookup: dict[str, tuple[float, float, float, float]],
    recovery: str,
) -> None:
    if not source_id or not target_id or source_id == target_id:
        return
    key = (source_id, target_id)
    if key in seen:
        return
    seen.add(key)
    source_bbox = module_bbox_lookup[source_id]
    target_bbox = module_bbox_lookup[target_id]
    source_side = _infer_anchor_side(source_point, source_bbox)
    target_side = _infer_anchor_side(target_point, target_bbox)
    edges.append(
        {
            "id": f"edge-{len(edges) + 1:03d}",
            "source": source_id,
            "target": target_id,
            "source_port": f"{source_id}__{source_side}",
            "target_port": f"{target_id}__{target_side}",
            "source_anchor": {
                "side": source_side,
                "x": source_point[0],
                "y": source_point[1],
            },
            "target_anchor": {
                "side": target_side,
                "x": target_point[0],
                "y": target_point[1],
            },
            "recovery": recovery,
            "drawio_style": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=2;",
        }
    )


def _infer_edges_from_connectors(
    root: ET.Element,
    *,
    style_map: dict[str, dict[str, str]],
    module_cells: list[tuple[str, tuple[float, float, float, float]]],
    module_bbox_lookup: dict[str, tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    valid_tags = {"line", "polyline", "polygon", "path"}
    for elem in root.iter():
        tag = _local_name(elem.tag)
        if tag not in valid_tags:
            continue
        points = _connector_points(elem)
        if len(points) < 2:
            continue
        style = _resolve_style(elem, style_map)
        marker_start = _has_marker(elem, "marker-start")
        marker_end = _has_marker(elem, "marker-end")
        marker_mid = _has_marker(elem, "marker-mid")
        stroke = style.get("stroke", elem.get("stroke", ""))
        fill = style.get("fill", elem.get("fill", "none"))
        if not stroke and not (marker_start or marker_end or marker_mid):
            continue
        if tag == "path" and fill not in {"", "none", "transparent"} and not (marker_start or marker_end):
            continue

        start_point = points[0]
        end_point = points[-1]
        if marker_start and not marker_end:
            start_point, end_point = end_point, start_point

        source_id = _nearest_group_id(start_point, module_cells, tolerance=140.0)
        target_id = _nearest_group_id(end_point, module_cells, tolerance=140.0)
        if not source_id or not target_id or source_id == target_id:
            continue
        _append_edge(
            edges,
            seen,
            source_id=source_id,
            target_id=target_id,
            source_point=start_point,
            target_point=end_point,
            module_bbox_lookup=module_bbox_lookup,
            recovery="connector",
        )
    return edges


def _infer_edges_from_adjacency(
    module_cells: list[tuple[str, tuple[float, float, float, float]]],
    *,
    existing_edges: list[dict[str, Any]],
    canvas_width: float,
    canvas_height: float,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen = {(edge["source"], edge["target"]) for edge in existing_edges}
    bbox_lookup = {module_id: bbox for module_id, bbox in module_cells}
    max_h_gap = min(max(canvas_width * 0.08, 70.0), 180.0)
    max_v_gap = min(max(canvas_height * 0.08, 60.0), 160.0)

    for idx, (source_id, source_bbox) in enumerate(module_cells):
        sx1, sy1, sx2, sy2 = source_bbox
        for target_id, target_bbox in module_cells[idx + 1 :]:
            tx1, ty1, tx2, ty2 = target_bbox
            pair = _infer_adjacency_side(source_bbox, target_bbox)
            if not pair:
                continue
            source_side, target_side = pair
            if source_side in {"left", "right"}:
                gap = max(tx1 - sx2, sx1 - tx2, 0.0)
                if gap > max_h_gap or _has_blocking_module_between(source_id, target_id, "horizontal", module_cells):
                    continue
            else:
                gap = max(ty1 - sy2, sy1 - ty2, 0.0)
                if gap > max_v_gap or _has_blocking_module_between(source_id, target_id, "vertical", module_cells):
                    continue

            if source_side == "left":
                src_id, tgt_id = target_id, source_id
                src_side, tgt_side = "right", "left"
            elif source_side == "top":
                src_id, tgt_id = target_id, source_id
                src_side, tgt_side = "bottom", "top"
            else:
                src_id, tgt_id = source_id, target_id
                src_side, tgt_side = source_side, target_side

            _append_edge(
                edges,
                seen,
                source_id=src_id,
                target_id=tgt_id,
                source_point=_anchor_point_for_side(bbox_lookup[src_id], src_side),
                target_point=_anchor_point_for_side(bbox_lookup[tgt_id], tgt_side),
                module_bbox_lookup=bbox_lookup,
                recovery="adjacency",
            )
    return edges


def _containing_panel_id(
    bbox: tuple[float, float, float, float],
    panels: list[tuple[str, tuple[float, float, float, float]]],
) -> Optional[str]:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    best_id: Optional[str] = None
    best_area: Optional[float] = None
    for panel_id, panel_bbox in panels:
        if not (panel_bbox[0] <= cx <= panel_bbox[2] and panel_bbox[1] <= cy <= panel_bbox[3]):
            continue
        area = (panel_bbox[2] - panel_bbox[0]) * (panel_bbox[3] - panel_bbox[1])
        if best_area is None or area < best_area:
            best_id = panel_id
            best_area = area
    return best_id


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _bbox_to_geometry(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x": float(bbox[0]),
        "y": float(bbox[1]),
        "width": float(max(1.0, bbox[2] - bbox[0])),
        "height": float(max(1.0, bbox[3] - bbox[1])),
    }


def _parse_mx_style(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in style.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            result[token] = "1"
            continue
        key, value = token.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _html_lines(value: str) -> list[str]:
    if not value:
        return []
    text = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    return [unescape(line).strip() for line in text.splitlines() if line.strip()]


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _has_group_ancestor(elem: ET.Element, parents: dict[ET.Element, ET.Element], class_token: str) -> bool:
    current = parents.get(elem)
    while current is not None:
        if _local_name(current.tag) == "g" and class_token in _classes(current):
            return True
        current = parents.get(current)
    return False


def _extract_ocr_blocks(image_path: str, existing_boxes: list[tuple[float, float, float, float]]) -> tuple[list[dict[str, Any]], str]:
    tesseract_bin = _resolve_tesseract_bin()
    if not tesseract_bin or not image_path or not Path(image_path).is_file():
        return [], "unavailable"

    result = subprocess.run(
        [tesseract_bin, image_path, "stdout", "--psm", "11", "tsv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0 or not result.stdout.strip():
        return [], "failed"

    rows = result.stdout.splitlines()
    if len(rows) < 2:
        return [], "empty"
    header = rows[0].split("\t")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) != len(header):
            continue
        data = dict(zip(header, cols))
        text = (data.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(data.get("conf", "-1"))
            left = float(data.get("left", "0"))
            top = float(data.get("top", "0"))
            width = float(data.get("width", "0"))
            height = float(data.get("height", "0"))
        except ValueError:
            continue
        if conf < 45 or width <= 1 or height <= 1:
            continue
        key = (
            data.get("block_num", "0"),
            data.get("par_num", "0"),
            data.get("line_num", "0"),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": conf,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        )

    blocks: list[dict[str, Any]] = []
    for line_items in grouped.values():
        text = " ".join(item["text"] for item in line_items if item["text"]).strip()
        if not text:
            continue
        x1 = min(item["left"] for item in line_items)
        y1 = min(item["top"] for item in line_items)
        x2 = max(item["left"] + item["width"] for item in line_items)
        y2 = max(item["top"] + item["height"] for item in line_items)
        bbox = (x1, y1, x2, y2)
        if any(_bbox_iou(bbox, existing) > 0.55 for existing in existing_boxes):
            continue
        blocks.append(
            {
                "id": f"ocr-{len(blocks) + 1:03d}",
                "text": text,
                "bbox": _bbox_to_geometry(bbox),
                "source": "ocr",
            }
        )
    return blocks, "tesseract"


def _resolve_tesseract_bin() -> Optional[str]:
    candidates = [
        shutil.which("tesseract"),
        str(Path("C:/Program Files/Tesseract-OCR/tesseract.exe")),
        str(Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _relative_geometry(
    bbox: dict[str, float],
    parent_bbox: Optional[dict[str, float]],
) -> dict[str, float]:
    if not parent_bbox:
        return dict(bbox)
    return {
        "x": bbox["x"] - parent_bbox["x"],
        "y": bbox["y"] - parent_bbox["y"],
        "width": bbox["width"],
        "height": bbox["height"],
    }


def _port_position(bbox: dict[str, float], side: str, size: float = 10.0) -> dict[str, float]:
    if side == "left":
        return {"x": -size / 2, "y": bbox["height"] / 2 - size / 2, "width": size, "height": size}
    if side == "right":
        return {"x": bbox["width"] - size / 2, "y": bbox["height"] / 2 - size / 2, "width": size, "height": size}
    if side == "top":
        return {"x": bbox["width"] / 2 - size / 2, "y": -size / 2, "width": size, "height": size}
    return {"x": bbox["width"] / 2 - size / 2, "y": bbox["height"] - size / 2, "width": size, "height": size}


def _infer_anchor_side(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> str:
    px, py = point
    x1, y1, x2, y2 = bbox
    distances = {
        "left": abs(px - x1),
        "right": abs(px - x2),
        "top": abs(py - y1),
        "bottom": abs(py - y2),
    }
    return min(distances, key=distances.get)


def _center_in_bbox(inner_bbox: dict[str, float], outer_bbox: dict[str, float]) -> bool:
    cx = inner_bbox["x"] + inner_bbox["width"] / 2
    cy = inner_bbox["y"] + inner_bbox["height"] / 2
    return (
        outer_bbox["x"] <= cx <= outer_bbox["x"] + outer_bbox["width"]
        and outer_bbox["y"] <= cy <= outer_bbox["y"] + outer_bbox["height"]
    )


def extract_scene_graph_from_svg(
    svg_path: str,
    *,
    figure_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    svg_text = Path(svg_path).read_text(encoding="utf-8", errors="ignore")
    svg_text = semanticize_svg_for_editing(svg_text)
    root = ET.fromstring(svg_text.encode("utf-8"))
    style_map = _parse_svg_style_classes(root)
    parents = _parent_map(root)

    view_box = root.get("viewBox", "0 0 1600 900").split()
    if len(view_box) >= 4:
        canvas_width = float(view_box[2])
        canvas_height = float(view_box[3])
    else:
        canvas_width = 1600.0
        canvas_height = 900.0
    canvas_area = max(canvas_width * canvas_height, 1.0)

    panels: list[dict[str, Any]] = []
    panel_cells: list[tuple[str, tuple[float, float, float, float]]] = []
    for panel in _iter_groups(root, "fw-panel"):
        bbox = _element_bbox(panel)
        if not bbox:
            continue
        panel_id = panel.get("id") or f"panel-{len(panels) + 1:02d}"
        panels.append(
            {
                "id": panel_id,
                "label": _label_from_group(panel, role="panel", skip_group_classes={"fw-module"}),
                "bbox": _bbox_to_geometry(bbox),
                "drawio_style": _drawio_style_for_group(panel, style_map, "panel"),
            }
        )
        panel_cells.append((panel_id, bbox))
    panel_bbox_map = {panel["id"]: panel["bbox"] for panel in panels}

    modules: list[dict[str, Any]] = []
    module_cells: list[tuple[str, tuple[float, float, float, float]]] = []
    accepted_module_bboxes: list[tuple[float, float, float, float]] = []
    for module in _iter_groups(root, "fw-module"):
        bbox = _element_bbox(module)
        if not bbox:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = (width * height) / canvas_area
        if area_ratio > 0.35 or width > canvas_width * 0.92 or height > canvas_height * 0.92:
            continue
        if any(_bbox_iou(bbox, existing) > 0.92 for existing in accepted_module_bboxes):
            continue
        module_id = module.get("id") or f"module-{len(modules) + 1:02d}"
        geom = _bbox_to_geometry(bbox)
        modules.append(
            {
                "id": module_id,
                "label": _label_from_group(module, role="module"),
                "bbox": geom,
                "parent_panel_id": _containing_panel_id(bbox, panel_cells),
                "drawio_style": _drawio_style_for_group(module, style_map, "module"),
                "ports": {
                    side: {
                        "id": f"{module_id}__{side}",
                        "side": side,
                        "geometry": _port_position(geom, side),
                    }
                    for side in ("left", "right", "top", "bottom")
                },
            }
        )
        module_cells.append((module_id, bbox))
        accepted_module_bboxes.append(bbox)
    module_bbox_map = {module["id"]: module["bbox"] for module in modules}
    module_bbox_lookup = {module_id: bbox for module_id, bbox in module_cells}

    text_blocks: list[dict[str, Any]] = []
    existing_text_bboxes: list[tuple[float, float, float, float]] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "text":
            continue
        if _has_group_ancestor(elem, parents, "fw-module"):
            continue
        bbox = _element_bbox(elem)
        text = " ".join(part.strip() for part in elem.itertext() if part and part.strip()).strip()
        if not bbox or not text:
            continue
        geom = _bbox_to_geometry(bbox)
        owner_module_id = next(
            (module["id"] for module in modules if _center_in_bbox(geom, module["bbox"])),
            None,
        )
        owner_panel_id = None
        if owner_module_id is None:
            owner_panel_id = next(
                (panel["id"] for panel in panels if _center_in_bbox(geom, panel["bbox"])),
                None,
            )
        text_blocks.append(
            {
                "id": elem.get("id") or f"text-{len(text_blocks) + 1:03d}",
                "text": text,
                "bbox": geom,
                "source": "svg",
                "owner_module_id": owner_module_id,
                "owner_panel_id": owner_panel_id,
            }
        )
        existing_text_bboxes.append(bbox)

    ocr_blocks, ocr_backend = _extract_ocr_blocks(figure_path or "", existing_text_bboxes)
    for block in ocr_blocks:
        geom = block["bbox"]
        owner_module_id = next(
            (module["id"] for module in modules if _center_in_bbox(geom, module["bbox"])),
            None,
        )
        owner_panel_id = None
        if owner_module_id is None:
            owner_panel_id = next(
                (panel["id"] for panel in panels if _center_in_bbox(geom, panel["bbox"])),
                None,
            )
        block["owner_module_id"] = owner_module_id
        block["owner_panel_id"] = owner_panel_id
    text_blocks.extend(ocr_blocks)

    edges = _infer_edges_from_connectors(
        root,
        style_map=style_map,
        module_cells=module_cells,
        module_bbox_lookup=module_bbox_lookup,
    )
    edges.extend(
        _infer_edges_from_adjacency(
            module_cells,
            existing_edges=edges,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
    )
    connector_edges = sum(1 for edge in edges if edge.get("recovery") == "connector")
    adjacency_edges = sum(1 for edge in edges if edge.get("recovery") == "adjacency")

    scene_graph = {
        "version": 1,
        "canvas": {"width": canvas_width, "height": canvas_height},
        "panels": panels,
        "modules": modules,
        "text_blocks": text_blocks,
        "edges": edges,
        "meta": {
            "source_svg": str(Path(svg_path)),
            "figure_path": str(Path(figure_path)) if figure_path else None,
            "ocr_backend": ocr_backend,
            "module_count": len(modules),
            "panel_count": len(panels),
            "text_count": len(text_blocks),
            "edge_count": len(edges),
            "owned_text_count": sum(1 for block in text_blocks if block.get("owner_module_id") or block.get("owner_panel_id")),
            "anchored_edge_count": sum(1 for edge in edges if edge.get("source_anchor") and edge.get("target_anchor")),
            "edge_recovery": {
                "connector": connector_edges,
                "adjacency": adjacency_edges,
            },
        },
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(scene_graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return scene_graph


def normalize_scene_graph(scene_graph: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(scene_graph))
    canvas = normalized.setdefault("canvas", {})
    canvas["width"] = float(canvas.get("width", 1600))
    canvas["height"] = float(canvas.get("height", 900))

    panels = normalized.setdefault("panels", [])
    for idx, panel in enumerate(panels, start=1):
        panel.setdefault("id", f"panel-{idx:02d}")
        panel.setdefault("label", f"Panel {idx}")
        panel["bbox"] = {
            "x": float(panel.get("bbox", {}).get("x", 0)),
            "y": float(panel.get("bbox", {}).get("y", 0)),
            "width": float(panel.get("bbox", {}).get("width", 200)),
            "height": float(panel.get("bbox", {}).get("height", 120)),
        }
        panel.setdefault(
            "drawio_style",
            "rounded=1;whiteSpace=wrap;html=1;container=1;collapsible=0;fillColor=none;strokeColor=#000000;strokeWidth=2;dashed=1;fontStyle=1;verticalAlign=top;align=center;spacingTop=8;",
        )

    panel_bbox_map = {panel["id"]: panel["bbox"] for panel in panels}

    modules = normalized.setdefault("modules", [])
    for idx, module in enumerate(modules, start=1):
        module.setdefault("id", f"module-{idx:02d}")
        module.setdefault("label", f"Module {idx}")
        module["bbox"] = {
            "x": float(module.get("bbox", {}).get("x", 0)),
            "y": float(module.get("bbox", {}).get("y", 0)),
            "width": float(module.get("bbox", {}).get("width", 160)),
            "height": float(module.get("bbox", {}).get("height", 100)),
        }
        if module.get("parent_panel_id") not in panel_bbox_map:
            module["parent_panel_id"] = None
        module.setdefault(
            "drawio_style",
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;verticalAlign=middle;align=center;spacing=6;",
        )
        bbox = module["bbox"]
        module["ports"] = {
            side: {
                "id": module.get("ports", {}).get(side, {}).get("id", f"{module['id']}__{side}"),
                "side": side,
                "geometry": module.get("ports", {}).get(side, {}).get("geometry", _port_position(bbox, side)),
            }
            for side in ("left", "right", "top", "bottom")
        }

    module_bbox_lookup = {
        module["id"]: (
            module["bbox"]["x"],
            module["bbox"]["y"],
            module["bbox"]["x"] + module["bbox"]["width"],
            module["bbox"]["y"] + module["bbox"]["height"],
        )
        for module in modules
    }

    text_blocks = normalized.setdefault("text_blocks", [])
    for idx, block in enumerate(text_blocks, start=1):
        block.setdefault("id", f"text-{idx:03d}")
        block.setdefault("text", "")
        block["bbox"] = {
            "x": float(block.get("bbox", {}).get("x", 0)),
            "y": float(block.get("bbox", {}).get("y", 0)),
            "width": float(block.get("bbox", {}).get("width", 80)),
            "height": float(block.get("bbox", {}).get("height", 24)),
        }
        if block.get("owner_module_id") not in module_bbox_lookup:
            block["owner_module_id"] = None
        if block.get("owner_panel_id") not in panel_bbox_map:
            block["owner_panel_id"] = None
        if block.get("owner_module_id") is None:
            block["owner_module_id"] = next(
                (module["id"] for module in modules if _center_in_bbox(block["bbox"], module["bbox"])),
                None,
            )
        if block.get("owner_panel_id") is None and block.get("owner_module_id") is None:
            block["owner_panel_id"] = next(
                (panel["id"] for panel in panels if _center_in_bbox(block["bbox"], panel["bbox"])),
                None,
            )

    edges = normalized.setdefault("edges", [])
    for idx, edge in enumerate(edges, start=1):
        edge.setdefault("id", f"edge-{idx:03d}")
        if edge.get("source") not in module_bbox_lookup or edge.get("target") not in module_bbox_lookup:
            continue
        edge.setdefault("recovery", "manual")
        source_bbox = module_bbox_lookup[edge["source"]]
        target_bbox = module_bbox_lookup[edge["target"]]
        source_anchor = edge.get("source_anchor") or {}
        target_anchor = edge.get("target_anchor") or {}
        inferred_pair = _infer_adjacency_side(source_bbox, target_bbox) or ("right", "left")
        source_side = source_anchor.get("side") or inferred_pair[0]
        target_side = target_anchor.get("side") or inferred_pair[1]
        sx, sy = _anchor_point_for_side(source_bbox, source_side)
        tx, ty = _anchor_point_for_side(target_bbox, target_side)
        edge["source_port"] = edge.get("source_port") or f"{edge['source']}__{source_side}"
        edge["target_port"] = edge.get("target_port") or f"{edge['target']}__{target_side}"
        edge["source_anchor"] = {"side": source_side, "x": source_anchor.get("x", sx), "y": source_anchor.get("y", sy)}
        edge["target_anchor"] = {"side": target_side, "x": target_anchor.get("x", tx), "y": target_anchor.get("y", ty)}
        edge.setdefault(
            "drawio_style",
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=2;",
        )

    meta = normalized.setdefault("meta", {})
    meta["module_count"] = len(modules)
    meta["panel_count"] = len(panels)
    meta["text_count"] = len(text_blocks)
    meta["edge_count"] = len(edges)
    meta["owned_text_count"] = sum(1 for block in text_blocks if block.get("owner_module_id") or block.get("owner_panel_id"))
    meta["anchored_edge_count"] = sum(1 for edge in edges if edge.get("source_anchor") and edge.get("target_anchor"))
    meta["edge_recovery"] = {
        "connector": sum(1 for edge in edges if edge.get("recovery") == "connector"),
        "adjacency": sum(1 for edge in edges if edge.get("recovery") == "adjacency"),
        "manual": sum(1 for edge in edges if edge.get("recovery") not in {"connector", "adjacency"}),
    }
    return normalized


def export_drawio_from_scene_graph(
    scene_graph: dict[str, Any],
    output_path: str,
    *,
    page_name: str = "FigureWeave",
) -> str:
    scene_graph = normalize_scene_graph(scene_graph)
    canvas_width = float(scene_graph.get("canvas", {}).get("width", 1600))
    canvas_height = float(scene_graph.get("canvas", {}).get("height", 900))

    mxfile = ET.Element(
        "mxfile",
        host="app.diagrams.net",
        modified="2026-04-08T00:00:00Z",
        agent="FigureWeave",
        compressed="false",
    )
    diagram = ET.SubElement(mxfile, "diagram", id="figureweave-page-1", name=page_name)
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1600",
        dy="900",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(int(max(canvas_width, 1600))),
        pageHeight=str(int(max(canvas_height, 900))),
        math="0",
        shadow="0",
    )
    root_node = ET.SubElement(model, "root")
    ET.SubElement(root_node, "mxCell", id="0")
    ET.SubElement(root_node, "mxCell", id="1", parent="0")

    panel_ids = {panel["id"] for panel in scene_graph.get("panels", [])}
    panel_bbox_map = {panel["id"]: panel["bbox"] for panel in scene_graph.get("panels", [])}
    module_bbox_map = {module["id"]: module["bbox"] for module in scene_graph.get("modules", [])}

    for panel in scene_graph.get("panels", []):
        bbox = panel["bbox"]
        cell = ET.SubElement(
            root_node,
            "mxCell",
            id=panel["id"],
            value=panel.get("label", ""),
            style=panel.get("drawio_style", ""),
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=f"{bbox['x']:.1f}",
            y=f"{bbox['y']:.1f}",
            width=f"{bbox['width']:.1f}",
            height=f"{bbox['height']:.1f}",
            **{"as": "geometry"},
        )

    for module in scene_graph.get("modules", []):
        bbox = module["bbox"]
        parent_id = module.get("parent_panel_id")
        parent_bbox = None
        if parent_id not in panel_ids:
            parent_id = "1"
        else:
            parent_bbox = panel_bbox_map.get(parent_id)
        geom = _relative_geometry(bbox, parent_bbox)
        cell = ET.SubElement(
            root_node,
            "mxCell",
            id=module["id"],
            value=module.get("label", ""),
            style=module.get("drawio_style", ""),
            vertex="1",
            parent=parent_id,
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=f"{geom['x']:.1f}",
            y=f"{geom['y']:.1f}",
            width=f"{geom['width']:.1f}",
            height=f"{geom['height']:.1f}",
            **{"as": "geometry"},
        )
        for port in module.get("ports", {}).values():
            port_geom = port["geometry"]
            port_cell = ET.SubElement(
                root_node,
                "mxCell",
                id=port["id"],
                value="",
                style=(
                    "shape=ellipse;perimeter=ellipsePerimeter;fillColor=#1f1f1f;"
                    "strokeColor=#1f1f1f;opacity=0;connectable=1;movable=0;resizable=0;"
                    "deletable=0;editable=0;fwPort=1;"
                ),
                vertex="1",
                connectable="1",
                parent=module["id"],
            )
            ET.SubElement(
                port_cell,
                "mxGeometry",
                x=f"{port_geom['x']:.1f}",
                y=f"{port_geom['y']:.1f}",
                width=f"{port_geom['width']:.1f}",
                height=f"{port_geom['height']:.1f}",
                **{"as": "geometry"},
            )

    for text_block in scene_graph.get("text_blocks", []):
        bbox = text_block["bbox"]
        owner_id = text_block.get("owner_module_id") or text_block.get("owner_panel_id")
        parent_bbox = None
        if text_block.get("owner_module_id") in module_bbox_map:
            parent_bbox = module_bbox_map[text_block["owner_module_id"]]
        elif text_block.get("owner_panel_id") in panel_bbox_map:
            parent_bbox = panel_bbox_map[text_block["owner_panel_id"]]
        if owner_id not in panel_ids and owner_id not in module_bbox_map:
            owner_id = "1"
            parent_bbox = None
        geom = _relative_geometry(bbox, parent_bbox)
        cell = ET.SubElement(
            root_node,
            "mxCell",
            id=text_block["id"],
            value=_escape_html(text_block.get("text", "")),
            style="shape=text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;",
            vertex="1",
            parent=owner_id,
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=f"{geom['x']:.1f}",
            y=f"{geom['y']:.1f}",
            width=f"{geom['width']:.1f}",
            height=f"{geom['height']:.1f}",
            **{"as": "geometry"},
        )

    for edge in scene_graph.get("edges", []):
        style = edge.get(
            "drawio_style",
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=2;",
        )
        source_anchor = edge.get("source_anchor", {})
        target_anchor = edge.get("target_anchor", {})
        side_to_entry = {
            "left": ("0", "0.5"),
            "right": ("1", "0.5"),
            "top": ("0.5", "0"),
            "bottom": ("0.5", "1"),
        }
        if source_anchor.get("side") in side_to_entry:
            exit_x, exit_y = side_to_entry[source_anchor["side"]]
            style += f"exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0;"
        if target_anchor.get("side") in side_to_entry:
            entry_x, entry_y = side_to_entry[target_anchor["side"]]
            style += f"entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0;"
        edge_cell = ET.SubElement(
            root_node,
            "mxCell",
            id=edge["id"],
            style=style,
            edge="1",
            parent="1",
            source=edge.get("source_port") or edge["source"],
            target=edge.get("target_port") or edge["target"],
        )
        ET.SubElement(edge_cell, "mxGeometry", relative="1", **{"as": "geometry"})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mxfile).write(output_path, encoding="utf-8", xml_declaration=True)
    return str(Path(output_path))


def render_drawio_to_svg(
    drawio_path: str,
    *,
    output_path: Optional[str] = None,
) -> str:
    root = ET.parse(drawio_path).getroot()
    model = root.find(".//mxGraphModel")
    if model is None:
        raise ValueError("Invalid DrawIO XML: missing mxGraphModel")

    canvas_width = float(model.get("pageWidth", "1600"))
    canvas_height = float(model.get("pageHeight", "900"))
    mx_root = model.find("root")
    if mx_root is None:
        raise ValueError("Invalid DrawIO XML: missing root")

    cells = {cell.get("id"): cell for cell in mx_root.findall("mxCell") if cell.get("id")}
    geometries: dict[str, dict[str, float]] = {}
    for cell_id, cell in cells.items():
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        geometries[cell_id] = {
            "x": float(geom.get("x", "0") or 0),
            "y": float(geom.get("y", "0") or 0),
            "width": float(geom.get("width", "0") or 0),
            "height": float(geom.get("height", "0") or 0),
        }
    abs_geometry_cache: dict[str, dict[str, float]] = {}

    def effective_geometry(cell_id: str) -> dict[str, float]:
        if cell_id in abs_geometry_cache:
            return abs_geometry_cache[cell_id]
        geom = dict(geometries.get(cell_id, {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}))
        cell = cells.get(cell_id)
        parent_id = cell.get("parent") if cell is not None else None
        if parent_id and parent_id in geometries and parent_id not in {"0", "1"}:
            parent_geom = effective_geometry(parent_id)
            geom["x"] += parent_geom["x"]
            geom["y"] += parent_geom["y"]
        abs_geometry_cache[cell_id] = geom
        return geom

    svg_root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(int(canvas_width)),
            "height": str(int(canvas_height)),
            "viewBox": f"0 0 {int(canvas_width)} {int(canvas_height)}",
            "data-figureweave-export": "drawio-svg",
        },
    )
    defs = ET.SubElement(svg_root, "defs")
    marker = ET.SubElement(
        defs,
        "marker",
        {
            "id": "fw-arrow-end",
            "viewBox": "0 0 10 10",
            "refX": "9",
            "refY": "5",
            "markerWidth": "8",
            "markerHeight": "8",
            "orient": "auto-start-reverse",
        },
    )
    ET.SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": "#1f1f1f"})

    vertex_cells = [cell for cell in cells.values() if cell.get("vertex") == "1" and cell.get("id") in geometries]
    edge_cells = [cell for cell in cells.values() if cell.get("edge") == "1"]

    def draw_text_group(parent: ET.Element, cell: ET.Element, geom: dict[str, float], style: dict[str, str]) -> None:
        lines = _html_lines(cell.get("value", ""))
        if not lines:
            return
        is_panel = style.get("container") == "1"
        is_text_only = style.get("shape") == "text"
        font_size = 18 if is_text_only else (22 if is_panel else 20)
        line_height = font_size * 1.25
        if is_panel:
            text_x = geom["x"] + geom["width"] / 2
            text_y = geom["y"] + 28
            anchor = "middle"
        elif is_text_only:
            text_x = geom["x"] + 4
            text_y = geom["y"] + font_size
            anchor = "start"
        else:
            text_x = geom["x"] + geom["width"] / 2
            total_h = line_height * max(len(lines) - 1, 0)
            text_y = geom["y"] + geom["height"] / 2 - total_h / 2
            anchor = "middle"
        text_elem = ET.SubElement(
            parent,
            "text",
            {
                "x": f"{text_x:.1f}",
                "y": f"{text_y:.1f}",
                "font-family": "Arial, Helvetica, sans-serif",
                "font-size": f"{font_size:.1f}",
                "text-anchor": anchor,
                "fill": "#222222",
            },
        )
        for idx, line in enumerate(lines):
            attrs = {"x": f"{text_x:.1f}"}
            if idx > 0:
                attrs["dy"] = f"{line_height:.1f}"
            tspan = ET.SubElement(text_elem, "tspan", attrs)
            tspan.text = line

    panel_first = sorted(
        vertex_cells,
        key=lambda cell: (
            0 if "container=1" in (cell.get("style") or "") else 1,
            effective_geometry(cell.get("id", ""))["y"],
            effective_geometry(cell.get("id", ""))["x"],
        ),
    )

    for cell in panel_first:
        cell_id = cell.get("id")
        if not cell_id:
            continue
        geom = effective_geometry(cell_id)
        style = _parse_mx_style(cell.get("style", ""))
        is_text_only = style.get("shape") == "text"
        is_port = style.get("fwPort") == "1"
        if is_port:
            continue
        group = ET.SubElement(svg_root, "g", {"id": cell_id, "class": "fw-drawio-cell"})
        if not is_text_only:
            rect_attrs = {
                "x": f"{geom['x']:.1f}",
                "y": f"{geom['y']:.1f}",
                "width": f"{geom['width']:.1f}",
                "height": f"{geom['height']:.1f}",
                "fill": style.get("fillColor", "#ffffff"),
                "stroke": style.get("strokeColor", "#000000"),
                "stroke-width": style.get("strokeWidth", "2"),
            }
            if style.get("rounded") == "1":
                rect_attrs["rx"] = "12"
                rect_attrs["ry"] = "12"
            if style.get("dashed") == "1":
                rect_attrs["stroke-dasharray"] = "12 8"
            ET.SubElement(group, "rect", rect_attrs)
        draw_text_group(group, cell, geom, style)

    for cell in edge_cells:
        source = cell.get("source")
        target = cell.get("target")
        if not source or not target or source not in geometries or target not in geometries:
            continue
        sg = effective_geometry(source)
        tg = effective_geometry(target)
        x1 = sg["x"] + sg["width"] / 2
        y1 = sg["y"] + sg["height"] / 2
        x2 = tg["x"] + tg["width"] / 2
        y2 = tg["y"] + tg["height"] / 2
        mid_x = (x1 + x2) / 2
        path_d = f"M {x1:.1f} {y1:.1f} L {mid_x:.1f} {y1:.1f} L {mid_x:.1f} {y2:.1f} L {x2:.1f} {y2:.1f}"
        ET.SubElement(
            svg_root,
            "path",
            {
                "id": cell.get("id", ""),
                "d": path_d,
                "fill": "none",
                "stroke": "#1f1f1f",
                "stroke-width": "3",
                "marker-end": "url(#fw-arrow-end)",
            },
        )

    svg_code = ET.tostring(svg_root, encoding="unicode")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(svg_code, encoding="utf-8")
    return svg_code


def export_drawio_from_svg(
    svg_path: str,
    output_path: str,
    *,
    page_name: str = "FigureWeave",
    figure_path: Optional[str] = None,
    scene_graph_output_path: Optional[str] = None,
) -> str:
    scene_graph = extract_scene_graph_from_svg(
        svg_path,
        figure_path=figure_path,
        output_path=scene_graph_output_path,
    )
    return export_drawio_from_scene_graph(
        scene_graph,
        output_path,
        page_name=page_name,
    )
