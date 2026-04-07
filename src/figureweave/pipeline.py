from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from .config import (
    DEFAULT_SAM_PROMPT,
    GEMINI_DEFAULT_IMAGE_SIZE,
    PROVIDER_CONFIGS,
    PlaceholderMode,
    ProviderType,
)
from .svg_ops import (
    calculate_scale_factors,
    check_and_fix_svg,
    generate_svg_template,
    optimize_svg_with_llm,
    replace_icons_in_svg,
)
from .vision import (
    _ensure_rmbg2_access_ready,
    crop_and_remove_background,
    generate_figure_from_method,
    segment_with_sam3,
)

def method_to_svg(
    method_text: str,
    output_dir: str = "./output",
    api_key: str = None,
    base_url: str = None,
    provider: ProviderType = "bianxie",
    image_provider: Optional[ProviderType] = None,
    image_api_key: Optional[str] = None,
    image_base_url: Optional[str] = None,
    svg_provider: Optional[ProviderType] = None,
    svg_api_key: Optional[str] = None,
    svg_base_url: Optional[str] = None,
    image_gen_model: str = None,
    svg_gen_model: str = None,
    sam_prompts: str = DEFAULT_SAM_PROMPT,
    min_score: float = 0.5,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
    rmbg_model_path: Optional[str] = None,
    stop_after: int = 5,
    placeholder_mode: PlaceholderMode = "label",
    optimize_iterations: int = 2,
    merge_threshold: float = 0.9,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
    figure_caption: Optional[str] = None,
) -> dict:
    """
    完整流程：Paper Method → SVG with Icons

    Args:
        method_text: Paper method 文本内容
        output_dir: 输出目录
        api_key: API Key
        base_url: API base URL
        provider: API 提供商
        image_gen_model: 生图模型
        svg_gen_model: SVG 生成模型
        sam_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: SAM3 最低置信度
        sam_backend: SAM3 后端（local/fal/roboflow/api）
        sam_api_key: SAM3 API Key（api 模式使用）
        sam_max_masks: SAM3 API 最大 masks 数（api 模式使用）
        rmbg_model_path: RMBG 模型路径
        stop_after: 执行到指定步骤后停止
        placeholder_mode: 占位符模式
            - "none": 无特殊样式
            - "box": 传入 boxlib 坐标
            - "label": 灰色填充+黑色边框+序号标签（推荐）
        optimize_iterations: 步骤 4.6 优化迭代次数（0 表示跳过优化）
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）

    Returns:
        结果字典
    """
    image_provider = image_provider or provider
    svg_provider = svg_provider or provider
    image_api_key = image_api_key or api_key
    svg_api_key = svg_api_key or api_key

    image_config = PROVIDER_CONFIGS[image_provider]
    svg_config = PROVIDER_CONFIGS[svg_provider]
    if image_base_url is None:
        image_base_url = base_url if image_provider == provider and base_url else image_config["base_url"]
    if svg_base_url is None:
        svg_base_url = base_url if svg_provider == provider and base_url else svg_config["base_url"]
    if image_gen_model is None:
        image_gen_model = image_config["default_image_model"]
    if svg_gen_model is None:
        svg_gen_model = svg_config["default_svg_model"]

    if not image_api_key:
        raise ValueError("必须提供 image_api_key（或 api_key）")
    if not svg_api_key:
        raise ValueError("必须提供 svg_api_key（或 api_key）")
    if image_provider == "anthropic":
        raise ValueError("Anthropic Claude 不能用于生图阶段，请选择 Gemini 或 OpenAI。")
    if not image_gen_model:
        raise ValueError(f"image_provider={image_provider} 没有可用的生图模型")
    if not svg_gen_model:
        raise ValueError(f"svg_provider={svg_provider} 没有可用的重建模型")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Paper Method 到 SVG 图标替换流程 (Label 模式增强版 + Box合并)")
    print("=" * 60)
    print(f"Image Provider: {image_provider}")
    print(f"SVG Provider: {svg_provider}")
    print(f"输出目录: {output_dir}")
    print(f"生图模型: {image_gen_model}")
    print(f"SVG模型: {svg_gen_model}")
    print(f"SAM提示词: {sam_prompts}")
    print(f"最低置信度: {min_score}")
    sam_backend_value = "fal" if sam_backend == "api" else sam_backend
    print(f"SAM后端: {sam_backend_value}")
    if sam_backend_value == "fal":
        print(f"SAM3 API max_masks: {sam_max_masks}")
    print(f"执行到步骤: {stop_after}")
    print(f"占位符模式: {placeholder_mode}")
    print(f"优化迭代次数: {optimize_iterations}")
    print(f"Box合并阈值: {merge_threshold}")
    if figure_caption:
        print(f"Figure Caption: {figure_caption}")
    if image_provider == "gemini":
        print(f"生图分辨率: {image_size}")
    print("=" * 60)

    # 步骤一：生成图片
    figure_path = output_dir / "figure.png"
    generate_figure_from_method(
        method_text=method_text,
        output_path=str(figure_path),
        api_key=image_api_key,
        model=image_gen_model,
        base_url=image_base_url,
        provider=image_provider,
        figure_caption=figure_caption,
        image_size=image_size,
    )

    if stop_after == 1:
        print("\n" + "=" * 60)
        print("已在步骤 1 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": None,
            "boxlib_path": None,
            "icon_infos": [],
            "placeholder_count": 0,
            "no_icon_mode": True,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤二：SAM3 分割（包含Box合并）
    samed_path, boxlib_path, valid_boxes = segment_with_sam3(
        image_path=str(figure_path),
        output_dir=str(output_dir),
        text_prompts=sam_prompts,
        min_score=min_score,
        merge_threshold=merge_threshold,
        sam_backend=sam_backend_value,
        sam_api_key=sam_api_key,
        sam_max_masks=sam_max_masks,
    )

    no_icon_mode = len(valid_boxes) == 0
    if no_icon_mode:
        print("\n警告: 没有检测到有效的图标，切换到纯 SVG 回退模式")
    else:
        print(f"\n检测到 {len(valid_boxes)} 个图标")

    if stop_after == 2:
        print("\n" + "=" * 60)
        print("已在步骤 2 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": [],
            "placeholder_count": len(valid_boxes),
            "no_icon_mode": no_icon_mode,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤三：裁切 + 去背景
    icon_infos = []
    if no_icon_mode:
        print("步骤三跳过：当前为无图标回退模式")
    else:
        _ensure_rmbg2_access_ready(rmbg_model_path)
        icon_infos = crop_and_remove_background(
            image_path=str(figure_path),
            boxlib_path=boxlib_path,
            output_dir=str(output_dir),
            rmbg_model_path=rmbg_model_path,
        )

    if stop_after == 3:
        print("\n" + "=" * 60)
        print("已在步骤 3 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "placeholder_count": len(valid_boxes),
            "no_icon_mode": no_icon_mode,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤四：生成 SVG 模板
    template_svg_path = output_dir / "template.svg"
    optimized_template_path = output_dir / "optimized_template.svg"
    final_svg_path = output_dir / "final.svg"
    try:
        generate_svg_template(
            figure_path=str(figure_path),
            samed_path=samed_path,
            boxlib_path=boxlib_path,
            output_path=str(template_svg_path),
            api_key=svg_api_key,
            model=svg_gen_model,
            base_url=svg_base_url,
            provider=svg_provider,
            figure_caption=figure_caption,
            placeholder_mode=placeholder_mode,
            no_icon_mode=no_icon_mode,
        )

        # 步骤 4.6：LLM 优化 SVG 模板（可配置迭代次数，0 表示跳过）
        optimize_svg_with_llm(
            figure_path=str(figure_path),
            samed_path=samed_path,
            final_svg_path=str(template_svg_path),
            output_path=str(optimized_template_path),
            api_key=svg_api_key,
            model=svg_gen_model,
            base_url=svg_base_url,
            provider=svg_provider,
            max_iterations=optimize_iterations,
            skip_base64_validation=True,
            no_icon_mode=no_icon_mode,
        )
    except Exception as exc:
        if not no_icon_mode:
            raise
        print(f"无图标模式下 SVG 重建失败（{exc}），改用内嵌原图的保底 SVG")
        create_embedded_figure_svg(
            figure_path=str(figure_path),
            output_path=str(final_svg_path),
        )

    if stop_after == 4:
        print("\n" + "=" * 60)
        print("已在步骤 4 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "placeholder_count": len(valid_boxes),
            "no_icon_mode": no_icon_mode,
            "template_svg_path": str(template_svg_path) if template_svg_path.is_file() else None,
            "optimized_template_path": str(optimized_template_path) if optimized_template_path.is_file() else None,
            "final_svg_path": None,
        }

    svg_template_for_replace = optimized_template_path if optimized_template_path.is_file() else template_svg_path

    # 步骤五：图标替换
    if no_icon_mode:
        if svg_template_for_replace.is_file():
            shutil.copyfile(svg_template_for_replace, final_svg_path)
            print("无图标模式：跳过图标替换，直接输出 SVG")
        else:
            print("无图标模式缺少模板 SVG，生成保底 final.svg")
            create_embedded_figure_svg(
                figure_path=str(figure_path),
                output_path=str(final_svg_path),
            )
    else:
        # 步骤 4.7：坐标系对齐
        print("\n" + "-" * 50)
        print("步骤 4.7：坐标系对齐")
        print("-" * 50)

        figure_img = Image.open(figure_path)
        figure_width, figure_height = figure_img.size
        print(f"原图尺寸: {figure_width} x {figure_height}")

        with open(svg_template_for_replace, 'r', encoding='utf-8') as f:
            svg_code = f.read()

        svg_width, svg_height = get_svg_dimensions(svg_code)

        if svg_width and svg_height:
            print(f"SVG 尺寸: {svg_width} x {svg_height}")

            if abs(svg_width - figure_width) < 1 and abs(svg_height - figure_height) < 1:
                print("尺寸匹配，使用 1:1 坐标映射")
                scale_factors = (1.0, 1.0)
            else:
                scale_x, scale_y = calculate_scale_factors(
                    figure_width, figure_height, svg_width, svg_height
                )
                scale_factors = (scale_x, scale_y)
                print(f"尺寸不匹配，计算缩放因子: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
        else:
            print("警告: 无法提取 SVG 尺寸，使用 1:1 坐标映射")
            scale_factors = (1.0, 1.0)

        replace_icons_in_svg(
            template_svg_path=str(svg_template_for_replace),
            icon_infos=icon_infos,
            output_path=str(final_svg_path),
            scale_factors=scale_factors,
            match_by_label=(placeholder_mode == "label"),
        )

    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)
    print(f"原始图片: {figure_path}")
    print(f"标记图片: {samed_path}")
    print(f"Box信息: {boxlib_path}")
    print(f"图标数量: {len(icon_infos)}")
    print(f"SVG模板: {template_svg_path}")
    print(f"优化后模板: {optimized_template_path}")
    print(f"最终SVG: {final_svg_path}")

    return {
        "figure_path": str(figure_path),
        "samed_path": samed_path,
        "boxlib_path": boxlib_path,
        "icon_infos": icon_infos,
        "placeholder_count": len(valid_boxes),
        "no_icon_mode": no_icon_mode,
        "template_svg_path": str(template_svg_path) if template_svg_path.is_file() else None,
        "optimized_template_path": str(optimized_template_path) if optimized_template_path.is_file() else None,
        "final_svg_path": str(final_svg_path),
    }


def create_embedded_figure_svg(
    figure_path: str,
    output_path: str,
) -> str:
    """Wrap the generated raster figure in a minimal SVG as a final fallback."""
    figure_img = Image.open(figure_path)
    width, height = figure_img.size
    buf = io.BytesIO()
    figure_img.save(buf, format="PNG")
    figure_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    svg_code = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        f'href="data:image/png;base64,{figure_b64}" preserveAspectRatio="none"/>\n'
        f"</svg>\n"
    )

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"内嵌 figure.png 的保底 SVG 已保存: {output_path_obj}")
    return str(output_path_obj)


def _copy_selected_candidate_outputs(candidate_dir: Path, output_dir: Path) -> None:
    """Promote one chosen candidate to the root directory for compatibility with the existing UI."""
    files_to_copy = [
        "figure.png",
        "samed.png",
        "boxlib.json",
        "template.svg",
        "optimized_template.svg",
        "final.svg",
    ]
    for filename in files_to_copy:
        src = candidate_dir / filename
        dst = output_dir / filename
        if src.is_file():
            shutil.copyfile(src, dst)

    src_icons = candidate_dir / "icons"
    dst_icons = output_dir / "icons"
    if src_icons.is_dir():
        if dst_icons.exists():
            shutil.rmtree(dst_icons)
        shutil.copytree(src_icons, dst_icons)


def _candidate_selection_key(summary: dict) -> tuple:
    """Prefer true SVG reconstructions with fewer raster placeholders and fewer extracted icons."""
    has_template_svg = bool(summary.get("has_template_svg"))
    return (
        0 if has_template_svg else 1,
        int(summary.get("icon_count", 0)),
        int(summary.get("placeholder_count", 0)),
        summary.get("candidate_id", ""),
    )


def method_to_svg_candidates(
    method_text: str,
    output_dir: str = "./output",
    num_candidates: int = 1,
    **kwargs,
) -> dict:
    """
    Generate multiple full candidates and keep each run in its own subdirectory.
    The selected candidate is copied to the root output directory as the default result.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    num_candidates = max(1, int(num_candidates))

    if num_candidates == 1:
        selected_result = method_to_svg(
            method_text=method_text,
            output_dir=str(output_dir_path),
            **kwargs,
        )
        manifest = {
            "num_candidates": 1,
            "selected_candidate": "candidate_01",
            "selection_rule": "single_candidate",
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "status": "ok",
                    "icon_count": len(selected_result.get("icon_infos", [])),
                    "placeholder_count": int(selected_result.get("placeholder_count", 0)),
                    "no_icon_mode": bool(selected_result.get("no_icon_mode", False)),
                    "has_template_svg": bool(
                        selected_result.get("template_svg_path")
                        or selected_result.get("optimized_template_path")
                    ),
                    "final_svg_path": selected_result.get("final_svg_path"),
                }
            ],
        }
        manifest_path = output_dir_path / "candidates_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return {
            "selected_candidate": "candidate_01",
            "manifest_path": str(manifest_path),
            "candidates": manifest["candidates"],
            "selected_result": selected_result,
        }

    print("\n" + "=" * 60)
    print(f"多候选生成模式：共 {num_candidates} 个候选")
    print("=" * 60)

    candidates: list[dict] = []
    successful: list[tuple[dict, dict, Path]] = []

    for idx in range(num_candidates):
        candidate_id = f"candidate_{idx + 1:02d}"
        candidate_dir = output_dir_path / candidate_id
        print("\n" + "#" * 60)
        print(f"开始生成 {candidate_id} ({idx + 1}/{num_candidates})")
        print("#" * 60)
        try:
            result = method_to_svg(
                method_text=method_text,
                output_dir=str(candidate_dir),
                **kwargs,
            )
            summary = {
                "candidate_id": candidate_id,
                "status": "ok",
                "icon_count": len(result.get("icon_infos", [])),
                "placeholder_count": int(result.get("placeholder_count", 0)),
                "no_icon_mode": bool(result.get("no_icon_mode", False)),
                "has_template_svg": bool(
                    result.get("template_svg_path") or result.get("optimized_template_path")
                ),
                "final_svg_path": result.get("final_svg_path"),
                "figure_path": result.get("figure_path"),
            }
            candidates.append(summary)
            successful.append((summary, result, candidate_dir))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"{candidate_id} 失败: {error_text}")
            candidate_dir.mkdir(parents=True, exist_ok=True)
            with open(candidate_dir / "candidate_error.log", "w", encoding="utf-8") as f:
                f.write(error_text + "\n")
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "status": "error",
                    "error": error_text,
                }
            )

    if not successful:
        raise RuntimeError("所有候选都生成失败")

    selected_summary, selected_result, selected_dir = min(
        successful,
        key=lambda item: _candidate_selection_key(item[0]),
    )
    _copy_selected_candidate_outputs(selected_dir, output_dir_path)

    manifest = {
        "num_candidates": num_candidates,
        "selected_candidate": selected_summary["candidate_id"],
        "selection_rule": "prefer_true_svg_then_min_icon_count_then_min_placeholder_count_then_earliest_id",
        "candidates": candidates,
    }
    manifest_path = output_dir_path / "candidates_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"已选择默认候选: {selected_summary['candidate_id']}")
    print(f"候选清单: {manifest_path}")
    print("=" * 60)

    return {
        "selected_candidate": selected_summary["candidate_id"],
        "manifest_path": str(manifest_path),
        "candidates": candidates,
        "selected_result": selected_result,
    }


# ============================================================================
# 命令行入口
# ============================================================================

