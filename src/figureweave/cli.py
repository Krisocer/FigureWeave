from __future__ import annotations

import argparse
from pathlib import Path

from .config import COMPLEX_PAPER_SAM_PROMPT, DEFAULT_SAM_PROMPT, GEMINI_DEFAULT_IMAGE_SIZE, IMAGE_SIZE_CHOICES
from . import vision
from .pipeline import method_to_svg_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Paper Method ? SVG ?????? (Label ????? + Box??)'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--method_text', help='Paper method ????')
    input_group.add_argument('--method_file', default='./paper.txt', help='?? paper method ???????')

    parser.add_argument('--output_dir', default='./output', help='???????: ./output?')
    parser.add_argument(
        '--provider',
        choices=['openrouter', 'bianxie', 'gemini', 'openai'],
        default='gemini',
        help='API ??????: gemini?',
    )
    parser.add_argument('--api_key', default=None, help='API Key')
    parser.add_argument('--base_url', default=None, help='API base URL????? provider ?????')
    parser.add_argument(
        '--image_provider',
        choices=['openrouter', 'bianxie', 'gemini', 'openai'],
        default=None,
        help='???? provider????????? --provider?',
    )
    parser.add_argument('--image_api_key', default=None, help='???? API Key????????? --api_key?')
    parser.add_argument('--image_base_url', default=None, help='???? API base URL????')
    parser.add_argument(
        '--svg_provider',
        choices=['openrouter', 'bianxie', 'gemini', 'openai', 'anthropic'],
        default=None,
        help='SVG ??????? provider????????? --provider?',
    )
    parser.add_argument('--svg_api_key', default=None, help='SVG ??????? API Key????????? --api_key?')
    parser.add_argument('--svg_base_url', default=None, help='SVG ??????? API base URL????')

    parser.add_argument('--image_model', default=None, help='????????? provider ?????')
    parser.add_argument(
        '--image_size',
        choices=list(IMAGE_SIZE_CHOICES),
        default=GEMINI_DEFAULT_IMAGE_SIZE,
        help='????????: 1K/2K/4K???: 2K?',
    )
    parser.add_argument('--svg_model', default=None, help='SVG????????? provider ?????')

    parser.add_argument(
        '--use_reference_image',
        action='store_true',
        help='?????????????????? --reference_image_path?',
    )
    parser.add_argument('--reference_image_path', default=None, help='??????????')
    parser.add_argument('--figure_caption', default=None, help='Figure caption / figure brief????')
    parser.add_argument(
        '--figure_mode',
        choices=['simple_flowchart', 'complex_paper'],
        default='simple_flowchart',
        help='Figure generation mode: simple_flowchart or complex_paper',
    )

    parser.add_argument('--sam_prompt', default=None, help=f"SAM3 prompt. Defaults to {DEFAULT_SAM_PROMPT} for simple_flowchart and {COMPLEX_PAPER_SAM_PROMPT} for complex_paper.")
    parser.add_argument('--min_score', type=float, default=0.0, help='SAM3 ??????????: 0.0?')
    parser.add_argument(
        '--sam_backend',
        choices=['local', 'fal', 'roboflow', 'api'],
        default='local',
        help='SAM3 ???local(????)/fal(fal.ai)/roboflow(Roboflow)/api(???=fal)',
    )
    parser.add_argument('--sam_api_key', default=None, help='SAM3 API Key????? FAL_KEY?')
    parser.add_argument(
        '--sam_max_masks',
        type=int,
        default=32,
        help='SAM3 API ?? masks ??? api ?????: 32?',
    )
    parser.add_argument('--rmbg_model_path', default=None, help='RMBG ??????????')
    parser.add_argument(
        '--stop_after',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help='???????????1-5???: 5 ?????',
    )
    parser.add_argument(
        '--placeholder_mode',
        choices=['none', 'box', 'label'],
        default='label',
        help='??????none(???)/box(???)/label(????)???: label?',
    )
    parser.add_argument(
        '--optimize_iterations',
        type=int,
        default=0,
        help='?? 4.6 LLM ???????0 ?????????: 0?',
    )
    parser.add_argument(
        '--merge_threshold',
        type=float,
        default=0.001,
        help='Box?????????????????0????????: 0.9?',
    )
    parser.add_argument('--num_candidates', type=int, default=1, help='?????????: 1?')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.use_reference_image and not args.reference_image_path:
        parser.error('--use_reference_image ?? --reference_image_path')
    if args.reference_image_path and not Path(args.reference_image_path).is_file():
        parser.error(f'???????: {args.reference_image_path}')

    vision.USE_REFERENCE_IMAGE = bool(args.use_reference_image)
    vision.REFERENCE_IMAGE_PATH = args.reference_image_path
    if vision.REFERENCE_IMAGE_PATH:
        vision.USE_REFERENCE_IMAGE = True

    method_text = args.method_text
    if method_text is None:
        with open(args.method_file, 'r', encoding='utf-8') as f:
            method_text = f.read()

    method_to_svg_candidates(
        method_text=method_text,
        output_dir=args.output_dir,
        num_candidates=args.num_candidates,
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
        image_provider=args.image_provider,
        image_api_key=args.image_api_key,
        image_base_url=args.image_base_url,
        svg_provider=args.svg_provider,
        svg_api_key=args.svg_api_key,
        svg_base_url=args.svg_base_url,
        image_gen_model=args.image_model,
        image_size=args.image_size,
        svg_gen_model=args.svg_model,
        sam_prompts=args.sam_prompt or (COMPLEX_PAPER_SAM_PROMPT if args.figure_mode == 'complex_paper' else DEFAULT_SAM_PROMPT),
        figure_mode=args.figure_mode,
        min_score=args.min_score,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        sam_max_masks=args.sam_max_masks,
        rmbg_model_path=args.rmbg_model_path,
        stop_after=args.stop_after,
        placeholder_mode=args.placeholder_mode,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
        figure_caption=args.figure_caption,
    )
