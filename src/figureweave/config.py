from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env')

PROVIDER_CONFIGS = {
    'openrouter': {
        'base_url': 'https://openrouter.ai/api/v1',
        'default_image_model': 'google/gemini-3-pro-image-preview',
        'default_svg_model': 'google/gemini-3.1-pro-preview',
    },
    'bianxie': {
        'base_url': 'https://api.bianxie.ai/v1',
        'default_image_model': 'gemini-3-pro-image-preview',
        'default_svg_model': 'gemini-3.1-pro-preview',
    },
    'gemini': {
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'default_image_model': 'gemini-3.1-flash-image-preview',
        'default_svg_model': 'gemini-3.1-pro-preview',
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'default_image_model': 'gpt-4.1',
        'default_svg_model': 'gpt-4.1',
    },
    'anthropic': {
        'base_url': 'https://api.anthropic.com/v1/messages',
        'default_image_model': None,
        'default_svg_model': 'claude-sonnet-4-20250514',
    },
}

ProviderType = Literal['openrouter', 'bianxie', 'gemini', 'openai', 'anthropic']
PlaceholderMode = Literal['none', 'box', 'label']
FigureMode = Literal['simple_flowchart', 'complex_paper']
GEMINI_DEFAULT_IMAGE_SIZE = '2K'
IMAGE_SIZE_CHOICES = ('1K', '2K', '4K')
BOXLIB_NO_ICON_MODE_KEY = 'no_icon_mode'
DEFAULT_SAM_PROMPT = 'plot,chart,heatmap,matrix,image'
COMPLEX_PAPER_SAM_PROMPT = 'module,block,encoder,head,panel,plot,heatmap,matrix'
GEMINI_IMAGE_MAX_RETRIES = max(1, int(os.environ.get('GEMINI_IMAGE_MAX_RETRIES', '4')))
GEMINI_IMAGE_RETRY_BASE_DELAY = float(os.environ.get('GEMINI_IMAGE_RETRY_BASE_DELAY', '15'))
SVG_MAX_PLACEHOLDERS = max(1, int(os.environ.get('FIGUREWEAVE_MAX_PLACEHOLDERS', '10')))
SVG_MIN_BOX_AREA_RATIO = float(os.environ.get('FIGUREWEAVE_MIN_BOX_AREA_RATIO', '0.008'))
LOCAL_OPEN_VOCAB_DETECTOR_MODEL = os.environ.get(
    'AUTOFIGURE_LOCAL_DETECTOR_MODEL',
    'IDEA-Research/grounding-dino-base',
)
LOCAL_DETECTOR_MIN_SCORE = float(os.environ.get('AUTOFIGURE_LOCAL_DETECTOR_THRESHOLD', '0.2'))
LOCAL_DETECTOR_MAX_BOX_AREA_RATIO = float(
    os.environ.get('AUTOFIGURE_LOCAL_MAX_BOX_AREA_RATIO', '0.2')
)

SAM3_FAL_API_URL = 'https://fal.run/fal-ai/sam-3/image'
SAM3_ROBOFLOW_API_URL = os.environ.get(
    'ROBOFLOW_API_URL',
    'https://serverless.roboflow.com/sam3/concept_segment',
)
SAM3_API_TIMEOUT = 300

FLOWCHART_STYLE_PROMPT = """Render the method as a clean academic machine learning flowchart instead of a decorative illustration.

Hard constraints:
- Prefer at most three major stages.
- Use a left-to-right or top-to-bottom pipeline layout.
- Use simple rectangular modules, token strips, feature blocks, arrows, braces, and captions.
- Keep the composition sparse and easy to reconstruct as SVG.
- Avoid human characters, mascots, faces, hands, animals, or photorealistic scenes.
- Avoid unnecessary decorative icons, textured backgrounds, and visual clutter.
- Use a paper-figure style with thin borders, readable labels, and restrained colors.
- Prefer abstract blocks, plots, matrices, and heatmaps over pictorial stickers or illustrative avatars.
- Only use image crops or raster insets when they are essential to explain the method."""

PAPER_FLOWCHART_PROMPT_TEMPLATE = """Generate a clean academic paper figure for the method below.

{flowchart_style_prompt}

Additional layout guidance:
- Keep the figure compact and modular.
- Prefer three major stages or fewer.
- Show only the essential modules, arrows, plots, and labels needed to explain the pipeline.
- Favor module-level structure over small decorative subcomponents.
- Prefer abstract scientific visuals such as blocks, matrices, distributions, token bars, and plots.
- Do not add humans, robots, mascots, animals, hands, or decorative stickers.
- Do not insert extra example photos unless the method explicitly depends on image crops as an input modality.
- If the source text is long, compress it into the minimal set of modules needed to explain the workflow.

Method text:
{method_text}
{figure_caption_block}"""

COMPLEX_PAPER_PROMPT_TEMPLATE = """Generate a publication-style machine learning paper figure for the method below.

{flowchart_style_prompt}

Additional layout guidance for complex paper figures:
- Preserve the full scientific structure when it is essential, including nested subpanels, auxiliary branches, loss terms, legends, and small statistical plots.
- Keep the figure organized into clearly separated regions or stages, but do not over-compress if the method genuinely requires multiple submodules.
- Prioritize fidelity to module names, formulas, arrows, brackets, heatmaps, token strips, bar charts, and panel boundaries.
- Use clean paper-style visuals rather than illustrative icons.
- Prefer abstract diagrams, matrices, distributions, and plots over decorative imagery.
- If the method includes image inputs, keep them as small task-relevant insets only when necessary.

Method text:
{method_text}
{figure_caption_block}"""
