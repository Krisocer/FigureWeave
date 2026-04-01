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
GEMINI_DEFAULT_IMAGE_SIZE = '2K'
IMAGE_SIZE_CHOICES = ('1K', '2K', '4K')
BOXLIB_NO_ICON_MODE_KEY = 'no_icon_mode'
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
- Use a paper-figure style with thin borders, readable labels, and restrained colors."""
