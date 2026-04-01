<div align="center">

<img src="web/favicon.svg" alt="FigureWeave Logo" width="120"/>

# FigureWeave

**From method text to editable scientific figures**

[English](README.md) | [中文](README_ZH.md)

</div>

---

## Overview

FigureWeave is a research-engineering project for turning paper method descriptions into publication-style figures that remain editable as SVG.

This project is **inspired by AutoFigure**, but it is no longer a mirror of the original system. The current codebase has been reworked into a more practical figure authoring pipeline with local GPU segmentation, dual-provider model routing, candidate generation, and a redesigned interactive UI.

FigureWeave is especially useful for:

- method overviews
- pipeline diagrams
- system schematics
- architecture figures
- editable draft figures for papers, slides, and reports

It is **not** intended to replace precise plotting tools such as matplotlib, seaborn, ggplot, or Origin for charts driven by exact numeric data.

---

## What Is New In FigureWeave

Compared with the original AutoFigure-style workflow, this project adds several concrete contributions:

1. **Editable-first workflow**
   The pipeline is centered on producing SVG outputs that can be inspected and refined inside the browser.

2. **Local SAM3 on GPU**
   Segmentation can now run locally on GPU instead of relying only on hosted APIs, which improves speed, control, and privacy.

3. **Dual-provider design**
   The pipeline separates:
   - image drafting
   - SVG reasoning and reconstruction

   This allows practical combinations such as:
   - `Gemini -> Gemini`
   - `OpenAI -> OpenAI`
   - `Gemini -> Anthropic Claude`
   - `OpenAI -> Anthropic Claude`

4. **Multi-candidate generation**
   A single request can produce multiple end-to-end candidates, keep each candidate artifact set, and promote the selected result.

5. **Figure caption support**
   In addition to method text, the system accepts a figure caption / intent field to better constrain layout and narrative structure.

6. **GPU-accelerated local post-processing**
   Background removal and local segmentation can run with CUDA-enabled PyTorch.

7. **Redesigned UI**
   The web interface has been rebuilt into a cleaner studio-style workflow with:
   - separate provider controls
   - candidate controls
   - artifact review
   - canvas-centered editing

---

## Pipeline

FigureWeave currently runs in five major stages:

1. **Image Draft**
   Generate a scientific-style draft figure from method text, optional figure caption, and optional reference image.

2. **Segmentation**
   Run local SAM3 or an API backend to detect icons and visual regions, producing:
   - `samed.png`
   - `boxlib.json`

3. **Asset Extraction**
   Crop detected regions and remove backgrounds to create transparent assets.

4. **SVG Reasoning And Reconstruction**
   Use a multimodal model to reconstruct the draft into an editable SVG template, then optionally refine it.

5. **Assembly**
   Replace placeholders with extracted assets and emit:
   - `template.svg`
   - `optimized_template.svg`
   - `final.svg`

---

## Supported Model Routing

### Image draft provider

- `Gemini`
- `OpenAI`

### SVG reasoning and reconstruction provider

- `Gemini`
- `OpenAI`
- `Anthropic Claude`

### Practical note

Anthropic Claude is used here for **understanding and reconstruction**, not for native image generation. In this project, the image drafting stage should use Gemini or OpenAI.

---

## Web Interface

Start the server:

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The main configuration page now includes:

- `Method Text`
- `Figure Caption`
- `Image Draft Provider`
- `SVG Reasoning Provider`
- `Candidates`
- `Generation Mode`
- `SAM3 Backend`
- `Reference Image`

The canvas page lets you:

- inspect intermediate artifacts
- switch between candidate SVGs
- review logs
- open the result in the embedded SVG editor

---

## CLI Usage

### Basic

```bash
python figureweave.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --image_provider gemini \
  --image_api_key YOUR_GEMINI_KEY \
  --svg_provider anthropic \
  --svg_api_key YOUR_ANTHROPIC_KEY
```

### Single-provider fallback

If you want to use one provider for both stages, you can still use:

```bash
python figureweave.py \
  --method_file paper.txt \
  --output_dir outputs/demo \
  --provider gemini \
  --api_key YOUR_GEMINI_KEY
```

### Multi-candidate generation

```bash
python figureweave.py \
  --method_file paper.txt \
  --output_dir outputs/demo_multi \
  --image_provider gemini \
  --image_api_key YOUR_GEMINI_KEY \
  --svg_provider openai \
  --svg_api_key YOUR_OPENAI_KEY \
  --num_candidates 3
```

---

## Local SAM3

FigureWeave supports local SAM3 execution on GPU.

Typical setup:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

You also need:

- a supported NVIDIA GPU
- CUDA-capable PyTorch in this environment
- Hugging Face access to gated SAM3 weights

If local SAM3 is unavailable, the codebase can still fall back to other segmentation paths depending on configuration.

---

## Installation

### Python environment

```bash
pip install -r requirements.txt
```

### Environment variables

At minimum, you will usually want:

```env
HF_TOKEN=your_huggingface_token
ROBOFLOW_API_KEY=your_roboflow_key
```

Depending on your selected routing, you may also need:

- Gemini API key
- OpenAI API key
- Anthropic API key

---

## Docker

Build and run:

```bash
docker compose up -d --build
```

Check health:

```bash
docker compose ps
curl http://127.0.0.1:8000/healthz
```

Logs:

```bash
docker compose logs -f figureweave
```

Restart:

```bash
docker compose restart figureweave
```

---

## Output Structure

Typical outputs include:

- `figure.png`
- `samed.png`
- `boxlib.json`
- `icons/`
- `template.svg`
- `optimized_template.svg`
- `final.svg`
- `candidates_manifest.json`

When multi-candidate mode is enabled, each run is stored under:

- `candidate_01/`
- `candidate_02/`
- `candidate_03/`

---

## Showcase

The following assets are now used as the current project showcase example from the `multimodal_medical_report` run:

1. Draft image: [`img/case/multimodal_medical_report_draft.png`](img/case/multimodal_medical_report_draft.png)
2. Optimized SVG template: [`img/case/multimodal_medical_report_template.svg`](img/case/multimodal_medical_report_template.svg)
3. Final assembled SVG: [`img/case/multimodal_medical_report_final.svg`](img/case/multimodal_medical_report_final.svg)

<p align="center">
  <img src="img/case/multimodal_medical_report_draft.png" alt="FigureWeave showcase draft" width="32%"/>
  <img src="img/case/multimodal_medical_report_template.svg" alt="FigureWeave showcase optimized template" width="32%"/>
  <img src="img/case/multimodal_medical_report_final.svg" alt="FigureWeave showcase final svg" width="32%"/>
</p>

This trio highlights the intended FigureWeave workflow:

- `figure.png` as the model-generated draft
- `optimized_template.svg` as the editable structural reconstruction
- `final.svg` as the assembled showcase result

---

## Credits

FigureWeave is **inspired by AutoFigure** and builds on the broader idea of converting scientific method descriptions into figure drafts.

The current project extends that direction with a more editing-oriented and deployment-oriented system, especially in:

- local GPU segmentation
- dual-provider routing
- multi-candidate generation
- browser-centered SVG refinement
- UI and workflow redesign

---

## License

This repository currently keeps the existing project license in [`LICENSE`](LICENSE).

