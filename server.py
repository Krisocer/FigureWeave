from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXECUTABLE = os.environ.get("AUTOFIGURE_PYTHON") or sys.executable

DEFAULT_SAM_PROMPT = "plot,chart,heatmap,matrix,image"
COMPLEX_PAPER_SAM_PROMPT = "module,block,encoder,head,panel,plot,heatmap,matrix"
DEFAULT_PLACEHOLDER_MODE = "label"
DEFAULT_MERGE_THRESHOLD = 0.01

SVG_EDIT_CANDIDATES = [
    ("vendor/svg-edit/editor/index.html", WEB_DIR / "vendor" / "svg-edit" / "editor" / "index.html"),
    ("vendor/svg-edit/editor.html", WEB_DIR / "vendor" / "svg-edit" / "editor.html"),
    ("vendor/svg-edit/index.html", WEB_DIR / "vendor" / "svg-edit" / "index.html"),
]

SENSITIVE_CMD_FLAGS = {
    "--api_key",
    "--sam_api_key",
    "--image_api_key",
    "--svg_api_key",
}


def _resolve_svg_edit_path() -> tuple[bool, str | None]:
    for rel, path in SVG_EDIT_CANDIDATES:
        if path.is_file():
            return True, f"/{rel}"
    return False, None


def _redact_cmd_args(cmd: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for token in cmd:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        redacted.append(token)
        if token in SENSITIVE_CMD_FLAGS:
            hide_next = True
    return " ".join(redacted)


@dataclass
class Job:
    job_id: str
    output_dir: Path
    process: subprocess.Popen
    queue: queue.Queue
    log_path: Path
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    seen: set[str] = field(default_factory=set)
    done: bool = False

    def push(self, event: str, data: dict) -> None:
        self.queue.put({"event": event, "data": data})

    def write_log(self, stream: str, line: str) -> None:
        with self.log_lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stream}] {line}\n")


class RunRequest(BaseModel):
    method_text: str = Field(..., min_length=1)
    figure_caption: Optional[str] = None
    enable_svg_reconstruction: bool = True
    provider: str = "gemini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    image_provider: Optional[str] = None
    image_api_key: Optional[str] = None
    image_base_url: Optional[str] = None
    svg_provider: Optional[str] = None
    svg_api_key: Optional[str] = None
    svg_base_url: Optional[str] = None
    image_model: Optional[str] = None
    image_size: Optional[str] = None
    svg_model: Optional[str] = None
    figure_mode: Optional[str] = None
    sam_prompt: Optional[str] = None
    sam_backend: Optional[str] = None
    sam_api_key: Optional[str] = None
    sam_max_masks: Optional[int] = None
    placeholder_mode: Optional[str] = None
    merge_threshold: Optional[float] = None
    optimize_iterations: Optional[int] = None
    reference_image_path: Optional[str] = None
    num_candidates: int = Field(default=1, ge=1, le=8)


class SceneGraphUpdateRequest(BaseModel):
    path: str = "scene_graph.json"
    scene_graph: dict[str, Any]


app = FastAPI()

JOBS: dict[str, Job] = {}


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/config")
def get_config() -> JSONResponse:
    available, rel_path = _resolve_svg_edit_path()
    return JSONResponse({"svgEditAvailable": available, "svgEditPath": rel_path})


@app.post("/api/run")
def run_job(req: RunRequest) -> JSONResponse:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(BASE_DIR / "figureweave.py"),
        "--method_text",
        req.method_text,
        "--output_dir",
        str(output_dir),
        "--provider",
        req.provider,
    ]

    if req.api_key:
        cmd += ["--api_key", req.api_key]
    if req.figure_caption:
        cmd += ["--figure_caption", req.figure_caption]
    if not req.enable_svg_reconstruction:
        cmd += ["--disable_svg_reconstruction", "--stop_after", "1"]
    if req.base_url:
        cmd += ["--base_url", req.base_url]
    if req.image_provider:
        cmd += ["--image_provider", req.image_provider]
    if req.image_api_key:
        cmd += ["--image_api_key", req.image_api_key]
    if req.image_base_url:
        cmd += ["--image_base_url", req.image_base_url]
    if req.svg_provider:
        cmd += ["--svg_provider", req.svg_provider]
    if req.svg_api_key:
        cmd += ["--svg_api_key", req.svg_api_key]
    if req.svg_base_url:
        cmd += ["--svg_base_url", req.svg_base_url]
    if req.image_model:
        cmd += ["--image_model", req.image_model]
    if req.image_size:
        cmd += ["--image_size", req.image_size]
    if req.svg_model:
        cmd += ["--svg_model", req.svg_model]

    figure_mode = req.figure_mode or "simple_flowchart"
    sam_prompt = req.sam_prompt or (
        COMPLEX_PAPER_SAM_PROMPT if figure_mode == "complex_paper" else DEFAULT_SAM_PROMPT
    )
    placeholder_mode = req.placeholder_mode or DEFAULT_PLACEHOLDER_MODE
    merge_threshold = (
        req.merge_threshold if req.merge_threshold is not None else DEFAULT_MERGE_THRESHOLD
    )

    cmd += ["--figure_mode", figure_mode]
    cmd += ["--sam_prompt", sam_prompt]
    cmd += ["--placeholder_mode", placeholder_mode]
    cmd += ["--merge_threshold", str(merge_threshold)]
    if req.sam_backend:
        cmd += ["--sam_backend", req.sam_backend]
    if req.sam_api_key:
        cmd += ["--sam_api_key", req.sam_api_key]
    if req.sam_max_masks is not None:
        cmd += ["--sam_max_masks", str(req.sam_max_masks)]
    if req.optimize_iterations is not None:
        cmd += ["--optimize_iterations", str(req.optimize_iterations)]
    if req.num_candidates and req.num_candidates > 1:
        cmd += ["--num_candidates", str(req.num_candidates)]

    reference_path = req.reference_image_path
    if reference_path:
        reference_path = (
            str((BASE_DIR / reference_path).resolve())
            if not Path(reference_path).is_absolute()
            else reference_path
        )
        cmd += ["--reference_image_path", reference_path]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path = output_dir / "run.log"
    log_path.write_text(
        f"[meta] python={PYTHON_EXECUTABLE}\n[meta] cmd={_redact_cmd_args(cmd)}\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(BASE_DIR),
    )

    job = Job(
        job_id=job_id,
        output_dir=output_dir,
        process=process,
        queue=queue.Queue(),
        log_path=log_path,
    )
    JOBS[job_id] = job

    monitor_thread = threading.Thread(target=_monitor_job, args=(job,), daemon=True)
    monitor_thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/upload")
async def upload_reference(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        ext = ".png"

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    name = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / name
    out_path.write_bytes(data)

    rel_path = out_path.relative_to(BASE_DIR).as_posix()
    return JSONResponse(
        {"path": rel_path, "url": f"/api/uploads/{name}", "name": file.filename}
    )


@app.get("/api/events/{job_id}")
def stream_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        while True:
            try:
                item = job.queue.get(timeout=1.0)
            except queue.Empty:
                if job.done:
                    break
                continue
            if item.get("event") == "close":
                break
            yield _format_sse(item["event"], item["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/artifacts/{job_id}/{path:path}")
def get_artifact(job_id: str, path: str) -> FileResponse:
    output_dir = _resolve_job_output_dir(job_id)
    candidate = (output_dir / path).resolve()
    if not str(candidate).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.get("/api/jobs/{job_id}/artifacts")
def list_job_artifacts(job_id: str) -> JSONResponse:
    output_dir = _resolve_job_output_dir(job_id)
    return JSONResponse({"artifacts": _list_artifacts(output_dir, job_id)})


@app.get("/api/scene-graph/{job_id}")
def get_scene_graph(job_id: str, path: str = "scene_graph.json") -> JSONResponse:
    output_dir = _resolve_job_output_dir(job_id)
    candidate = (output_dir / path).resolve()
    if not str(candidate).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Scene graph not found")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid scene graph JSON: {exc}") from exc
    return JSONResponse({"path": path, "scene_graph": payload})


@app.post("/api/scene-graph/{job_id}")
def save_scene_graph(job_id: str, req: SceneGraphUpdateRequest) -> JSONResponse:
    output_dir = _resolve_job_output_dir(job_id)
    graph_path = (output_dir / req.path).resolve()
    if not str(graph_path).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if graph_path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Scene graph path must be a JSON file")

    from src.figureweave.drawio_ops import (
        export_drawio_from_scene_graph,
        normalize_scene_graph,
        render_drawio_to_svg,
    )

    try:
        scene_graph = normalize_scene_graph(req.scene_graph)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to normalize scene graph: {exc}") from exc

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps(scene_graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    drawio_path = graph_path.with_name("final.drawio")
    svg_path = graph_path.with_name("final.svg")
    try:
        export_drawio_from_scene_graph(scene_graph, str(drawio_path))
        render_drawio_to_svg(str(drawio_path), output_path=str(svg_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild DrawIO/SVG: {exc}") from exc

    stamp = int(time.time() * 1000)
    rel_graph = graph_path.relative_to(output_dir).as_posix()
    rel_drawio = drawio_path.relative_to(output_dir).as_posix()
    rel_svg = svg_path.relative_to(output_dir).as_posix()
    return JSONResponse(
        {
            "path": rel_graph,
            "scene_graph": scene_graph,
            "drawio_path": rel_drawio,
            "svg_path": rel_svg,
            "drawio_url": f"/api/artifacts/{job_id}/{rel_drawio}?ts={stamp}",
            "svg_url": f"/api/artifacts/{job_id}/{rel_svg}?ts={stamp}",
        }
    )


@app.get("/api/uploads/{filename}")
def get_upload(filename: str) -> FileResponse:
    candidate = (UPLOADS_DIR / filename).resolve()
    if not str(candidate).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _monitor_job(job: Job) -> None:
    job.push("status", {"state": "started"})

    stdout_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    idle_cycles = 0
    while True:
        _scan_artifacts(job)

        if job.process.poll() is not None:
            idle_cycles += 1
        else:
            idle_cycles = 0

        if idle_cycles >= 4:
            break
        time.sleep(0.5)

    _scan_artifacts(job)
    job.push("status", {"state": "finished", "code": job.process.returncode})
    job.push(
        "artifact",
        {
            "kind": "log",
            "name": job.log_path.name,
            "path": job.log_path.relative_to(job.output_dir).as_posix(),
            "url": f"/api/artifacts/{job.job_id}/{job.log_path.name}",
        },
    )
    job.done = True
    job.push("close", {})


def _pipe_output(job: Job, pipe, stream_name: str) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        text = line.rstrip()
        if text:
            job.write_log(stream_name, text)
            job.push("log", {"stream": stream_name, "line": text})
    pipe.close()


def _scan_artifacts(job: Job) -> None:
    for item in _list_artifacts(job.output_dir, job.job_id):
        rel_path = item["path"]
        if rel_path in job.seen:
            continue
        job.seen.add(rel_path)
        job.push("artifact", item)


def _classify_artifact(rel_path: str) -> str:
    filename = Path(rel_path).name
    if filename == "figure.png":
        return "figure"
    if filename == "samed.png":
        return "samed"
    if filename.endswith("_nobg.png"):
        return "icon_nobg"
    if "icons/" in rel_path and filename.endswith(".png"):
        return "icon_raw"
    if filename == "template.svg":
        return "template_svg"
    if filename == "optimized_template.svg":
        return "optimized_template_svg"
    if filename == "final.svg":
        return "final_svg"
    if filename == "final.drawio":
        return "final_drawio"
    if filename == "llm_final.svg":
        return "llm_final_svg"
    if filename == "scene_graph.json":
        return "scene_graph"
    if filename == "candidates_manifest.json":
        return "candidate_manifest"
    if filename == "candidate_error.log":
        return "candidate_error"
    if filename == "run.log":
        return "log"
    return "artifact"


def _extract_candidate_label(rel_path: str) -> str | None:
    parts = Path(rel_path).parts
    if parts and parts[0].startswith("candidate_"):
        return parts[0].replace("candidate_", "C")
    return None


def _display_artifact_name(rel_path: str) -> str:
    candidate_label = _extract_candidate_label(rel_path)
    name = Path(rel_path).name
    if candidate_label:
        return f"{candidate_label} / {name}"
    return name


def _resolve_job_output_dir(job_id: str) -> Path:
    job = JOBS.get(job_id)
    if job:
        return job.output_dir.resolve()
    candidate = (OUTPUTS_DIR / job_id).resolve()
    if not str(candidate).startswith(str(OUTPUTS_DIR.resolve())) or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    return candidate


def _candidate_paths(output_dir: Path) -> list[Path]:
    candidates = [
        output_dir / "figure.png",
        output_dir / "samed.png",
        output_dir / "template.svg",
        output_dir / "final.svg",
        output_dir / "final.drawio",
        output_dir / "llm_final.svg",
        output_dir / "scene_graph.json",
        output_dir / "optimized_template.svg",
        output_dir / "candidates_manifest.json",
        output_dir / "candidate_error.log",
        output_dir / "run.log",
    ]

    icons_dir = output_dir / "icons"
    if icons_dir.is_dir():
        candidates.extend(sorted(icons_dir.glob("icon_*.png")))

    for candidate_dir in sorted(output_dir.glob("candidate_*")):
        if not candidate_dir.is_dir():
            continue
        candidates.extend(
            [
                candidate_dir / "figure.png",
                candidate_dir / "samed.png",
                candidate_dir / "template.svg",
                candidate_dir / "final.svg",
                candidate_dir / "final.drawio",
                candidate_dir / "llm_final.svg",
                candidate_dir / "scene_graph.json",
                candidate_dir / "optimized_template.svg",
                candidate_dir / "candidate_error.log",
            ]
        )
        candidate_icons_dir = candidate_dir / "icons"
        if candidate_icons_dir.is_dir():
            candidates.extend(sorted(candidate_icons_dir.glob("icon_*.png")))
    return candidates


def _list_artifacts(output_dir: Path, job_id: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in _candidate_paths(output_dir):
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        candidate_label = _extract_candidate_label(rel_path)
        artifacts.append(
            {
                "kind": _classify_artifact(rel_path),
                "name": path.name,
                "display_name": _display_artifact_name(rel_path),
                "path": rel_path,
                "url": f"/api/artifacts/{job_id}/{rel_path}",
                "candidate_label": candidate_label,
                "primary": candidate_label is None,
            }
        )
    artifacts.sort(key=lambda item: (0 if item["primary"] else 1, item["path"]))
    return artifacts


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def _pids_on_port(port: int) -> set[int]:
    pids: set[int] = set()

    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-lptn", f"sport = :{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "pid=" in line:
                for part in line.split("pid=")[1:]:
                    pid_str = "".join(ch for ch in part if ch.isdigit())
                    if pid_str:
                        pids.add(int(pid_str))
        return pids

    if shutil.which("netstat"):
        result = subprocess.run(
            ["netstat", "-tlnp"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port} " not in line or "LISTEN" not in line:
                continue
            fields = line.split()
            if fields and "/" in fields[-1]:
                pid_part = fields[-1].split("/")[0]
                if pid_part.isdigit():
                    pids.add(int(pid_part))

    return pids


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read()
        parts = [p for p in data.split(b"\x00") if p]
        return " ".join(part.decode(errors="ignore") for part in parts)
    except OSError:
        return ""


def _is_uvicorn_process(pid: int) -> bool:
    cmdline = _read_cmdline(pid)
    if not cmdline:
        return False
    if "uvicorn" not in cmdline:
        return False
    return "server:app" in cmdline or "server.py" in cmdline


def _terminate_pids(pids: set[int], timeout: float = 2.0) -> None:
    current_pid = os.getpid()
    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = False
        for pid in pids:
            if pid <= 1 or pid == current_pid:
                continue
            if not _is_uvicorn_process(pid):
                continue
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                continue
        if not alive:
            return
        time.sleep(0.1)

    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _ensure_port_free(port: int) -> None:
    if not _port_in_use(port):
        return
    pids = _pids_on_port(port)
    if not pids:
        return
    _terminate_pids(pids)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    def find_available_port(start_port: int, max_attempts: int = 100) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    print(f"Port {port} is in use, trying next...")
                    continue
        raise IOError(f"No available ports found in range ({start_port} - {start_port + max_attempts})")

    initial_port = 8000
    
    try:
        actual_port = find_available_port(initial_port)
        
        print(f"--- Starting Server ---")
        print(f"Local access: http://127.0.0.1:{actual_port}")
        print(f"-----------------------")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=actual_port,
            reload=False,
            access_log=False,
        )
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)
