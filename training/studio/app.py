from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.studio.core import (
    DEFAULT_WORK_ROOT,
    create_batch,
    finalize_dataset,
    generate_candidates,
    review_counts,
    review_rows,
    roi_preview_paths,
    update_review_row,
)

ROOT = Path(__file__).resolve().parents[2]

STUDIO_CSS = """
:root {
  --studio-ink: #1d1d1f;
  --studio-muted: #6e6e73;
  --studio-surface: rgba(255, 255, 255, .72);
  --studio-blue: #0071e3;
}
body, .gradio-container {
  background: #f5f5f7 !important;
  color: var(--studio-ink) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif !important;
  font-optical-sizing: auto;
}
.gradio-container { max-width: 1440px !important; padding: clamp(1rem, 3vw, 3.5rem) !important; }
#studio-heading {
  background: var(--studio-surface);
  backdrop-filter: blur(22px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, .75);
  border-radius: 22px;
  box-shadow: 0 12px 30px rgba(29, 29, 31, .08);
  margin-bottom: 1.5rem;
  padding: clamp(1.25rem, 3vw, 2.5rem);
}
#studio-heading h1 { font-size: clamp(2rem, 5vw, 3.7rem); letter-spacing: -.045em; line-height: 1; margin: 0; }
#studio-heading p { color: var(--studio-muted); font-size: 1rem; margin: .75rem 0 0; }
.gr-button-primary { background: var(--studio-blue) !important; border: 0 !important; }
.gr-button { border-radius: 980px !important; transition: transform 120ms ease-out, opacity 120ms ease-out !important; }
.gr-button:active { transform: scale(.97); }
.tab-nav button { font-weight: 600 !important; letter-spacing: -.01em; }
.block, .gr-panel { border-radius: 16px !important; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
@media (prefers-reduced-transparency: reduce), (prefers-contrast: more) {
  #studio-heading { background: #fff; backdrop-filter: none; border-color: #86868b; }
}
"""


def _message(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def create_app(work_root: Path = DEFAULT_WORK_ROOT):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("OCRKit Studio requires `uv sync --extra studio --extra vision` (including its SOCKS extra).") from exc

    def import_images(files: list[Any] | None, holdout_ratio: float) -> tuple[str, str, list[tuple[str, str]]]:
        paths = [item.name if hasattr(item, "name") else str(item) for item in files or []]
        batch_dir, summary = create_batch(paths, work_root=work_root, holdout_ratio=holdout_ratio)
        return str(batch_dir), _message(summary), roi_preview_paths(batch_dir)

    def generate(batch: str) -> tuple[str, str]:
        if not batch:
            raise gr.Error("先导入一个批次。")
        return _message(generate_candidates(Path(batch))), _message(review_counts(Path(batch)))

    def list_rows(batch: str, split: str, status: str) -> tuple[list[list[str]], list[str]]:
        rows = review_rows(Path(batch), split, status) if batch else []
        choices = [str(row["crop"]) for row in rows]
        table = [[row["crop"], row.get("roi", ""), row.get("review_status", ""), row.get("candidate_text") or "", str(row.get("confidence") or "")] for row in rows]
        return table, choices

    def select_row(batch: str, split: str, crop: str) -> tuple[str | None, str, str]:
        rows = review_rows(Path(batch), split) if batch else []
        row = next((item for item in rows if item.get("crop") == crop), None)
        if row is None:
            return None, "", ""
        return str(Path(batch) / "dataset" / str(row["crop"])), str(row.get("transcription") or row.get("candidate_text") or ""), _message(row)

    def save_review(batch: str, split: str, crop: str, status: str, transcription: str) -> tuple[str, str]:
        row = update_review_row(Path(batch), split, crop, status, transcription)
        return _message(row), _message(review_counts(Path(batch)))

    def finalize(batch: str) -> str:
        return _message(finalize_dataset(Path(batch)))

    def start_smoke(batch: str) -> str:
        if not batch:
            raise gr.Error("先导入并完成标注。")
        batch_dir = Path(batch)
        finalize_dataset(batch_dir)
        run_dir = batch_dir / "runs" / f"smoke-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training.log"
        output_dir = run_dir / "checkpoints"
        command = [str(ROOT / "training/run_rec_smoke.sh"), "--labels-dir", str(batch_dir / "dataset"), "--output-dir", str(output_dir)]
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = {"pid": process.pid, "status": "training", "command": command, "log": str(log_path)}
        (run_dir / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (batch_dir / "runs/latest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return _message(state)

    def training_status(batch: str) -> tuple[str, str]:
        state_path = Path(batch) / "runs/latest.json"
        if not state_path.is_file():
            return "尚未启动训练。", ""
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(state["pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            state["status"] = "completed_or_failed"
        log_path = Path(state["log"])
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.is_file() else ""
        return _message(state), tail

    with gr.Blocks(title="OCRKit Studio", theme=gr.themes.Soft(), css=STUDIO_CSS) as app:
        gr.Markdown(
            "# OCRKit Studio\n"
            "本机离线的 Recognition 数据集与训练工作台。批次、截图、切片和日志只保存在 `training/.work/studio/`。",
            elem_id="studio-heading",
        )
        batch = gr.Textbox(label="当前批次目录", interactive=False)
        with gr.Tab("1 · 导入"):
            uploads = gr.File(label="截图", file_count="multiple", file_types=["image"])
            ratio = gr.Slider(0, 0.5, value=0.2, step=0.05, label="按原始截图保留 holdout 比例")
            import_button = gr.Button("创建私有批次", variant="primary")
            import_summary = gr.Code(label="批次摘要", language="json")
            previews = gr.Gallery(label="首张截图的规范化画布与固定 ROI", columns=3, height="auto")
            import_button.click(import_images, [uploads, ratio], [batch, import_summary, previews])
        with gr.Tab("2 · 候选与 ROI"):
            generate_button = gr.Button("生成或显示 RapidOCR + Vision 候选", variant="primary")
            candidate_summary = gr.Code(label="候选生成结果", language="json")
            counts = gr.Code(label="审核状态", language="json")
            generate_button.click(generate, batch, [candidate_summary, counts])
        with gr.Tab("3 · 人工复核"):
            with gr.Row():
                split = gr.Radio(["train", "holdout"], value="train", label="数据分组")
                status_filter = gr.Radio(["pending", "accepted", "rejected", "all"], value="pending", label="筛选")
                refresh = gr.Button("刷新候选")
            rows_table = gr.Dataframe(headers=["crop", "ROI", "状态", "候选文本", "置信度"], interactive=False)
            crop = gr.Dropdown(label="当前切片", choices=[])
            preview = gr.Image(label="切片预览", type="filepath")
            transcription = gr.Textbox(label="人工转写")
            candidate_detail = gr.Code(label="候选详情", language="json")
            with gr.Row():
                accept = gr.Button("接受", variant="primary")
                reject = gr.Button("拒绝")
            saved = gr.Code(label="保存结果", language="json")
            refresh.click(list_rows, [batch, split, status_filter], [rows_table, crop])
            crop.change(select_row, [batch, split, crop], [preview, transcription, candidate_detail])
            accept.click(lambda b, s, c, t: save_review(b, s, c, "accepted", t), [batch, split, crop, transcription], [saved, counts])
            reject.click(lambda b, s, c, t: save_review(b, s, c, "rejected", t), [batch, split, crop, transcription], [saved, counts])
        with gr.Tab("4 · 数据集"):
            finalize_button = gr.Button("验证并生成 recognition labels", variant="primary")
            finalize_result = gr.Code(label="结果", language="json")
            finalize_button.click(finalize, batch, finalize_result)
        with gr.Tab("5 · 训练"):
            gr.Markdown("Smoke 训练会在独立进程中运行；它不会发布模型，也不会修改正式数据集。")
            start = gr.Button("启动 CPU Smoke 训练", variant="primary")
            poll = gr.Button("刷新训练状态")
            training_state = gr.Code(label="任务状态", language="json")
            training_log = gr.Textbox(label="训练日志末尾", lines=18, interactive=False)
            start.click(start_smoke, batch, training_state)
            poll.click(training_status, batch, [training_state, training_log])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only OCRKit dataset and training studio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    args = parser.parse_args()
    create_app(args.work_root).launch(server_name=args.host, server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
