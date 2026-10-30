"""Write "Part 6. Code reference" at the bottom of docs/GUIDE.md: real excerpts of the code, each with what it does.

  python docs/build_code_reference.py        (also run by docs/build_pdf.sh)

The excerpts are cut from the source files every time (by function name, or by line markers for YAML and other
files), so they always match the code. Only the text between the two markers in GUIDE.md is replaced. If a function
is renamed, this script stops with an error naming it: update ENTRIES below.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "GUIDE.md"
START, END = "<!-- CODE-REFERENCE:START -->", "<!-- CODE-REFERENCE:END -->"


def py_symbol(path: str, name: str) -> tuple[int, int, str]:
    """(first line, last line, source) of a top-level function/class, or Class.method, including decorators."""
    text = (REPO / path).read_text()
    nodes = ast.parse(text).body
    for part in name.split("."):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            raise SystemExit(f"{path}: no {name!r} (renamed? update ENTRIES in docs/build_code_reference.py)")
        nodes = getattr(node, "body", [])
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines()[first - 1:node.end_lineno]
    return first, node.end_lineno, "\n".join(lines)


def text_block(path: str, start_marker: str, n_lines: int | None = None) -> tuple[int, int, str]:
    """Lines from the first line containing `start_marker`: `n_lines` of them, or up to the next blank line."""
    lines = (REPO / path).read_text().splitlines()
    i = next((k for k, l in enumerate(lines) if start_marker in l), None)
    if i is None:
        raise SystemExit(f"{path}: marker {start_marker!r} not found (update ENTRIES in docs/build_code_reference.py)")
    j = i + n_lines if n_lines else next((k for k in range(i, len(lines)) if not lines[k].strip()), len(lines))
    return i + 1, j, "\n".join(lines[i:j])


def whole(path: str) -> tuple[int, int, str]:
    lines = (REPO / path).read_text().rstrip("\n").splitlines()
    return 1, len(lines), "\n".join(lines)


# (section title, [(path, how to cut, language)], explanation in Markdown)
ENTRIES = [
    ("Start-up: the config helper every command loads first",
     [("src/__init__.py", ("whole",), "python")],
     """Any `python -m src.<something>` imports `src/__init__.py` first. This one line teaches the config a function
`len`, so `conf/config.yaml` can say `classes: ${len:${data.classes}}`: the network gets one output channel per class
in `conf/data/<class>.yaml` (1 for a per-class model, 4 for `data=loveda`), and 1 when there is no class list."""),

    ("ClearML <-> Hydra: how a run gets its settings (`init_clearml`)",
     [("src/common/clearml_utils.py", ("py", "init_clearml"), "python")],
     """Every task script calls this first. Step by step:

1. **ClearML off** (`clearml.enabled=false`): return `(None, cfg)`. Every report call then does nothing, so a local
   run needs no server.
2. **`Task.add_requirements`**, before `Task.init`: packages ClearML's import scan would miss, plus the `onnx<1.19` /
   `protobuf<6` pins that keep ONNX export working next to Kaggle's system protobuf 5.
3. **`Task.init`** creates a new task every time (`reuse_last_task_id=False`), with Hydra's automatic logging off,
   because Hydra stays the only place settings come from.
4. **`state`**: the current value of every key in `clearml.tunable`, as text. For a config group such as `data`, the
   value is the chosen file name (`road`), not its contents.
5. **`task.connect(state, name="Overrides")`**: on your computer it gives the same values back; on an agent it gives
   the values the UI or the pipeline wrote into the clone. The code compares against a copy (`original`) because
   `connect()` changes the dict it is given.
6. **Different?** Then `recompose` builds the Hydra config again with those values (next excerpt).
7. The final config is stored on the task (`connect_configuration`), plus the class names and tags.
8. Only if you asked for it: `execute_remotely` sends this run to a queue."""),

    ("Turning ClearML values back into Hydra overrides (`_fmt`, `_quote`, `recompose`)",
     [("src/common/clearml_utils.py", ("py", "_fmt"), "python"),
      ("src/common/clearml_utils.py", ("py", "_quote"), "python"),
      ("src/common/clearml_utils.py", ("py", "recompose"), "python")],
     """- `_fmt` writes a value the way Hydra expects on a command line: `true`/`false`, and `null` for ClearML's
  `"None"` or `""`.
- `_quote` puts quotes around values Hydra would otherwise misread (anything with `= , : [ ] { } ( )` or spaces), for
  example the pipeline's `serve.spec="building=unet:resnet34_imagenet,..."`.
- `recompose` rebuilds the config: the original command-line overrides, minus the tunable keys, plus `key=value` for
  every tunable key. Dropping them from the command-line part is what lets the edited value win."""),

    ("Failing a gate (`fail_gate`)",
     [("src/common/clearml_utils.py", ("py", "fail_gate"), "python")],
     """Both gates end this way when a limit is broken: the task is marked **failed** (with the reason visible in the
ClearML UI), a line such as `SERVE GATE FAILED: ...` is printed, and the process exits with **code 2**. A pipeline
step that exits non-zero fails, and ClearML never creates the steps after it."""),

    ("The dataset: where the data comes from and what a sample is (`src/data/tiles.py`)",
     [("src/data/tiles.py", ("py", "resolve_root"), "python"),
      ("src/data/tiles.py", ("py", "TileDataset.image"), "python"),
      ("src/data/tiles.py", ("py", "TileDataset.mask"), "python"),
      ("src/data/tiles.py", ("py", "TileDataset.__getitem__"), "python")],
     """- `resolve_root`: with `data.clearml_dataset_id` set, the dataset is downloaded once from ClearML into a local
  cache and that folder is used; otherwise the local `data.root`.
- `image`: reads the file and divides by 255, the same scaling the Chisel server applies (no mean/std).
- `mask`: turns LoveDA's class ids into **one yes/no channel per class**: `np.isin(m, ids)` is 1 where the pixel's id
  is in that class's list (`vegetation: [6, 7]` -> forest or agriculture). Shape `(classes, H, W)`.
- `__getitem__`: one training sample. With `augment` (training only), random flips and a random 90-degree rotation are
  applied to image and mask together, so they stay aligned."""),

    ("One training run, top to bottom (`src/training/train.py`)",
     [("src/training/train.py", ("py", "main"), "python")],
     """Read it as the list in section 3.1:

1. Settings: `init_clearml`, a fixed seed, and the resolved config saved as an artifact.
2. Data: `resolve_root`; then the data gate, unless `preflight.enabled=false` (the pipeline sets that, because
   `domain_check` already ran).
3. Training: Lightning's `Trainer` with mixed precision on a GPU, the best checkpoint kept by `val/iou`, and early
   stopping after `train.patience` epochs without improvement. Then the best checkpoint is loaded back.
4. Sweep: every TTA x threshold on the validation split; results to ClearML and `metadata.json`.
5. ONNX: the model and its sidecar; the threshold is the best one **without TTA**, because the server does no TTA.
6. Publish: the task is set to become read-only when it completes."""),

    ("Loss and metrics (`src/training/module.py`)",
     [("src/training/module.py", ("py", "confusion"), "python"),
      ("src/training/module.py", ("py", "iou_f1"), "python"),
      ("src/training/module.py", ("py", "SegModule.loss"), "python"),
      ("src/training/module.py", ("py", "SegModule.on_validation_epoch_end"), "python")],
     """- `confusion` counts true positives, false positives and false negatives; `iou_f1` turns them into IoU
  (`tp / (tp + fp + fn)`) and F1 (`2tp / (2tp + fp + fn)`). The sweep uses the same two functions.
- `loss` = `bce_weight x BCE + dice_weight x Dice`. BCE scores every pixel; Dice scores the overlap of the whole mask,
  which helps thin or rare classes such as road. `pos_weight` can make positive pixels count more.
- `on_validation_epoch_end`: logs `val/iou` (the mean over classes: this is what picks the best checkpoint) and, with
  several classes, `val/iou_<class>`; it also remembers the best epoch and sends one picture per class to ClearML
  (prediction red, truth green)."""),

    ("Choosing the threshold: the sweep (`src/training/sweep.py`)",
     [("src/training/sweep.py", ("py", "run_sweep"), "python")],
     """For each TTA setting (`none`, `flip` = 4 views, `d4` = 8 views) the model predicts every validation tile,
averaging the views after undoing each flip/rotation. One pass then scores **all thresholds at once** (0.30 to 0.70 by
default): `probs[None] > thr.view(...)` gives one prediction per threshold. The result is a table with one row per
(TTA, threshold); `best` is the best row, and `best["class_thresholds"]` the best threshold per class. If the best
value sits at the edge of the range (road: 0.70), widen `sweep.thresholds`."""),

    ("The data gate (`src/gates/preflight.py`)",
     [("src/gates/preflight.py", ("py", "knn_interleaving"), "python"),
      ("src/gates/preflight.py", ("py", "gate_preflight"), "python")],
     """- `knn_interleaving`: every image is summarised as a few numbers (per-channel mean, std, percentiles). For each
  image, look at its 10 nearest neighbours: if train and val are alike, about as many neighbours come from the other
  split as chance predicts. `shift_score` 0 = well mixed, 1 = the two splits look completely different. The gate fails
  above `max_shift` (0.5); measured on LoveDA: 0.004.
- `gate_preflight`: samples up to `max_samples` tiles of each split, runs the checks and, if a check in `gate` mode
  fails, calls `fail_gate` (exit code 2). Used by `domain_check` and at the start of training."""),

    ("The serving measurement (`src/gates/serve_check.py`)",
     [("src/gates/serve_check.py", ("py", "Model"), "python"),
      ("src/gates/serve_check.py", ("py", "request"), "python")],
     """This copies how the Chisel server runs its models, so the numbers mean the same thing:

- `Model`: an onnxruntime CPU session with the server's options (2 threads per model, memory arena off), the threshold
  from the sidecar, and a **lock**: one model runs one tile at a time, as in the server.
- `request`: one user request = every tile x every model, submitted to a thread pool of `--workers` (the server's
  `THREAD_POOL_SIZE`, 2), with at most `max(2 x workers, models)` jobs waiting. The wall time of this is the latency."""),

    ("The serving gate as a task (`src/gates/serve_gate.py`)",
     [("src/gates/serve_gate.py", ("py", "measure_isolated"), "python"),
      ("src/gates/serve_gate.py", ("py", "main"), "python")],
     """- `main` gets the models (`serve.source`: `random` exports untrained ones before training, `tasks` downloads the
  ONNX files of this pipeline's train steps, `folder` uses local files), measures, reports to ClearML, and fails the
  gate on a violation.
- `measure_isolated` runs the measurement in a **new process**. Exporting models uses a lot of memory; measured in
  the same process, that would count as the models' peak."""),

    ("The pipeline (`src/orchestration/pipeline.py`)",
     [("src/orchestration/pipeline.py", ("py", "plan"), "python"),
      ("src/orchestration/pipeline.py", ("py", "newest_base_task_id"), "python"),
      ("src/orchestration/pipeline.py", ("py", "build"), "python")],
     """- `plan` returns the pipeline as a plain list (this is what `--dry-run` prints): two gates, one `train_<class>`
  step per class after both gates, and `serve_gate_post` after all training steps. Each step carries the settings it
  writes into its clone (`parameter_override`), e.g. `Overrides/data=road`.
- `newest_base_task_id` picks the newest **non-archived** base task by exact name. ClearML's own lookup also returns
  archived tasks and skips `failed` ones, which once made a pipeline run old code.
- `build` hands the steps to ClearML's `PipelineController`."""),

    ("ONNX export (`src/export_onnx.py`)",
     [("src/export_onnx.py", ("py", "threshold_from_rows"), "python"),
      ("src/export_onnx.py", ("py", "export_model"), "python")],
     """- `threshold_from_rows`: from the sweep table, keep only the rows with TTA `none` and take the best threshold.
- `export_model`: adds a sigmoid to the network (the file outputs probabilities), exports with a fixed input of
  `(1, 3, size, size)` (1280, the server's input size), then **runs the file with onnxruntime and compares it with
  PyTorch** on the same random input; a difference above 1e-3
  stops the run. Last, it writes the sidecar `<class>.onnx.json` with the class name and the threshold."""),

    ("Making the file smaller (`src/shrink.py`)",
     [("src/shrink.py", ("py", "quantize_tensor"), "python")],
     """For every large weight tensor: find its min and max, store it as **int8** (256 levels) plus a scale and an
offset, and add three small operations (Cast, Mul, Add) that turn it back into float when the model runs. The file
becomes about 4x smaller (97.7 -> 24.6 MB); latency was unchanged in the test. The code uses Cast/Mul/Add instead of
ONNX's `DequantizeLinear` because the latter is slow in onnxruntime (comment in the original code). The accuracy of a
shrunk *trained* model is not measured yet."""),

    ("Building the network (`src/model_factory.py`)",
     [("src/model_factory.py", ("py", "build_model"), "python")],
     """`instantiate(cfg.model)` calls `segmentation_models_pytorch.create_model` with the values from
`conf/architecture/` and `conf/encoder/` (e.g. `arch=Unet, encoder_name=resnet34, encoder_weights=imagenet`). Optional
extra step: load encoder weights from a local file (offline) or from torchgeo, and fail if keys are missing."""),

    ("Inside the server: probabilities -> mask (`serving/postprocess_semantic.py`)",
     [("serving/postprocess_semantic.py", ("py", "postprocess_semantic_masks"), "python")],
     """The server shrinks or enlarges every tile to fit 1280 x 1280 and pads the rest grey ("letterbox"). This undoes
that: cut away the padding (`top`, `left`, `bottom`, `right` from the server's own `pad` and `ratio`), resize the
probabilities back to the tile's real size, and threshold. The result is the same 0/255 mask the server's YOLO code
returns, so nothing else in the server changes."""),

    ("The server change (`serving/chisel_server.patch`)",
     [("serving/chisel_server.patch", ("whole",), "diff")],
     """Two changes to `server.py`. When the server loads a model, it looks for `<model>.onnx.json`; if there is one,
its `threshold` is kept. When it predicts, a model with a threshold goes through `postprocess_semantic_masks`; a
model without one (the current YOLO files) goes through the old code with its old `confidence`. So old and new models
can even be mixed."""),

    ("Turning a Kaggle notebook into an agent (`kaggle/start_agent.py`)",
     [("kaggle/start_agent.py", ("text", "secrets = UserSecretsClient()", 17), "python")],
     """Reads your ClearML keys from Kaggle's secrets, points the agent at app.clear.ml, tells it to reuse Kaggle's
preinstalled CUDA torch instead of installing its own (`system_site_packages`), installs `clearml-agent`, and starts it
on the `kaggle` queue with GPU 0. The last line keeps running; the agent stops when the notebook session ends."""),

    ("Which settings ClearML may change (`conf/config.yaml`)",
     [("conf/config.yaml", ("text", "  tunable:"), "yaml")],
     """Only these keys can be changed from the ClearML UI, a pipeline or HPO (they appear as "Overrides/<key>" on a
task). Anything else stays as the YAML and the command line say. To open a new knob: add its key here, push, and
re-register the base tasks."""),

    ("One architecture per class (`pipeline/pipeline.yaml`)",
     [("pipeline/pipeline.yaml", ("text", "classes:"), "yaml")],
     """The single table the pipeline reads for the classes. The names must match `conf/data/<class>.yaml`; the values
name files in `conf/architecture/` and `conf/encoder/`. Change a line (e.g. `road: {architecture: fpn, ...}`) and both
the pre-training gate and that class's training step use it."""),
]


def cut(path, how):
    kind = how[0]
    if kind == "py":
        return py_symbol(path, how[1])
    if kind == "text":
        return text_block(path, *how[1:])
    return whole(path)


def build() -> str:
    out = [START, "", "# Part 6. Code reference (annotated excerpts)", "",
           "The important pieces of the code, in the order a run uses them. Each excerpt is cut from the file it",
           "names, with its line numbers, so it is exactly the code in the repository. Read the explanation under it.",
           "Regenerate with `python docs/build_code_reference.py` after changing the code (`docs/build_pdf.sh` does it).",
           ""]
    for n, (title, parts, text) in enumerate(ENTRIES, 1):
        out += [f"## 6.{n} {title}", ""]
        for path, how, lang in parts:
            first, last, src = cut(path, how)
            out += [f"`{path}`, lines {first} to {last}:", "", f"```{lang}", src, "```", ""]
        out += ["**What it does.**", "", text.strip(), ""]
    return "\n".join(out + [END, ""])


if __name__ == "__main__":
    guide = GUIDE.read_text()
    block = build()
    if START in guide:
        guide = guide[:guide.index(START)] + block + guide[guide.index(END) + len(END):].lstrip("\n")
    else:
        guide = guide.rstrip("\n") + "\n\n---\n\n" + block
    GUIDE.write_text(guide)
    print(f"docs/GUIDE.md: Part 6 written ({len(ENTRIES)} sections)")
