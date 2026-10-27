# Building-segmentation pipeline: a step-by-step guide for people new to ClearML

This guide explains how the project works, how to run it, and what goes wrong. It assumes you know
basic Python and git, but nothing about ClearML.

Status labels used below:

- **Verified**: run end to end during development (on a Mac and on a Kaggle T4 GPU).
- **Untested**: written, but never run. Treat it as a draft.

---

## 1. What this project does

It trains a model that finds **buildings in aerial images** (binary segmentation: building / not building),
using the LoveDA dataset. Around the training it adds:

- **ClearML** to record every run (settings, metrics, files) and to run jobs on a remote GPU (Kaggle).
- **Hydra** for configuration (YAML files you can combine and override).
- **Two checks that run before training**: can the model run inside the production limits, and is the
  training data consistent? If a check fails, training does not start.

```
                 resource_check ─┐
(you start it)                   ├──> train ──> checkpoint + metadata.json
                 domain_check   ─┘
```

---

## 2. ClearML in ten minutes

| ClearML word | Plain meaning | In this project |
|---|---|---|
| **Server** | The website and database that store everything | Hosted: https://app.clear.ml (free account) |
| **Project** | A folder of tasks | `building-segmentation` |
| **Task** | One recorded run of a script: code version, settings, logs, metrics, files | One run of `src.train`, `src.resource_check`, ... |
| **Hyperparameters** | Settings stored on a task. You can edit them in the web UI before re-running | The `Overrides` section (see section 7) |
| **Artifact** | A file attached to a task | `best_checkpoint`, `metadata`, `resolved_config` |
| **Model** | A checkpoint file ClearML tracks | `best.ckpt` (captured automatically) |
| **Dataset** | A versioned folder of data stored on the server | `loveda-buildings` |
| **Queue** | A waiting list of tasks to run | `kaggle` |
| **Agent** (worker) | A program on a machine that takes tasks from a queue and runs them | `clearml-agent` running in a Kaggle notebook |
| **Draft task** | A task that is not running; you can edit it | The base tasks (section 9) |
| **Clone** | A copy of a task as a new draft | How every remote run is created |
| **Enqueue** | Put a draft task on a queue so an agent runs it | |
| **Pipeline** | Several tasks chained so that one starts after another finishes | `gated-training` |

The central idea: **a task stores which git commit, which command and which settings to run.** An agent
checks out that commit, installs the packages, applies the settings and runs the command. Your own laptop
only has to create the task and watch it.

---

## 3. What is in the repository

```
conf/                  Hydra configuration (the single source of truth for settings)
  config.yaml            defaults + the list of settings ClearML may override ("tunable")
  architecture/ encoder/ loss/ data/ (one file per class) preflight/ resource/ serve/
src/
  train.py               training (Lightning), TTA x threshold sweep, writes metadata.json
  resource_check.py      "can this model run in the production limits?" gate
  domain_check.py        "are train and val consistent?" gate (the preflight checks)
  pipeline.py            the ClearML pipeline: gates, one train step per class, final gate
  clearml_utils.py       ClearML glue: Task.init, Overrides, recompose
  clearml_logger.py      sends Lightning metrics to ClearML
  dataset.py             upload a folder to ClearML as a Dataset
  prepare_data.py        LoveDA -> train/val split (class-id masks, or one binary mask)
  export_weights.py      save pretrained encoder weights to a file (offline use)
  hpo.py                 hyperparameter search (Optuna) over ClearML tasks
  export_onnx.py         checkpoint -> ONNX in the Chisel server's format (+ threshold sidecar)
  serve_check.py         production-path timing of all ONNX models together (command line)
  serve_gate.py          the same as a ClearML gate: before training (untrained) and after (real models)
  envinfo.py             container / memory helpers shared by the resource checks
serving/                 drop-in post-processing for the Chisel server + its geometry test
pipeline/pipeline.yaml   queues, dataset id and task names used by the pipeline
kaggle/start_agent.py    the cell that turns a Kaggle notebook into an agent
offline/                 drafts for running without internet (Untested)
```

---

## 4. One-time setup on your computer

1. Install **Python 3.12**. (The newest Python often has no wheels for torch, rasterio or torchgeo.)
2. Create an environment and install the packages:
   ```bash
   cd chisel_pipeline_new
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a free account at https://app.clear.ml. Go to **Settings → Workspace → Create new
   credentials** and copy the block it shows.
4. Run `clearml-init` and paste that block. It writes `~/clearml.conf`.
   Check that the first lines of that file say `api.clear.ml`, not `localhost`.
5. Never commit `clearml.conf` or paste its keys anywhere. `.gitignore` already excludes it.

---

## 5. The data

The raw LoveDA masks have values 0 to 7 (0 no-data, 1 background, **2 building**, 3 road, 4 water,
5 barren, 6 forest, 7 agriculture). `prepare_data.py` splits them into train and val.

```bash
python -m src.prepare_data --src data/loveda --out data/loveda_multi                # several classes (default)
python -m src.prepare_data --src data/loveda --out data/loveda_split --mode binary  # one building mask
```

It creates `train/` and `val/` (80/20, random per tile, seed 42), each with `images/` and `masks/`.
In the default mode the masks keep their original class ids; in `binary` mode they are 0 or 255.
Both modes are **Verified** on a small sample. Known limit: the split is random per tile, so neighbouring
tiles of one scene can end up on both sides, which makes validation scores slightly optimistic.

### Several binary classes (water, road, building, vegetation)

The model is **multi-label**: one network with one output channel per class, and each channel is an
independent yes/no mask (a pixel may belong to several classes). The classes are defined in
`conf/data/loveda.yaml`, which is the default data config:

```yaml
classes:                      # name -> list of mask ids that count as that class
  building: [2]
  road: [3]
  water: [4]
  vegetation: [6, 7]          # forest + agriculture (LoveDA has no single "vegetation" class)
```

Add, remove or regroup entries and everything follows: the number of output channels
(`model.classes` is computed from the list), the loss (BCE + multilabel Dice), the metrics, the sweep and the
label-balance check. `val/iou` and `val/f1` are the **mean over classes** (this is the HPO objective);
`val/iou_<class>` and `val/f1_<class>` are per class. The sweep picks the best TTA and threshold on the mean
and also records a best threshold per class in `metadata.json` (`inference_defaults.class_thresholds`).

For a single binary mask use `data=buildings` (masks must already be 0/255). A guard rejects a mismatch,
for example binary masks together with the class list, instead of training on empty targets.
This feature is **Verified** locally on a 24-tile sample (training, sweep, metadata, resource check); it has
not been run on the full data or on Kaggle yet.

Upload the data to ClearML once so that remote agents can download it:

```bash
python -m src.dataset --path data/loveda_multi --name loveda-multi
# prints:  dataset id: <32 hex characters>
```

The dataset id is stored in `pipeline/pipeline.yaml`. **Datasets made by older versions of this guide
(binary masks, name `loveda-buildings`) only work with `data=buildings`.** A remote run uses the dataset when the setting
`data.clearml_dataset_id` is set; a local run uses the folder `data.root` instead.
Upload size was 2.25 GiB and fit the free tier.

---

## 6. Running on your own computer

Without ClearML (quick smoke test, **Verified**, about 7 minutes per epoch on an M3 Mac):

```bash
python -m src.train clearml.enabled=false data.root=data/loveda_multi \
    train.epochs=1 train.num_workers=0 train.batch_size=4 sweep.enabled=false
```

With ClearML (a task appears on app.clear.ml): leave out `clearml.enabled=false`.

Settings are Hydra overrides in the form `key=value`. In zsh, quote values containing brackets:
`'sweep.tta=[none]'`.

Memory warning: training uses full 1024x1024 tiles. At batch size 4 the process used about 10 GB on the
Mac. Close other heavy apps, or lower the batch size.

---

## 7. How one run works inside (read this once)

`src/clearml_utils.py: init_clearml` does the ClearML part in four steps:

1. **`Task.init`**: creates the task on the server and records the git repo, commit and command.
2. **`task.connect(state, name="Overrides")`**: publishes a short list of settings (the `tunable` list in
   `conf/config.yaml`, for example `train.epochs`, `optimizer.lr`, `architecture`) as editable
   hyperparameters. Everything is stored **as text**.
3. **Recompose**: when an agent runs a *clone* whose Overrides were edited, the code rebuilds the Hydra
   config with those values.
4. **`task.connect_configuration`**: stores the final resolved config on the task for the record.

Consequences you must remember:

- A setting can only be changed from the ClearML UI, from HPO or from a pipeline if its key is in the
  `tunable` list in `conf/config.yaml`. To make a new setting adjustable, add it there.
- Hydra stays in charge. ClearML never edits the YAML files; it only supplies override values.
- Training metrics reach ClearML through `ClearMLLogger`. The best checkpoint is uploaded automatically
  as a model; `metadata.json` is uploaded after the sweep.

Output of a finished training run:

- `best.ckpt` (the model), `metadata.json` (config, versions, checkpoint hash, best TTA and threshold,
  sweep table), `preflight.json`, `resolved_config.yaml`.
- The sweep tries test-time augmentation (none / flip / d4) and probability thresholds 0.30 to 0.70 on the
  validation split and records the best pair as the inference default.

---

## 8. The two gates

Both end with **exit code 2 and the task marked failed** when they find a problem. In a pipeline this
stops training from starting.

**`resource_check`** (`conf/resource/default.yaml`): builds the model with random weights (no downloads),
times a 1024x1024 tile at `cpu_request` (4) and `cpu_limit` (8) threads, and measures peak memory. It fails
if memory exceeds `memory_limit_gib` (8), or if p95 latency exceeds `max_latency_s` (off by default).
It prints warnings when it is not inside a container with the real limits.

> **Important:** the result is only meaningful on a machine that enforces your real production limits
> (a Kubernetes pod or `docker --cpus=8 --memory=8g`). On Kaggle (4 CPUs, 30 GB) it only tells you how heavy
> the model is. Measured values: about 2.1 to 2.3 GiB peak, about 2.1 s per tile on Kaggle's CPU.
> The Docker / Kubernetes enforcement itself is **Untested**.

**`domain_check`** (`conf/preflight/default.yaml`): compares train and val images (per-channel statistics,
k-nearest-neighbour mixing score) and the building-pixel ratio per tile. The domain-shift check is a gate;
label balance only reports. Measured: shift score 0.004 (0 means perfectly mixed).

---

## 9. Base tasks: the templates every remote run copies

Agents and pipelines never run your laptop's process. They clone a **base task**. Three exist in project
`building-segmentation`:

| Name | State | Created by |
|---|---|---|
| `resource-check` | completed | running `python -m src.resource_check` once |
| `domain-check` | completed | running `python -m src.domain_check` once |
| `Unet-resnet34` | **draft** | a clone of a smoke-run, with a bare entry point and default settings |
| `serve-gate` | completed | running `python -m src.serve_gate` once (no arguments) |

Create or refresh a base task:

```bash
python -m src.resource_check      # no arguments!
python -m src.domain_check        # no arguments!
```

**Rules that cost hours when ignored:**

1. **A base task is pinned to one git commit.** After you push new code, existing base tasks still run
   the *old* code. Re-register: archive the old task and run the script again (the new run records the new
   commit). For the draft train task, update it in place:
   ```python
   from clearml import Task
   Task.get_task(task_id="<draft id>").set_script(commit="<new commit sha>")
   ```
2. **Never register a base task with extra command-line arguments.** ClearML records them in the entry
   point *and* in a parameter called `Args/overrides`, and every clone inherits them. A smoke test with
   `sweep.enabled=false` once silently disabled the sweep in later pipeline runs. Check with:
   ```python
   Task.get_task(task_id="...").get_parameters().get("Args/overrides")   # should be None or ['hydra.mode=RUN']
   ```
   and remove a stale one with `task.delete_parameter("Args/overrides")`.
3. Push the code to git **before** registering. The agent clones the repository from GitHub; uncommitted
   local changes do not exist for it. (Repo: https://github.com/CheeKuang/clearml-kaggle, public.)
4. `Task.set_parameters(...)` **replaces all parameters**. Use `update_parameters(...)` to change a few.

---

## 10. Running on a remote GPU (Kaggle)

### 10.1 Start the agent (every session)

1. New Kaggle notebook. Settings: **Accelerator = GPU**, **Internet = On**.
2. Add-ons → Secrets: create **`CLEARML_KEY`** (access key) and **`CLEARML_SECRET`** (secret key) from your
   app.clear.ml credentials, and attach both to the notebook.
3. Cell 1:
   ```python
   !rm -rf /kaggle/working/repo && git clone https://github.com/CheeKuang/clearml-kaggle.git /kaggle/working/repo
   ```
4. Cell 2 (must be last; it keeps running):
   ```python
   %run /kaggle/working/repo/kaggle/start_agent.py
   ```
5. In ClearML (Workers & Queues) a worker such as `d333e505fc37:gpu0` should appear on queue `kaggle`.

**The agent dies when the Kaggle session ends** (browser tab closed, idle timeout, time limit). Queued
tasks then wait forever. Keep the tab open and active. After a restart the worker gets a new name and a
cold cache, so the first task is slower (about 5 extra minutes for the package install).

### 10.2 Send one training run to the agent

```python
from clearml import Task
base = Task.get_task(project_name="building-segmentation", task_name="Unet-resnet34")
t = Task.clone(source_task=base, name="my-run", parent=base.id)
t.update_parameters({
    "Overrides/data.clearml_dataset_id": "<dataset id>",
    "Overrides/train.epochs": 30,
    "Overrides/train.batch_size": 4,
})
Task.enqueue(t, queue_name="kaggle")
print(t.get_output_log_web_page())
```

Or edit the same values in the web UI: open the draft → **Configuration → Hyperparameters → Overrides**,
change, right-click → Enqueue.

### 10.3 Timings seen on a Kaggle T4 (batch size 4)

| Stage | Time |
|---|---|
| Package install (cold) | 3 to 6 min |
| Dataset download (2.25 GiB) | about 20 s |
| One epoch | about 2.5 min |
| TTA x threshold sweep | about 5 min |
| Whole 2-epoch training task | about 14 min |

A 30-epoch run is therefore about 75 minutes plus the sweep (an estimate, not measured).

---

## 11. The pipeline (one model per class)

Production serves **one model per class** (building, road, water, vegetation). The pipeline mirrors that.
Configured in `pipeline/pipeline.yaml` (queues, the class list with architecture and encoder per class, the dataset
id); built in `src/pipeline.py` as a plain list of steps (`plan()`), so it can be printed before it runs.

```
domain_check ──┐                  ┌─> train_building ──┐
               ├─ both gates ─────┼─> train_road       ├─> serve_gate_post
serve_gate_pre ┘                  ├─> train_water      │
(untrained models, all together)  └─> train_vegetation ┘   (the real ONNX models, all together)
```

```bash
python -m src.pipeline --dry-run                       # print every step and its overrides, touch nothing
python -m src.pipeline --local --queue kaggle \
    --train-param train.epochs=2 --train-param train.batch_size=4
python -m src.pipeline --local --classes road,water    # only some classes
```

Options: `--queue Q`, `--classes a,b`, `--param k=v` (every step), `--train-param k=v`, `--serve-param k=v`,
`--dry-run`, `--local`, `--config path`. The pipeline refuses to start until `params.data.clearml_dataset_id`
holds the id of the multi-class dataset.

What happens:

1. `domain_check` (image statistics plus the label balance of every class) and `serve_gate_pre` run first.
2. `serve_gate_pre` exports **untrained** models of the configured architectures to ONNX and measures them together
   on the production inference path (section 11.1). If they cannot fit, no GPU time is spent.
3. When both pass, one `train_<class>` step per class is created (`data=<class>`, own architecture and encoder).
   Each trains, runs the sweep and **exports its ONNX model** with a threshold, as artifacts `onnx` and `onnx_meta`.
4. `serve_gate_post` collects the ONNX artifacts of the train steps (through the `pipe: <id>` tag) and measures
   the real models together.
5. A failed gate (exit code 2) fails its step; nothing after it is created.

Where to look in the web UI: the **Pipelines** page. Step tasks are tagged `pipe: <controller id>`.

**Verified:** `plan()` output, each new override type through a simulated agent run (`data=water`, a spec string
with `=`, `,` and `:`), one-class training with ONNX export, both gate modes locally.
**Untested:** the per-class pipeline itself on Kaggle (it needs the code pushed, the base tasks registered and the
multi-class dataset uploaded), and `serve.source=tasks` against real pipeline steps.

### 11.1 The production server and what "matching" it means

The production service (Chisel, repository `lectern-models`) is a LitServe app running four ONNX models with
onnxruntime. Facts taken from its code:

- Input: a base64 image (RGBA, the alpha channel marks the visible area) and `total_area`. Output per category: an
  area and a coloured PNG mask, plus an `open` category for unclaimed visible pixels.
- Images larger than 1280 are cut into 1280x1280 tiles (edges padded), each tile goes through all four models in a
  thread pool of 2; every model's onnxruntime session uses 2 threads.
- Preprocessing: letterbox to 1280x1280 (pad value 114), RGB scaled to 0..1, no mean/std. Training uses the same scaling.
- Each class has its own confidence threshold. Overlaps are resolved winner-take-all (masks are 0/255, so ties go to
  the first category in the settings order: vegetation, building, water, road).
- The current models are **YOLO instance-segmentation** models. The models trained in this repository are
  **semantic** (one probability per pixel), so the server needs one small change: the post-processing in
  `serving/postprocess_semantic.py` is a drop-in for `postprocess_yolo_masks` (same signature, same 0/255 result).
  The server code itself has not been modified by this project.

Files and commands:

| Piece | What it does |
|---|---|
| `python -m src.export_onnx` | checkpoint to ONNX (input RGB/255 at 1280, output sigmoid probabilities); checks onnxruntime against PyTorch; writes `<name>.onnx.json` with the class and the best threshold at TTA "none" |
| `serving/postprocess_semantic.py` | the post-processing for the server; `serving/test_postprocess_semantic.py` tests its geometry against the server's own letterbox |
| `python -m src.serve_check --models <folder>` | loads all models in ONE process with the server's session options and thread pool and times a request of N tiles; fails above the memory or latency limit |
| `python -m src.serve_gate` | the same as a ClearML Task, with `serve.source` = `random` / `tasks` / `folder`; it measures in a fresh process so exporting does not inflate the memory reading |

Measured on an M3 Mac (not the production container, not real weights), one 1280x1280 tile, four models, pool of 2:

| Models | Request time | Peak memory |
|---|---|---|
| current YOLO models (noise input) | 5.97 s | 4.1 GiB |
| untrained Unet-resnet34, one per class | 4.4 s | 2.8 GiB |

The server's pool of 2 keeps only 4 cores busy; a pool of 4 used 8 cores for 3.1 s but 4.8 GiB. These numbers are only a
rough comparison.

Not covered by the gate: image decoding, alpha masking, winner-take-all, colouring and PNG encoding of the real server.
Also note the server returns **empty masks without an error** when a model fails, times out or the image is
larger than `PIXEL_SIZE_LIMIT`.

---

## 12. Changing things

| I want to... | Do this |
|---|---|
| Try another architecture or encoder | Add a YAML in `conf/architecture/` or `conf/encoder/`; run with `architecture=<name>` or `--param architecture=<name>`. Note `resnet50_torchgeo_s2` needs 13-channel Sentinel-2 data; your LoveDA tiles have 3. |
| Change a training setting | Edit `conf/config.yaml`, or pass `key=value`. For a remote run the key must be in the `tunable` list. |
| Change which classes are predicted | Edit `classes:` in `conf/data/loveda.yaml` (names and mask ids). No code change; the data must contain those ids. |
| Use a new dataset version | `python -m src.dataset --path <folder> --name loveda-buildings --parent <old id>`, then use the new id. |
| Search hyperparameters | Edit `hpo/space.yaml`, then `python -m src.hpo --mode clearml`. Uses clones of the base task. **Untested** in this project. |
| Compare several models in one pipeline | Not built. Idea: one `resource_check` per model, then train those that pass. |

After any code change: push to git, then re-register or re-pin the base tasks (section 9).

---

## 13. Troubleshooting (real problems from this project)

| Symptom | Cause | Fix |
|---|---|---|
| `No images under data/loveda_split/train/images` on the agent | Overrides were not applied on remote runs (a comparison bug) | Fixed in `clearml_utils.py`. Make sure the base task points at a commit that includes the fix. |
| A float override is ignored (e.g. `0.3` stays `8`) | ClearML casts values to the type of the default | Fixed: overrides are connected as text. |
| Pipeline step ran with `sweep.enabled=false`, 0 workers | Stale `Args/overrides` on the base task | Section 9, rule 2. |
| Task hangs at `import triton`, memory flat, no output | `segmentation_models_pytorch` imported after ClearML's import hooks | `resource_check.py` imports it at the top, before `Task.init`. Keep that order. |
| Peak memory shows `0.0` | `ru_maxrss` is KiB on Linux (was divided instead of multiplied) | Fixed. |
| Config group `conf/data` missing in the repo | `.gitignore` had `data/`, which also matched `conf/data/` | Now `/data/`. |
| Queue has tasks, nothing runs | No worker: the Kaggle session ended | Restart the notebook cells (section 10.1). Stop stranded tasks and remove stale queue entries. |
| `BackendError: No user secrets exist ... CLEARML_API_HOST` | Wrong secret names | Secrets must be `CLEARML_KEY` and `CLEARML_SECRET`, attached to the notebook. |
| Local training uses 17 GB and the Mac crawls | Full-size tiles, batch 4, plus other heavy jobs | Lower `train.batch_size`; close apps; consider cropping tiles. |
| `ValueError: None of the class ids ... occur in ... sampled masks` | The class list is set but the masks are binary 0/255 (an older dataset) | Use `data=buildings`, or upload a dataset made with `prepare_data --mode labels`. |
| Python segfaults in `cv2.resize` on a Mac | OpenCV 4.14 on macOS arm64 | Use `opencv-python-headless==4.13.0.92`, the version the server locks. |
| `zsh: no matches found: sweep.tta=[none]` | zsh treats brackets as a pattern | Quote the argument. |
| The `Task` name in a pipeline is not found | Names are looked up as `<project>/<task name>` and must match | Names used: `resource-check`, `domain-check`, `Unet-resnet34`. |

Debug trick: to test the remote path (Overrides) without a GPU, simulate an agent locally:
```python
Task.debug_simulate_remote_task(task_id=<clone id>)   # then run the script's main
```

---

## 14. Going offline (Untested)

You plan to move this to an environment with no internet and a GPU. See `offline/README.md`. Summary:

| Needs internet today | Replacement |
|---|---|
| ClearML server | Self-hosted ClearML server (docker-compose); change hosts in `clearml.conf` |
| Code from GitHub | An internal git server the agent can clone from |
| `pip install` at task start | A prebuilt image or a package mirror (`offline/Dockerfile`, `offline/build_wheelhouse.sh`) |
| Encoder weights from huggingface.co | `encoder=resnet34_local` after `python -m src.export_weights` (this part is **Verified**: the model builds with the network off) |
| Dataset | Re-upload with `src.dataset` to the internal server |
| Base tasks | Re-register on the new server |

Known pitfall: ClearML records the package versions of the machine that registered the task, and the agent
installs exactly those. The offline image must match, or requirements must be pinned from a file
(`Task.force_requirements_env_freeze`). Not done yet.

Task history on the hosted server does not move automatically.

---

## 15. Production monitoring

ClearML does not see model accuracy in production by itself, because a deployed model has no ground truth.
ClearML Serving can record latency, request counts and custom input/output statistics (for example the
predicted building ratio), which are useful early-warning signals. Accuracy needs a labelled sample scored
periodically. Nothing of this exists in the repo yet.

---

## 16. Cheat sheet

```bash
# local smoke test, no ClearML (add data=buildings data.root=data/loveda_split for one binary mask)
python -m src.train clearml.enabled=false data.root=data/loveda_multi train.epochs=1 \
    train.num_workers=0 train.batch_size=4 sweep.enabled=false

# prepare + upload data (several classes)
python -m src.prepare_data --src data/loveda --out data/loveda_multi
python -m src.dataset --path data/loveda_multi --name loveda-multi

# register base tasks (no arguments, after pushing code)
python -m src.resource_check
python -m src.domain_check

# per-class gated pipeline (agent must be running; see section 11)
python -m src.pipeline --dry-run
python -m src.pipeline --local --queue kaggle --train-param train.epochs=2 --train-param train.batch_size=4

# production artifacts
python -m src.export_onnx --checkpoint best.ckpt --metadata metadata.json --out models/vegetation.onnx
python -m src.serve_check --models models --tiles 1

# hyperparameter search
python -m src.hpo --mode clearml
```

Before asking for help, collect: the task URL, the first error in its console log, the commit the task is
pinned to, and its `Args/overrides` value.
