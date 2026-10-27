# Hydra + ClearML + Optuna segmentation pipeline

    pip install -r requirements.txt && clearml-init

    # 1. local run (creates the Task that HPO will clone)
    python -m src.train
    python -m src.train architecture=fpn encoder=resnet50_torchgeo_s2 data.in_channels=13 preflight.domain_shift.mode=diagnostic

    # 2. send to an agent queue (e.g. clearml-agent daemon --queue kaggle --gpus 0 inside a Kaggle notebook)
    python -m src.train clearml.execute_remotely=true clearml.queue=kaggle

    # 3. Optuna HPO (edit hpo/space.yaml first)
    python -m src.hpo --mode clearml     # HyperParameterOptimizer + OptimizerOptuna
    python -m src.hpo --mode manual      # explicit ask/tell + Task.clone / Task.enqueue

New architecture = new YAML in conf/architecture/ ; new encoder = new YAML in conf/encoder/ (no code changes).

## Gated pipeline (resource check + domain check -> train)

    resource_check ─┐
                    ├─> train          (python -m src.pipeline)
    domain_check   ─┘

    # one-time: create the base tasks the pipeline clones (each script registers itself in ClearML)
    python -m src.resource_check                 # model fits the inference envelope? (conf/resource/default.yaml)
    python -m src.domain_check                   # train/val domain shift + label balance (conf/preflight)
    python -m src.train                          # base task for the train step
    # then edit pipeline/pipeline.yaml (queues, dataset id) and:
    python -m src.pipeline

- `resource_check` needs no data. It times a dummy tile at `cpu_request` (4) and `cpu_limit` (8) threads and
  reads peak RSS against `memory_limit_gib`. Point its step at a queue whose agent enforces the real limits
  (k8s pod requests/limits or `docker --cpus=8 --memory=8g`); it prints a warning when no cgroup limit is seen.
- Both checks exit with code 2 and mark the Task failed on a `gate` violation, which stops the pipeline.
  Use `resource.mode=diagnostic` / `preflight.*.mode=diagnostic` to log only.
- Offline: agents need a PyPI mirror or a prebuilt image, and the pretrained encoder weights must be available
  locally (resource_check builds with random weights, so it never downloads).

## Where ClearML is used (handover map)

Every ClearML call is a standard API call; look the class/method up in the ClearML docs by name.

| File | ClearML API | What it does here |
|---|---|---|
| `src/clearml_utils.py` `init_clearml` | `Task.init`, `Task.connect(name="Overrides")`, `Task.connect_configuration`, `Task.execute_remotely` | creates the Task; exposes `clearml.tunable` keys as editable "Overrides" hyperparameters; stores the resolved Hydra config; optionally sends the run to an agent queue |
| `src/clearml_utils.py` `Reporter`, `src/clearml_logger.py` | `Task.get_logger()` -> `Logger.report_scalar / report_single_value / report_table / report_histogram / report_image` | all metrics and plots; Lightning's `self.log` is forwarded by `ClearMLLogger` |
| `src/train.py`, `src/domain_check.py`, `src/resource_check.py` | `Task.upload_artifact`, `Task.mark_failed`, `Task.publish_on_completion` | artifacts (config, checkpoint, metadata, check results); gates fail the Task |
| `src/dataset.py` | `Dataset.create / add_files / upload / finalize`; `Dataset.get(...).get_local_copy()` in `src/data.py` | versioned data; the id goes in `data.clearml_dataset_id` |
| `src/pipeline.py` | `PipelineController.add_step / start / start_locally` | resource_check + domain_check -> train |
| `src/hpo.py` | `HyperParameterOptimizer` + `OptimizerOptuna`; `Task.clone`, `Task.enqueue` (manual mode) | hyperparameter search over clones of the train Task |

Hydra stays the only config authority: ClearML "Overrides" are re-applied on top of it (`recompose`).

A step-by-step guide for people new to ClearML is in [docs/GUIDE.md](docs/GUIDE.md).
