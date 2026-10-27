# Running the pipeline on a Kaggle GPU

1. New Kaggle notebook: Accelerator = GPU, Internet = On.
2. Add the secrets listed at the top of `start_agent.py` (ClearML credentials of a server Kaggle can reach).
3. Run `start_agent.py` as the last cell. The agent now serves the `kaggle` queue.
4. From your machine:

       python -m src.train clearml.execute_remotely=true clearml.queue=kaggle data.clearml_dataset_id=<id>

   or `python -m src.pipeline` (train step uses the `kaggle` queue in `pipeline/pipeline.yaml`).

Status: written from the ClearML agent docs, not yet run on Kaggle. If the task fails on package install,
check the Execution tab (repo, commit, entry point) and the agent log first.

The ClearML server must be reachable from the internet. `~/clearml.conf` on a dev machine usually points
at `localhost`, which Kaggle cannot reach.
