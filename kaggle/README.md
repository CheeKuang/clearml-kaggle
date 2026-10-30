# Running the agent on a Kaggle GPU

1. New Kaggle notebook: Accelerator = GPU, Internet = On.
2. Add-ons -> Secrets: `CLEARML_KEY` and `CLEARML_SECRET` (app.clear.ml credentials), attached to the notebook.
3. Cell 1: `!rm -rf /kaggle/working/repo && git clone https://github.com/CheeKuang/clearml-kaggle.git /kaggle/working/repo`
4. Cell 2 (last, it keeps running): `%run /kaggle/working/repo/kaggle/start_agent.py`
5. A worker such as `<host>:gpu0` appears on queue `kaggle` (ClearML -> Workers & Queues).

Then start runs from your machine, e.g. `python -m src.orchestration.pipeline --local --queue kaggle ...`
(docs/GUIDE.md, sections 2.5 to 2.7). Verified: the per-class pipeline has run on this agent.

The agent dies when the Kaggle session ends (tab closed, idle timeout, time limit); queued tasks then wait.
