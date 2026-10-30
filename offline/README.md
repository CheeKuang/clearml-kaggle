# Offline replication checklist

Status: the local-weights path is tested (model builds with the network off). Everything else here is
written but NOT run: the wheelhouse, the Dockerfile and the self-hosted server are untested.

| Needs internet today | Offline replacement |
|---|---|
| ClearML server (app.clear.ml) | self-hosted ClearML server (docker-compose), `clearml.conf` hosts changed |
| code clone from GitHub | internal git server; the agent clones from it |
| `pip install` at task start | image built by `offline/Dockerfile` from `offline/build_wheelhouse.sh` |
| encoder weights from huggingface.co | `encoder=resnet34_local` + `python -m src.export_weights` (copy `weights/`) |
| dataset | `python -m src.data.upload --path <split folder>` against the internal server |
| base tasks | run `src.gates.domain_check`, `src.gates.serve_gate`, `src.training.train` once on the internal server |

Known pitfall: ClearML records the package versions of the machine that registered the task (the Mac) and the
agent installs exactly those. The offline image must contain the same versions, or the task must be
registered from a machine with the same environment, or requirements pinned from a file
(`Task.force_requirements_env_freeze`). Not done yet.
