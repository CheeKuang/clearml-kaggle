"""Start a clearml-agent inside a Kaggle notebook so the `kaggle` queue runs on Kaggle's GPU.

Notebook settings: Accelerator = GPU, Internet = On.
Add-ons > Secrets (all required unless noted):
  CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY
  GIT_USER, GIT_TOKEN        (only if the code repo is private)

Last cell of the notebook (it blocks while the agent is running):

    !git clone <repo url> /kaggle/working/repo     # or upload kaggle/start_agent.py directly
    %run /kaggle/working/repo/kaggle/start_agent.py

The agent stops when the Kaggle session ends; queued tasks wait until you start it again.
"""
import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

QUEUE = "kaggle"

secrets = UserSecretsClient()
for key in ("CLEARML_API_HOST", "CLEARML_WEB_HOST", "CLEARML_FILES_HOST",
            "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
    os.environ[key] = secrets.get_secret(key)
try:                                                    # private repo: the agent clones it with these
    os.environ["CLEARML_AGENT_GIT_USER"] = secrets.get_secret("GIT_USER")
    os.environ["CLEARML_AGENT_GIT_PASS"] = secrets.get_secret("GIT_TOKEN")
except Exception:
    print("no GIT_USER / GIT_TOKEN secret: assuming a public repo")

# Kaggle already ships a CUDA build of torch: let the agent's venv reuse it instead of reinstalling.
Path("~/clearml.conf").expanduser().write_text("agent {\n  package_manager { system_site_packages: true }\n}\n")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "clearml-agent"])
subprocess.run(["clearml-agent", "daemon", "--queue", QUEUE, "--gpus", "0"], check=True)
