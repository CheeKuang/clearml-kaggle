"""Start a clearml-agent inside a Kaggle notebook so the `kaggle` queue runs on Kaggle's GPU.

Notebook settings: Accelerator = GPU, Internet = On.
Add-ons > Secrets (the notebook must be attached to both):
  CLEARML_KEY      access key of the credentials created on app.clear.ml
  CLEARML_SECRET   secret key of the same credentials
  GIT_USER, GIT_TOKEN        (optional, only if the code repo is private)

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
os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"        # hosted server; change for a self-hosted one
os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml"
os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
os.environ["CLEARML_API_ACCESS_KEY"] = secrets.get_secret("CLEARML_KEY").strip()
os.environ["CLEARML_API_SECRET_KEY"] = secrets.get_secret("CLEARML_SECRET").strip()
try:                                                    # private repo: the agent clones it with these
    os.environ["CLEARML_AGENT_GIT_USER"] = secrets.get_secret("GIT_USER")
    os.environ["CLEARML_AGENT_GIT_PASS"] = secrets.get_secret("GIT_TOKEN")
except Exception:
    print("no GIT_USER / GIT_TOKEN secret: assuming a public repo")

# Kaggle already ships a CUDA build of torch: let the agent's venv reuse it instead of reinstalling.
Path("~/clearml.conf").expanduser().write_text("agent {\n  package_manager { system_site_packages: true }\n}\n")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "clearml-agent"])
subprocess.run(["clearml-agent", "daemon", "--queue", QUEUE, "--gpus", "0"], check=True)
