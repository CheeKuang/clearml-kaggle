# Agent / training image for the offline environment. Build on a machine WITH internet, then ship the image.
#   sh offline/build_wheelhouse.sh && python -m src.export_weights
#   docker build -f offline/Dockerfile -t chisel-train:1 .
# UNTESTED. Needs the NVIDIA driver + container toolkit on the offline host; torch's pip wheels bring the CUDA libs.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY offline/wheelhouse /wheelhouse
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links=/wheelhouse -r /tmp/requirements.txt clearml-agent

# pretrained encoder weights (conf/encoder/resnet34_local.yaml reads ./weights relative to the repo; the agent
# clones the repo at run time, so mount or copy this next to it, or point pretrain.path here)
COPY weights /opt/weights

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
