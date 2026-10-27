#!/bin/sh
# Run on a machine WITH internet. Collects every wheel the training image needs into ./wheelhouse so the
# offline environment can install with `pip install --no-index --find-links wheelhouse`.
# UNTESTED: the platform/Python flags must match the offline image (Linux x86_64, Python 3.12 here).
set -eu
cd "$(dirname "$0")/.."
pip download -r requirements.txt clearml-agent -d offline/wheelhouse \
    --python-version 3.12 --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_27_x86_64 --platform manylinux_2_24_x86_64 --platform any
du -sh offline/wheelhouse
