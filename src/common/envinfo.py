"""Container and process facts for the serving gate (no torch import, safe everywhere)."""
from __future__ import annotations

import resource
import sys
from pathlib import Path

GIB = 1024 ** 3


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_limits() -> tuple[float | None, float | None]:
    """(cpus, mem_gib) this container is limited to; None = unlimited or not in a cgroup (v2, then v1)."""
    cpu = mem = None
    if cm := _read("/sys/fs/cgroup/cpu.max"):                       # v2: "<quota|max> <period>"
        quota, period = cm.split()
        cpu = None if quota == "max" else int(quota) / int(period)
    else:
        q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q) > 0:
            cpu = int(q) / int(p)
    mm = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if mm and mm != "max" and int(mm) < 1 << 60:
        mem = int(mm) / GIB
    return cpu, mem


def peak_rss_gib() -> float:
    """Peak resident memory of THIS process image.
    Linux: VmHWM, which restarts at exec (ru_maxrss is inherited by a child from its parent, so a gate that exports in
    one process and measures in a subprocess would read the exporter's peak). macOS: ru_maxrss, in bytes there."""
    status = _read("/proc/self/status")
    for line in (status or "").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / GIB                 # kB
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS, KiB on Linux
    return r * (1 if sys.platform == "darwin" else 1024) / GIB
