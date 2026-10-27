"""Container / process facts shared by the resource checks (no torch import, so it is safe everywhere)."""
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
    """(cpus, mem_gib) the container is actually limited to; None = unlimited / not in a cgroup."""
    cpu = mem = None
    cm = _read("/sys/fs/cgroup/cpu.max")                       # cgroup v2: "<quota|max> <period>"
    if cm:
        quota, period = cm.split()
        cpu = None if quota == "max" else int(quota) / int(period)
    else:                                                      # cgroup v1
        q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q) > 0:
            cpu = int(q) / int(p)
    mm = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if mm and mm != "max" and int(mm) < 1 << 60:
        mem = int(mm) / GIB
    return cpu, mem


def peak_rss_gib() -> float:
    """Peak resident memory of THIS process image.
    Linux: VmHWM from /proc/self/status. It starts again at exec, whereas ru_maxrss is inherited by a child from the
    process that spawned it (a gate that exports in one process and measures in a subprocess would read the exporter's
    peak). macOS has no /proc: ru_maxrss, which is in bytes there."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / GIB             # kB
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS, KiB on Linux
    return r * (1 if sys.platform == "darwin" else 1024) / GIB
