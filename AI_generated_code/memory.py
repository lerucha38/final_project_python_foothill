"""Track peak memory and runtime for benchmark steps."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil


@dataclass
class RunStats:
    runtime_sec: float = 0.0
    peak_rss_mb: float = 0.0
    returncode: int = 0
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _monitor_peak_rss(pid: int, interval: float = 0.2) -> float:
    peak = 0.0
    try:
        proc = psutil.Process(pid)
        while proc.is_running():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass
    return peak / (1024 * 1024)


def run_subprocess(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> RunStats:
    stats = RunStats()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    peak_holder: list[float] = []

    def _watch() -> None:
        peak_holder.append(_monitor_peak_rss(proc.pid))

    watcher = threading.Thread(target=_watch, daemon=True)
    start = time.perf_counter()
    watcher.start()
    proc.wait()
    watcher.join(timeout=5)
    stats.runtime_sec = time.perf_counter() - start
    stats.returncode = proc.returncode
    stats.peak_rss_mb = peak_holder[0] if peak_holder else 0.0
    return stats


def run_callable(func: Callable[[], None], process: psutil.Process | None = None) -> RunStats:
    stats = RunStats()
    proc = process or psutil.Process()
    peak = proc.memory_info().rss

    def _watch() -> None:
        nonlocal peak
        while not done["flag"]:
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.2)

    done = {"flag": False}
    watcher = threading.Thread(target=_watch, daemon=True)
    start = time.perf_counter()
    watcher.start()
    try:
        func()
    except Exception as exc:
        stats.error = str(exc)
        stats.returncode = 1
    finally:
        done["flag"] = True
        watcher.join(timeout=5)
        stats.runtime_sec = time.perf_counter() - start
        stats.peak_rss_mb = peak / (1024 * 1024)
    return stats
