"""Statistics tracking: latency, throughput, GPU monitoring"""

import threading
import time
import statistics
import warnings
from collections import deque
from typing import Optional


class LatencyTracker:
    def __init__(self):
        self._samples = []

    def record(self, latency_ms: float):
        self._samples.append(latency_ms)

    def stats(self) -> dict:
        if not self._samples:
            return {}
        s = sorted(self._samples)
        n = len(s)

        def percentile(p):
            idx = int(n * p / 100)
            return s[min(idx, n - 1)]

        return {
            "mean": statistics.mean(s),
            "median": statistics.median(s),
            "std": statistics.stdev(s) if n > 1 else 0.0,
            "min": s[0],
            "max": s[-1],
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
            "count": n,
        }


class ThroughputTracker:
    """Tracks samples/sec over a rolling window"""

    def __init__(self, window: int = 50):
        self._window = deque(maxlen=window)

    def record(self, samples: int, elapsed_sec: float):
        if elapsed_sec > 0:
            self._window.append(samples / elapsed_sec)

    def current_throughput(self) -> float:
        if not self._window:
            return 0.0
        return statistics.mean(self._window)

    def stats(self) -> dict:
        if not self._window:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        vals = list(self._window)
        return {
            "mean": statistics.mean(vals),
            "min": min(vals),
            "max": max(vals),
        }


def _init_nvml(device):
    """
    Load nvidia-ml-py (pip: nvidia-ml-py). It still exposes the 'pynvml'
    namespace. Suppress the FutureWarning torch emits when it also tries to
    import the old 'pynvml' package name.
    Returns (handle, module) or (None, None).
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import pynvml as _pynvml
        _pynvml.nvmlInit()
        device_idx = getattr(device, "index", 0) or 0
        handle = _pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        return handle, _pynvml
    except Exception:
        return None, None


class GPUMonitor:
    """Background thread that polls GPU stats via nvidia-ml-py or torch"""

    def __init__(self, device, poll_interval: float = 0.5):
        self._device = device
        self._poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._utilizations = []
        self._memory_mbs = []
        self._temps = []
        self._current = {}

        self._nvml_handle, self._pynvml = _init_nvml(device)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def current(self) -> dict:
        return dict(self._current)

    def _poll(self):
        import torch
        while not self._stop.is_set():
            sample = {}
            try:
                if self._nvml_handle and self._pynvml:
                    nvml = self._pynvml
                    util = nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(
                            self._nvml_handle,
                            nvml.NVML_TEMPERATURE_GPU,
                        )
                        self._temps.append(float(temp))
                        sample["temp_c"] = float(temp)
                    except Exception:
                        pass
                    util_val = float(util.gpu)
                    mem_mb = mem.used / (1024 ** 2)
                    self._utilizations.append(util_val)
                    self._memory_mbs.append(mem_mb)
                    sample["utilization"] = util_val
                    sample["memory_mb"] = mem_mb
                elif torch.cuda.is_available() and str(self._device) != "cpu":
                    mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    self._memory_mbs.append(mem_mb)
                    sample["memory_mb"] = mem_mb
            except Exception:
                pass
            self._current = sample
            self._stop.wait(self._poll_interval)

    def stats(self) -> dict:
        result = {}
        if self._memory_mbs:
            result["memory_peak_mb"] = max(self._memory_mbs)
            result["memory_mean_mb"] = statistics.mean(self._memory_mbs)
        if self._utilizations:
            result["utilization_mean"] = statistics.mean(self._utilizations)
            result["utilization_max"] = max(self._utilizations)
        if self._temps:
            result["temp_max"] = max(self._temps)
            result["temp_mean"] = statistics.mean(self._temps)
        return result
