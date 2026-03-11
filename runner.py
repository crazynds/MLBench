"""
Benchmark and Stress runners
"""

import json
import time
import signal
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from utils.device import resolve_device, get_device_info
from utils.stats import LatencyTracker, ThroughputTracker, GPUMonitor
from utils.downloader import ensure_model_and_dataset
from models import get_model_handler


@dataclass
class BenchmarkResult:
    model: str
    device: str
    precision: str
    batch_size: int
    warmup_iterations: int
    benchmark_iterations: int
    # Latency (ms)
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_std_ms: float = 0.0
    # Throughput
    throughput_samples_per_sec: float = 0.0
    throughput_batches_per_sec: float = 0.0
    # GPU
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_mean: float = 0.0
    # Timing
    total_time_sec: float = 0.0


@dataclass
class StressResult:
    model: str
    device: str
    precision: str
    batch_size: int
    duration_sec: int
    actual_duration_sec: float = 0.0
    total_iterations: int = 0
    total_samples: int = 0
    throughput_mean: float = 0.0
    throughput_min: float = 0.0
    throughput_max: float = 0.0
    errors: int = 0
    gpu_memory_peak_mb: float = 0.0
    gpu_temp_max: float = 0.0
    gpu_utilization_mean: float = 0.0


class BenchmarkRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model_name = args.model
        self.batch_size = args.batch_size
        self.warmup = args.warmup
        self.iterations = args.iterations
        self.precision = args.precision
        self.device = resolve_device(args.device)
        self.output = args.output

    def run(self):
        logger = self.logger
        logger.info(f"🔧 Benchmark | Model: {self.model_name} | Device: {self.device} | Precision: {self.precision}")

        # Ensure model & dataset exist
        if not self.args.no_download:
            ensure_model_and_dataset(self.model_name, logger)

        # Load model handler
        handler = get_model_handler(
            self.model_name,
            device=self.device,
            precision=self.precision,
            batch_size=self.batch_size,
            logger=logger,
        )

        logger.info("📦 Loading model...")
        handler.load()

        logger.info("📊 Preparing dataset...")
        handler.prepare_data()

        gpu_monitor = GPUMonitor(self.device)
        gpu_monitor.start()

        latency_tracker = LatencyTracker()

        # ── Warmup ────────────────────────────────────────────────────────────
        logger.info(f"🔥 Warming up ({self.warmup} iterations)...")
        for i in range(self.warmup):
            handler.run_inference()

        logger.info(f"🏁 Running benchmark ({self.iterations} iterations)...")

        total_start = time.perf_counter()
        for i in range(self.iterations):
            t0 = time.perf_counter()
            handler.run_inference()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            latency_tracker.record(latency_ms)

            if (i + 1) % max(1, self.iterations // 10) == 0:
                pct = (i + 1) / self.iterations * 100
                logger.info(f"  Progress: {i+1}/{self.iterations} ({pct:.0f}%) | "
                            f"Last latency: {latency_ms:.2f}ms")

        total_elapsed = time.perf_counter() - total_start
        gpu_monitor.stop()

        # ── Results ───────────────────────────────────────────────────────────
        stats = latency_tracker.stats()
        throughput_samples = (self.iterations * self.batch_size) / total_elapsed
        throughput_batches = self.iterations / total_elapsed
        gpu_stats = gpu_monitor.stats()

        result = BenchmarkResult(
            model=self.model_name,
            device=str(self.device),
            precision=self.precision,
            batch_size=self.batch_size,
            warmup_iterations=self.warmup,
            benchmark_iterations=self.iterations,
            latency_mean_ms=stats["mean"],
            latency_median_ms=stats["median"],
            latency_p90_ms=stats["p90"],
            latency_p95_ms=stats["p95"],
            latency_p99_ms=stats["p99"],
            latency_min_ms=stats["min"],
            latency_max_ms=stats["max"],
            latency_std_ms=stats["std"],
            throughput_samples_per_sec=throughput_samples,
            throughput_batches_per_sec=throughput_batches,
            gpu_memory_peak_mb=gpu_stats.get("memory_peak_mb", 0),
            gpu_utilization_mean=gpu_stats.get("utilization_mean", 0),
            total_time_sec=total_elapsed,
        )

        self._print_results(result)

        if self.output:
            self._save_results(result)

        handler.cleanup()

    def _print_results(self, r: BenchmarkResult):
        print("\n" + "=" * 60)
        print(f"  📈 BENCHMARK RESULTS — {r.model.upper()}")
        print("=" * 60)
        print(f"  Device         : {r.device}")
        print(f"  Precision      : {r.precision}")
        print(f"  Batch size     : {r.batch_size}")
        print(f"  Iterations     : {r.benchmark_iterations} (warmup: {r.warmup_iterations})")
        print("-" * 60)
        print(f"  Latency (ms):")
        print(f"    Mean         : {r.latency_mean_ms:.2f}")
        print(f"    Median       : {r.latency_median_ms:.2f}")
        print(f"    Std Dev      : {r.latency_std_ms:.2f}")
        print(f"    Min          : {r.latency_min_ms:.2f}")
        print(f"    Max          : {r.latency_max_ms:.2f}")
        print(f"    P90          : {r.latency_p90_ms:.2f}")
        print(f"    P95          : {r.latency_p95_ms:.2f}")
        print(f"    P99          : {r.latency_p99_ms:.2f}")
        print("-" * 60)
        print(f"  Throughput:")
        print(f"    Samples/sec  : {r.throughput_samples_per_sec:.2f}")
        print(f"    Batches/sec  : {r.throughput_batches_per_sec:.2f}")
        print("-" * 60)
        if r.gpu_memory_peak_mb > 0:
            print(f"  GPU:")
            print(f"    Peak Memory  : {r.gpu_memory_peak_mb:.1f} MB")
            print(f"    Mean Util    : {r.gpu_utilization_mean:.1f}%")
        print(f"  Total Time     : {r.total_time_sec:.2f}s")
        print("=" * 60 + "\n")

    def _save_results(self, result: BenchmarkResult):
        path = Path(self.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        self.logger.info(f"💾 Results saved to {path}")


class StressRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model_name = args.model
        self.duration = args.duration
        self.batch_size = args.batch_size
        self.precision = args.precision
        self.device = resolve_device(args.device)
        self.interval = args.interval
        self.output = args.output
        self._stop_event = threading.Event()

    def run(self):
        logger = self.logger
        logger.info(f"💪 Stress Test | Model: {self.model_name} | Duration: {self.duration}s | Device: {self.device}")

        # Handle Ctrl+C gracefully
        def _sig_handler(sig, frame):
            logger.warning("\n⚠️  Interrupted by user. Stopping stress test...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        if not self.args.no_download:
            ensure_model_and_dataset(self.model_name, logger)

        handler = get_model_handler(
            self.model_name,
            device=self.device,
            precision=self.precision,
            batch_size=self.batch_size,
            logger=logger,
        )

        logger.info("📦 Loading model...")
        handler.load()

        logger.info("📊 Preparing dataset...")
        handler.prepare_data()

        gpu_monitor = GPUMonitor(self.device)
        gpu_monitor.start()

        throughput_tracker = ThroughputTracker()
        total_iterations = 0
        total_errors = 0

        start_time = time.perf_counter()
        last_report = start_time

        logger.info(f"🚀 Starting stress test for {self.duration}s... (Press Ctrl+C to stop)\n")

        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.duration:
                break

            try:
                t0 = time.perf_counter()
                handler.run_inference()
                t1 = time.perf_counter()

                iter_time = t1 - t0
                throughput_tracker.record(self.batch_size, iter_time)
                total_iterations += 1

                # Periodic report
                now = time.perf_counter()
                if now - last_report >= self.interval:
                    remaining = max(0, self.duration - elapsed)
                    tp = throughput_tracker.current_throughput()
                    gpu_info = gpu_monitor.current()
                    mem_str = f" | GPU Mem: {gpu_info.get('memory_mb', 0):.0f}MB" if gpu_info else ""
                    util_str = f" | GPU Util: {gpu_info.get('utilization', 0):.0f}%" if gpu_info else ""
                    print(
                        f"\r  ⏱  {elapsed:6.1f}s / {self.duration}s | "
                        f"Iter: {total_iterations:6d} | "
                        f"Throughput: {tp:8.2f} samples/s{mem_str}{util_str} | "
                        f"Remaining: {remaining:.0f}s   ",
                        end="",
                        flush=True,
                    )
                    last_report = now

            except Exception as e:
                total_errors += 1
                logger.error(f"\n❌ Error during inference: {e}")
                if total_errors > 10:
                    logger.error("Too many errors, stopping stress test.")
                    break

        print()  # newline after progress

        actual_duration = time.perf_counter() - start_time
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.stats()
        tp_stats = throughput_tracker.stats()

        result = StressResult(
            model=self.model_name,
            device=str(self.device),
            precision=self.precision,
            batch_size=self.batch_size,
            duration_sec=self.duration,
            actual_duration_sec=actual_duration,
            total_iterations=total_iterations,
            total_samples=total_iterations * self.batch_size,
            throughput_mean=tp_stats.get("mean", 0),
            throughput_min=tp_stats.get("min", 0),
            throughput_max=tp_stats.get("max", 0),
            errors=total_errors,
            gpu_memory_peak_mb=gpu_stats.get("memory_peak_mb", 0),
            gpu_temp_max=gpu_stats.get("temp_max", 0),
            gpu_utilization_mean=gpu_stats.get("utilization_mean", 0),
        )

        self._print_results(result)

        if self.output:
            self._save_results(result)

        handler.cleanup()

    def _print_results(self, r: StressResult):
        print("\n" + "=" * 60)
        print(f"  💪 STRESS RESULTS — {r.model.upper()}")
        print("=" * 60)
        print(f"  Device           : {r.device}")
        print(f"  Precision        : {r.precision}")
        print(f"  Batch size       : {r.batch_size}")
        print(f"  Target duration  : {r.duration_sec}s")
        print(f"  Actual duration  : {r.actual_duration_sec:.1f}s")
        print("-" * 60)
        print(f"  Total Iterations : {r.total_iterations}")
        print(f"  Total Samples    : {r.total_samples}")
        print(f"  Errors           : {r.errors}")
        print("-" * 60)
        print(f"  Throughput (samples/s):")
        print(f"    Mean           : {r.throughput_mean:.2f}")
        print(f"    Min            : {r.throughput_min:.2f}")
        print(f"    Max            : {r.throughput_max:.2f}")
        if r.gpu_memory_peak_mb > 0:
            print("-" * 60)
            print(f"  GPU:")
            print(f"    Peak Memory    : {r.gpu_memory_peak_mb:.1f} MB")
            print(f"    Max Temp       : {r.gpu_temp_max:.1f}°C")
            print(f"    Mean Util      : {r.gpu_utilization_mean:.1f}%")
        print("=" * 60 + "\n")

    def _save_results(self, result: StressResult):
        path = Path(self.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        self.logger.info(f"💾 Results saved to {path}")
