#!/usr/bin/env python3
"""
AI Model Benchmark & Stress Test Tool
Supports: ResNet50, RetinaNet, YOLOv11, Stable Diffusion, DLRM v2, Whisper
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*pynvml.*")

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from runner import BenchmarkRunner, StressRunner
from utils.logger import setup_logger
from utils.device import get_device_info, print_device_info

SUPPORTED_MODELS = {
    "resnet50": "ResNet50 - Image Classification",
    "retinanet": "RetinaNet - Object Detection",
    "yolov11": "YOLOv11 - Object Detection",
    "stable_diffusion": "Stable Diffusion - Image Generation",
    "dlrm_v2": "DLRM v2 - Recommendation System",
    "whisper": "Whisper - Speech Recognition",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 AI Model Benchmark & Stress Test Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py benchmark --model resnet50
  python main.py benchmark --model whisper --batch-size 4 --warmup 10 --iterations 100
  python main.py stress --model yolov11 --duration 300
  python main.py stress --model stable_diffusion --duration 600 --interval 0.5
  python main.py list-models
  python main.py device-info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── benchmark ──────────────────────────────────────────────────────────────
    bench = subparsers.add_parser("benchmark", help="Run throughput & latency benchmark")
    bench.add_argument(
        "--model",
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to benchmark",
    )
    bench.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    bench.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    bench.add_argument("--samples", type=int, default=50, help="Total samples to process; last batch may be partial (default: 50)")
    bench.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Precision (default: fp32)")
    bench.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | cuda:0 (default: auto)")
    bench.add_argument("--output", help="Save results to JSON file")
    bench.add_argument("--no-download", action="store_true", help="Skip model/dataset download check")

    # ── stress ─────────────────────────────────────────────────────────────────
    stress = subparsers.add_parser("stress", help="Run continuous stress test")
    stress.add_argument(
        "--model",
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to stress test",
    )
    stress.add_argument("--duration", type=int, default=60, help="Stress duration in seconds (default: 60, 0 = unlimited)")
    stress.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    stress.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Precision (default: fp32)")
    stress.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | cuda:0 (default: auto)")
    stress.add_argument("--interval", type=float, default=1.0, help="Stats reporting interval in seconds (default: 1.0)")
    stress.add_argument("--output", help="Save results to JSON file")
    stress.add_argument("--no-download", action="store_true", help="Skip model/dataset download check")

    # ── list-models ────────────────────────────────────────────────────────────
    subparsers.add_parser("list-models", help="List all supported models")

    # ── device-info ────────────────────────────────────────────────────────────
    subparsers.add_parser("device-info", help="Show device information")

    return parser.parse_args()


def cmd_list_models():
    print("\n📦 Supported Models:\n")
    for key, desc in SUPPORTED_MODELS.items():
        print(f"  {key:<20} {desc}")
    print()


def main():
    args = parse_args()

    if args.command is None:
        print("❌ No command specified. Use --help for usage information.")
        sys.exit(1)

    if args.command == "list-models":
        cmd_list_models()
        return

    if args.command == "device-info":
        print_device_info()
        return

    logger = setup_logger()

    if args.command == "benchmark":
        runner = BenchmarkRunner(args, logger)
        runner.run()

    elif args.command == "stress":
        runner = StressRunner(args, logger)
        runner.run()


if __name__ == "__main__":
    main()