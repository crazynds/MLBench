# 🚀 AI Model Benchmark & Stress Test Tool

Terminal tool to measure **throughput**, **latency** and run **stress tests** for popular AI models using PyTorch + CUDA.

---

## Supported Models

| Key | Model | Task | Dataset |
|-----|-------|------|---------|
| `resnet50` | ResNet50 | Image Classification | Synthetic ImageNet |
| `retinanet` | RetinaNet ResNet50-FPN v2 | Object Detection | Synthetic COCO |
| `yolov11` | YOLOv11n | Object Detection | Synthetic COCO |
| `stable_diffusion` | Stable Diffusion v1.5 | Text-to-Image | Built-in prompts |
| `dlrm_v2` | DLRM v2 | Recommendation | Synthetic Criteo |
| ` ` | Whisper-base | Speech Recognition | LibriSpeech sample / Synthetic |

---

## Installation

### 1. Install PyTorch with CUDA

```bash
pip install torch=={TORCH-VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu{YOUR-CUDA-VERSIOn}
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Check device info

```bash
python main.py device-info
```

### List supported models

```bash
python main.py list-models
```

### Benchmark

```bash
# Basic benchmark
python main.py benchmark --model resnet50

# Custom batch size, precision and iterations
python main.py benchmark --model resnet50 --batch-size 32 --precision fp16 --iterations 200

# Save results to JSON
python main.py benchmark --model whisper --output results/whisper_bench.json

# Run on specific GPU
python main.py benchmark --model yolov11 --device cuda:1
```

### Stress Test

```bash
# 5-minute stress test
python main.py stress --model resnet50 --duration 300

# indeterminated stress test
python main.py stress --model resnet50 --duration 0

# Stress with fp16 and custom reporting interval
python main.py stress --model stable_diffusion --duration 600 --precision fp16 --interval 2.0

# Save stress results
python main.py stress --model dlrm_v2 --duration 120 --output results/dlrm_stress.json
```

---

## Arguments

### Common
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model key (required) | — |
| `--batch-size` | Batch size | 1 |
| `--precision` | `fp32` / `fp16` / `bf16` | `fp32` |
| `--device` | `auto` / `cpu` / `cuda` / `cuda:0` | `auto` |
| `--output` | Save JSON results to path | — |
| `--no-download` | Skip model/dataset download check | False |

### Benchmark-only
| Argument | Description | Default |
|----------|-------------|---------|
| `--warmup` | Warmup iterations | 5 |
| `--iterations` | Benchmark iterations | 50 |

### Stress-only
| Argument | Description | Default |
|----------|-------------|---------|
| `--duration` | Duration in seconds | 60 |
| `--interval` | Stats reporting interval (s) | 1.0 |

---

## Metrics Reported

### Benchmark
- **Latency**: mean, median, std, min, max, P90, P95, P99
- **Throughput**: samples/sec and batches/sec
- **GPU**: peak memory (MB), mean utilization (%)

### Stress Test
- **Throughput**: mean, min, max (samples/sec)
- **Totals**: iterations, samples, errors
- **GPU**: peak memory, max temperature, mean utilization

---

## Data & Model Storage

```
data/
├── models/
│   ├── resnet50/        # torchvision cache
│   ├── retinanet/       # torchvision cache
│   ├── yolov11/         # yolo11n.pt (~6MB)
│   ├── stable_diffusion/# SD v1.5 (~4GB)
│   ├── dlrm_v2/         # built in memory
│   └── whisper/         # whisper-base (~145MB)
└── datasets/
    ├── imagenet_sample/  # synthetic
    ├── coco_sample/      # synthetic
    ├── sd_prompts/       # built-in
    ├── criteo_sample/    # synthetic
    └── librispeech_sample/ # ~20MB HF sample
```

Models and datasets are downloaded **once** and cached. Subsequent runs skip the download.

---

## Notes

- **Stable Diffusion** requires ~4GB VRAM in fp32, ~2.5GB in fp16. Strongly recommend `--precision fp16`.
- **YOLOv11** requires the `ultralytics` package (included in requirements).
- **DLRM v2** uses a custom PyTorch implementation of the MLPerf DLRM v2 architecture with synthetic Criteo-format data.
- GPU monitoring requires `pynvml`. If not available, only PyTorch memory stats are shown.
- Press `Ctrl+C` to gracefully stop a stress test and print results.
