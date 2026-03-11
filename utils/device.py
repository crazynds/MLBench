"""Device detection and resolution"""

import torch


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device_str)


def get_device_info() -> dict:
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [],
        "cpu_threads": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024 ** 3)
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "memory_gb": round(mem_total, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    return info


def print_device_info():
    info = get_device_info()
    print("\n" + "=" * 55)
    print("  🖥️  DEVICE INFORMATION")
    print("=" * 55)
    print(f"  PyTorch Version   : {info['pytorch_version']}")
    print(f"  CUDA Available    : {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  CUDA Version      : {info['cuda_version']}")
        print(f"  cuDNN Version     : {info['cudnn_version']}")
        print(f"  GPU Count         : {len(info['gpus'])}")
        for gpu in info["gpus"]:
            print(f"\n  GPU [{gpu['index']}]: {gpu['name']}")
            print(f"    Memory          : {gpu['memory_gb']:.1f} GB")
            print(f"    Compute Cap.    : {gpu['compute_capability']}")
            print(f"    Multiprocessors : {gpu['multi_processor_count']}")
    print(f"\n  CPU Threads       : {info['cpu_threads']}")
    print("=" * 55 + "\n")