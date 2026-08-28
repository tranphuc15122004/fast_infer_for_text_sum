import argparse

import torch


def bytes_to_str(num_bytes: int) -> str:
    """Human-friendly binary byte string (MiB, GiB)."""
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GiB"
    elif num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MiB"
    else:
        return f"{num_bytes} bytes"


def measure_bandwidth(num_bytes: int, repeats: int) -> float:
    """Return **effective** DRAM bandwidth (read + write) in GB/s.

    We copy a device buffer *src* into another device buffer *dst* using
    `Tensor.copy_`.  Each logical copy triggers **two** DRAM transactions:

        1. DRAM → L2/L1 → SM (read *src*)
        2. SM → L2/L1 → DRAM (write *dst*)

    To match the definition used in Roofline (total bus traffic), we therefore
    multiply the byte count by 2 when converting to GB/s.
    """

    # Number of bf16 elements (round up) — 1 element = 2 bytes
    elems = (num_bytes + 1) // 2

    src = torch.empty(elems, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)

    # Warm-up to engage clocks and populate caches
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Time *repeats* consecutive copies with one pair of events to minimise
    # measurement overhead.
    start.record()
    for _ in range(repeats):
        dst.copy_(src, non_blocking=False)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / repeats

    effective_bytes = 2 * num_bytes  # read + write traffic per copy
    bandwidth_gbps = (effective_bytes * 1e-9) / (avg_ms * 1e-3)  # GB/s
    return bandwidth_gbps


def measure_flops(mat_size: int, repeats: int) -> float:
    """Return average bfloat16 GEMM throughput in GFLOP/s using cuBLASLt."""
    A = torch.randn(mat_size, mat_size, device="cuda", dtype=torch.bfloat16)
    B = torch.randn_like(A)

    # Warm-up
    for _ in range(3):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / repeats

    flops_per_iter = 2.0 * mat_size**3
    gflops = (flops_per_iter / 1e9) / (avg_ms * 1e-3)
    return gflops


def flops_to_bandwidth_ratio(
    copy_bytes: int = 1 * 1024 * 1024 * 1024, mm_size: int = 8192, repeats: int = 100
) -> float:
    """
    Estimate GPU balance point (GFLOPs per GB/s) using roofline-style benchmarking.

    - Bandwidth: measured with streaming AXPY-like kernel (2 reads + 1 write).
    - FLOPs: measured with large GEMM on tensor cores (BF16).
    - Ratio = GFLOPs / GB/s, i.e. arithmetic intensity required to be compute-bound.

    Args:
        copy_bytes (int): Size of the buffer for bandwidth test (default: 1 GiB).
        mm_size (int): Matrix size for GEMM test (default: 8192).
        repeats (int): Number of iterations to average (default: 100).

    Returns:
        float: FLOPs-to-bandwidth ratio (GFLOPs per GB/s).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available — run on a GPU-equipped system.")

    # Bandwidth measurement
    elems = copy_bytes // 2  # bf16 = 2 bytes
    a = torch.randn(elems, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)

    # warm-up
    for _ in range(3):
        torch.add(a, b, out=b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(repeats):
        torch.add(a, b, out=b)  # 2 reads + 1 write
    end.record()
    torch.cuda.synchronize()

    avg_ms_bw = start.elapsed_time(end) / repeats
    bandwidth_gbps = (3 * copy_bytes * 1e-9) / (avg_ms_bw * 1e-3)
    print("\n\nBandwidth: ", bandwidth_gbps)

    # FLOPs measurement
    A = torch.randn(mm_size, mm_size, device="cuda", dtype=torch.bfloat16)
    B = torch.randn_like(A)

    for _ in range(3):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeats):
        torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()

    avg_ms_flops = start.elapsed_time(end) / repeats
    flops_gflops = (2.0 * mm_size**3 / 1e9) / (avg_ms_flops * 1e-3)
    print("GFLOPS: ", flops_gflops)

    print("Flops to bw ratio: ", flops_gflops / bandwidth_gbps)


def main():
    parser = argparse.ArgumentParser(
        description="GPU bfloat16 bandwidth & FLOP micro-benchmark (PyTorch-only)"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Kernel launch iterations to average (default: 100)",
    )
    parser.add_argument(
        "--copy-bytes",
        type=int,
        default=1 * 1024 * 1024 * 1024,
        help="Problem size for the copy test in bytes (default: 1 GiB)",
    )
    parser.add_argument(
        "--mm-size",
        type=int,
        default=4096,
        help="Matrix dimension N for GEMM (default: 4096)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available — run on a GPU-equipped system.")

    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)

    print(f"GPU: {prop.name}")
    print(f"Compute capability: {prop.major}.{prop.minor}")
    print(f"Total memory: {prop.total_memory / (1 << 30):.1f} GiB\n")

    # ----------------------- Bandwidth -----------------------
    print("---- Memory Copy Test ----")
    print(f"Problem size: {bytes_to_str(args.copy_bytes)} (device→device)")
    bw = measure_bandwidth(args.copy_bytes, args.repeats)
    print(f"Effective DRAM bandwidth: {bw:.2f} GB/s\n")

    # ----------------------- Compute (GEMM) ------------------
    print("---- Compute (GEMM) Test ----")
    print(f"Matrix size: {args.mm_size} x {args.mm_size}")
    gflops = measure_flops(args.mm_size, args.repeats)
    print(f"Average Tensor-Core throughput: {gflops:.0f} GFLOP/s")

    flops_to_bandwidth_ratio()


if __name__ == "__main__":
    main()
