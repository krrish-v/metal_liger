# Copyright 2025 MetalLiger Contributors
"""M4 Pro GPU tuning constants and threadgroup optimization.

Apple M4 Pro GPU specs (16-core variant):
  - 16 GPU cores, each with 256 ALUs
  - SIMD width: 32 (warp equivalent)
  - Max threadgroup size: 1024
  - Threadgroup memory: 32KB
  - Memory bandwidth: 273 GB/s (unified)
  - Metal feature set: Apple GPU Family 9
"""

import torch


class M4ProConfig:
    """Hardware-aware configuration for M4 Pro GPU kernels."""

    SIMD_WIDTH = 32              # Apple GPU SIMD lane width (like CUDA warp)
    MAX_THREADGROUP = 1024       # Metal maximum threads per threadgroup
    THREADGROUP_MEMORY = 32768   # 32KB shared memory per threadgroup
    GPU_CORES = 16               # M4 Pro 16-core GPU
    MEMORY_BANDWIDTH_GBS = 273   # GB/s unified memory bandwidth

    # N_READS: number of elements each thread reads in a single pass.
    # MLX uses 4 for RMSNorm. This amortizes thread launch overhead.
    N_READS = 4

    @staticmethod
    def optimal_threadgroup_1d(axis_size: int) -> int:
        """Pick threadgroup size for row-wise reductions (RMSNorm, LayerNorm).

        Each thread reads N_READS elements, so we need axis_size/N_READS threads
        to cover one row. Round down to SIMD boundary.
        """
        needed = (axis_size + M4ProConfig.N_READS - 1) // M4ProConfig.N_READS
        tg = min(needed, M4ProConfig.MAX_THREADGROUP)
        # Round down to SIMD boundary for efficiency
        tg = max((tg // M4ProConfig.SIMD_WIDTH) * M4ProConfig.SIMD_WIDTH,
                 M4ProConfig.SIMD_WIDTH)
        return tg

    @staticmethod
    def needs_looped_kernel(axis_size: int) -> bool:
        """Check if axis_size is too large for single-pass coverage.

        If axis_size > MAX_THREADGROUP * N_READS, we need the 'looped' variant
        that iterates over the axis in chunks.
        """
        return axis_size > M4ProConfig.MAX_THREADGROUP * M4ProConfig.N_READS

    @staticmethod
    def grid_for_batch(num_rows: int) -> int:
        """Compute grid size for batch of rows (one threadgroup per row)."""
        return num_rows

    @staticmethod
    def is_mps_available() -> bool:
        """Check if MPS backend is available."""
        return (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available())
