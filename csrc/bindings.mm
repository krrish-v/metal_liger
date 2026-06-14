// Copyright 2025 MetalLiger Contributors
// csrc/bindings.mm — THE ONLY file with PYBIND11_MODULE.
//
// Each operation is implemented in its own .mm file (rms_norm_metal.mm,
// swiglu_metal.mm). This file just wires them to Python.
// Having one PYBIND11_MODULE = one _PyInit_metal_liger_ext symbol = no linker error.

#include "metal_liger_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MetalLiger Phase 4: Native Metal dispatch for fused ML ops on Apple Silicon";

    // ── RMSNorm ──────────────────────────────────────────────────────────────
    m.def("rms_norm_forward", &metal_rms_norm_forward, "Metal RMSNorm forward",
          py::arg("x"), py::arg("weight"), py::arg("eps"), py::arg("kernel_dir"));

    m.def("rms_norm_backward",
          &metal_rms_norm_backward,
          "Fused RMSNorm backward — per-row grad_weight (no GPU atomics)",
          py::arg("x"), py::arg("weight"), py::arg("grad_out"),
          py::arg("eps"), py::arg("kernel_dir"));

    // ── SwiGLU ───────────────────────────────────────────────────────────────
    m.def("swiglu_forward",
          &metal_swiglu_forward,
          "Fused SwiGLU forward — SiLU(gate)*up, single Metal dispatch",
          py::arg("gate"), py::arg("up"), py::arg("kernel_dir"));

    m.def("swiglu_backward",
          &metal_swiglu_backward,
          "Fused SwiGLU backward — recomputes sigmoid, no activation cache",
          py::arg("gate"), py::arg("up"), py::arg("grad_out"), py::arg("kernel_dir"));
}
