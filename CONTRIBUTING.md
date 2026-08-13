# Contributing to Vestra Kernels

Vestra Kernels owns low-level work only. Do not add model loading, GGUF
parsing, CLI code, engine configuration, or reconstruction types here.

Every kernel change needs an oracle that states its accumulation order and
numerical tolerance. It must retain an ISA fallback and may become the default
only after an alternating same-binary benchmark, the engine's four-image F32
parity gate, and an end-to-end benchmark qualification. Do not tune a kernel
by changing the model, input, precision, thread budget, or measurement scope.

Run the crate tests and the real-shape microbenchmarks before handing a change
to Vestra Engine. Record the target CPU, compiler version, target CPU flag,
and exact environment switches with every performance result.
