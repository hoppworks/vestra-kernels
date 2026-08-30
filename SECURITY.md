# Security policy

## Supported versions

Security fixes target the latest release and the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for the Vestra Kernels
repository. Include the affected revision, reproduction steps, impact, and any
suggested mitigation. Do not open a public issue before a fix is available.

## Security boundaries

The project treats tensor shapes, strides, buffers, CPU feature detection, and
CUDA runtime interaction as security-sensitive boundaries. Callers must not
infer that a model, fixture, or device input is safe merely because a kernel
accepts it.
