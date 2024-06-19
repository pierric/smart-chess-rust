[![Rust](https://github.com/pierric/smart-chess-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/pierric/smart-chess-rust/actions/workflows/rust.yml)

# Build
## Python part
```
poetry install
```

## Rust part
```
poetry shell
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo build
```
Tch is always built against the torch with patch-version being 0. But very safe to be run with the other patch versions. It is necessary to export the environment variable `LIBTORCH_BYPASS_VERSION_CHECK=1`.

# Quantization
Experiments with quantized model (int8) is also tried. No convicing results are seen, neither faster nor better performance. If interested, installing pytorch_quantization as described here: https://github.com/NVIDIA/TensorRT/tree/release/9.3/tools/pytorch-quantization/pytorch_quantization
