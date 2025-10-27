# SL5-inference

Tiny workspace with multiple inference backends: `candle-inf`, `llamacpp-inf`, and `mistral.rs-inf`.

**Prerequisites**: WSL/Ubuntu, Rust, C/C++ toolchain, CMake, Git.

## Clone with submodules

```bash
git clone --recursive <your-repo-url>
# or if already cloned:
git submodule update --init --recursive
```

## Minimal setup

```bash
# in WSL at repo root
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# build llama.cpp
cd llamacpp-inf/llama.cpp
cmake -B build
cmake --build build --config Release

# build Rust backends
cd ../../candle-inf && cargo build --release
cd ../mistral.rs-inf && cargo build --release
```

See each subfolder README for detailed run instructions.

