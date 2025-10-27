# Candle Inference Script

This directory contains a basic LLM inference script using the Candle framework.

## File Structure
```
candle-inf/
├── base-inf.rs           # Main inference script (Rust)
├── candle/               # Candle repository (submodule)
├── Cargo.toml            # Rust project configuration
└── README.md             # This file
```

## Prerequisites

1. **Rust toolchain:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **CUDA (optional, for GPU support):**
   - Install CUDA Toolkit 11.8+ or 12.x
   - Ensure `nvcc` is in your PATH

## Setup

### Create Cargo.toml

Create a `Cargo.toml` file in this directory:

```toml
[package]
name = "candle-inference"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "base-inf"
path = "base-inf.rs"

[dependencies]
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
tokenizers = "0.19"
hf-hub = "0.3"
serde = "1.0"
serde_json = "1.0"

# Candle dependencies
candle-core = { git = "https://github.com/huggingface/candle.git" }
candle-nn = { git = "https://github.com/huggingface/candle.git" }
candle-transformers = { git = "https://github.com/huggingface/candle.git" }

# Optional: for CUDA support
# candle-core = { git = "https://github.com/huggingface/candle.git", features = ["cuda"] }

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
mkl = ["candle-core/mkl", "candle-nn/mkl"]
```

## Building

### CPU-only build:
```bash
cargo build --release
```

### With CUDA support:
```bash
cargo build --release --features cuda
```

### With Apple Accelerate (macOS):
```bash
cargo build --release --features accelerate
```

### With Intel MKL:
```bash
cargo build --release --features mkl
```

## Running the Script

### Basic usage:
```bash
cargo run --release -- -m meta-llama/Llama-2-7b-hf
```

### With custom prompt:
```bash
cargo run --release -- -m meta-llama/Llama-2-7b-hf -p "Tell me about Rust programming"
```

### All options:
```bash
cargo run --release -- \
  -m <model_id> \
  -p "Your prompt" \
  -n <tokens> \
  --temperature <temp> \
  --top-p <value> \
  --top-k <value> \
  --dtype <f16|bf16|f32>
```

**Options:**
- `-m, --model-id` - HuggingFace model ID (required)
- `-p, --prompt` - Text prompt (default: "Hello, my name is")
- `-n, --num-tokens` - Number of tokens to generate (default: 128)
- `--cpu` - Force CPU usage
- `--temperature` - Sampling temperature (default: 0.8)
- `--top-p` - Nucleus sampling threshold
- `--top-k` - Top-k sampling
- `--seed` - Random seed (default: 299792458)
- `--dtype` - Data type: f16, bf16, or f32 (default: f16)
- `--repeat-penalty` - Penalty for repeating tokens (default: 1.1)
- `--repeat-last-n` - Context for repeat penalty (default: 128)
- `--no-kv-cache` - Disable key-value cache
- `--revision` - Model revision/branch

### Examples:

**Basic inference:**
```bash
cargo run --release -- -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "What is quantum computing?" -n 100
```

**With GPU (CUDA):**
```bash
cargo run --release --features cuda -- \
  -m meta-llama/Llama-2-7b-hf \
  -p "Explain machine learning" \
  -n 200 \
  --temperature 0.7 \
  --top-p 0.9
```

**CPU inference with f32:**
```bash
cargo run --release -- \
  -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --cpu \
  --dtype f32 \
  -p "Write a poem about the ocean" \
  -n 150
```

**Greedy decoding (deterministic):**
```bash
cargo run --release -- \
  -m meta-llama/Llama-2-7b-hf \
  -p "The capital of France is" \
  --temperature 0.0
```

## Model Support

This script supports models with Llama-compatible architecture:
- Meta Llama 2 & 3
- TinyLlama
- SmolLM
- Solar
- And other Llama-based models

**Note:** You may need to accept model licenses on HuggingFace and use authentication:
```bash
export HF_TOKEN=your_huggingface_token
```

## Features

- ✅ HuggingFace Hub integration
- ✅ GPU acceleration (CUDA)
- ✅ Multiple sampling strategies (greedy, top-k, top-p)
- ✅ Repeat penalty
- ✅ Key-value caching
- ✅ Multiple data types (f16, bf16, f32)
- ✅ Performance statistics

## Troubleshooting

**CUDA out of memory:**
- Try using f16 instead of f32: `--dtype f16`
- Use a smaller model
- Reduce context size

**Model download fails:**
- Check your internet connection
- For gated models, set your HF token: `export HF_TOKEN=your_token`
- Try specifying a revision: `--revision main`

**Slow CPU inference:**
- Compile with optimizations: `--release`
- Use smaller precision: `--dtype f16`
- Enable CPU-specific features: `--features mkl` or `--features accelerate`

## Performance Tips

1. **GPU inference:** Always build with `--features cuda` for GPU support
2. **Data types:** Use f16 for best GPU performance, f32 for CPU
3. **Model size:** Start with smaller models (TinyLlama) for testing
4. **Batch size:** The script uses batch size of 1 for simplicity
5. **KV cache:** Keep it enabled (default) for better performance

## License

This script is MIT licensed. Candle is Apache 2.0 / MIT licensed.
