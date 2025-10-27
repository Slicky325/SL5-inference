# mistral.rs Inference Script

This directory contains a basic LLM inference script using the mistral.rs framework.

## File Structure
```
mistral.rs-inf/
├── base-inf.rs           # Main inference script (Rust)
├── mistral.rs/           # mistral.rs repository (submodule)
├── Cargo.toml            # Rust project configuration
└── README.md             # This file
```

## Prerequisites

1. **Rust toolchain:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **CUDA (optional, for GPU support):**
   - Install CUDA Toolkit 12.x
   - Ensure `nvcc` is in your PATH

3. **OpenSSL development files:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libssl-dev pkg-config
   
   # RHEL/CentOS
   sudo yum install openssl-devel
   ```

## Setup

### Create Cargo.toml

Create a `Cargo.toml` file in this directory:

```toml
[package]
name = "mistralrs-inference"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "base-inf"
path = "base-inf.rs"

[dependencies]
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
tokio = { version = "1.40", features = ["full"] }
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git", features = ["cuda"] }

[features]
default = ["cuda"]
cuda = ["mistralrs/cuda"]
metal = ["mistralrs/metal"]
```

## Building

### CPU-only build:
```bash
cargo build --release --no-default-features
```

### With CUDA support:
```bash
cargo build --release --features cuda
```

### With Metal support (macOS):
```bash
cargo build --release --features metal
```

## Running the Script

### Basic usage (HuggingFace models):
```bash
cargo run --release -- \
  -m meta-llama/Llama-2-7b-hf \
  -a llama \
  -p "Tell me about Rust"
```

### Using GGUF models:
```bash
cargo run --release -- \
  -m TheBloke/Llama-2-7B-GGUF \
  -t gguf \
  --gguf-file llama-2-7b.Q4_K_M.gguf \
  --tokenizer-id meta-llama/Llama-2-7b-hf \
  -a llama \
  -p "What is machine learning?"
```

### Chat completion format:
```bash
cargo run --release -- \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  -a mistral \
  --chat \
  -p "Explain quantum computing in simple terms"
```

### All options:
```bash
cargo run --release -- \
  -m <model_id> \
  -a <arch> \
  -t <model_type> \
  -p "Your prompt" \
  -n <tokens> \
  --temperature <temp> \
  --top-p <value> \
  --top-k <value>
```

**Options:**
- `-m, --model-id` - HuggingFace model ID (required)
- `-a, --arch` - Architecture: llama, mistral, phi3, qwen2, gemma, gemma2
- `-t, --model-type` - Model type: normal or gguf (default: normal)
- `-p, --prompt` - Text prompt (default: "Hello, my name is")
- `-n, --max-tokens` - Number of tokens to generate (default: 128)
- `--gguf-file` - GGUF filename (required for gguf type)
- `--tokenizer-id` - Tokenizer model ID (for GGUF)
- `--temperature` - Sampling temperature (default: 0.8)
- `--top-p` - Nucleus sampling (default: 0.95)
- `--top-k` - Top-k sampling
- `--repeat-penalty` - Repeat penalty (default: 1.1)
- `--presence-penalty` - Presence penalty (default: 0.0)
- `--frequency-penalty` - Frequency penalty (default: 0.0)
- `--chat` - Use chat completion format
- `--cpu` - Force CPU usage

### Examples:

**Mistral 7B Instruct:**
```bash
cargo run --release -- \
  -m mistralai/Mistral-7B-Instruct-v0.1 \
  -a mistral \
  --chat \
  -p "Write a Python function to sort a list" \
  -n 200
```

**Llama 2 with custom parameters:**
```bash
cargo run --release -- \
  -m meta-llama/Llama-2-7b-chat-hf \
  -a llama \
  --chat \
  -p "What is the meaning of life?" \
  -n 150 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.2
```

**Phi-3 Mini:**
```bash
cargo run --release -- \
  -m microsoft/Phi-3-mini-4k-instruct \
  -a phi3 \
  --chat \
  -p "Explain neural networks" \
  -n 100
```

**GGUF quantized model:**
```bash
# First download the GGUF file
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Then run inference
cargo run --release -- \
  -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
  -t gguf \
  --gguf-file mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  --tokenizer-id mistralai/Mistral-7B-Instruct-v0.1 \
  -a mistral \
  -p "What is Rust programming language?" \
  -n 100
```

**CPU inference:**
```bash
cargo run --release --no-default-features -- \
  -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -a llama \
  --cpu \
  -p "Hello world" \
  -n 50
```

## Supported Architectures

- **Llama** - Llama 2, Llama 3, TinyLlama, etc.
- **Mistral** - Mistral 7B, Mixtral
- **Phi3** - Microsoft Phi-3 models
- **Qwen2** - Qwen 2 models
- **Gemma** - Google Gemma models
- **Gemma2** - Google Gemma 2 models

## Model Types

### Normal (HuggingFace)
Standard HuggingFace models in safetensors or PyTorch format.

### GGUF
Quantized models in GGUF format (faster, less memory):
- Q4_K_M - Good balance of quality and speed
- Q5_K_M - Higher quality
- Q8_0 - Highest quality quantization

## Features

- ✅ Multiple model architectures
- ✅ GGUF quantized model support
- ✅ GPU acceleration (CUDA/Metal)
- ✅ Chat completion format
- ✅ Advanced sampling (top-p, top-k, temperature)
- ✅ Repeat/presence/frequency penalties
- ✅ Async runtime (Tokio)
- ✅ HuggingFace Hub integration

## Troubleshooting

**Compilation errors:**
- Make sure you have the latest Rust: `rustup update`
- Check OpenSSL is installed: `pkg-config --libs openssl`

**CUDA errors:**
- Ensure CUDA 12.x is installed
- Check `nvcc --version`
- Build with `--features cuda`

**Out of memory:**
- Use GGUF quantized models: `-t gguf`
- Try Q4_K_M quantization level
- Reduce max tokens: `-n 50`
- Use CPU: `--cpu`

**Model not found:**
- Check model ID is correct on HuggingFace
- For gated models: `export HF_TOKEN=your_token`
- Verify architecture matches: `-a llama` for Llama models

**Slow inference:**
- Build in release mode: `--release`
- Enable GPU: `--features cuda` or `--features metal`
- Use quantized models (GGUF with Q4_K_M)

## Performance Tips

1. **Use GGUF models** for best performance/quality trade-off
2. **GPU acceleration** provides 5-10x speedup over CPU
3. **Quantization:** Q4_K_M is recommended for most use cases
4. **Batch size:** mistral.rs handles batching automatically
5. **Chat format:** Use `--chat` for instruct/chat models

## Authentication

For gated models (like Llama 2):
```bash
export HF_TOKEN=your_huggingface_token
```

Or use:
```bash
huggingface-cli login
```

## Advanced Usage

### With custom sampling:
```bash
cargo run --release -- \
  -m mistralai/Mistral-7B-v0.1 \
  -a mistral \
  -p "Once upon a time" \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 50 \
  --repeat-penalty 1.15 \
  --presence-penalty 0.1 \
  -n 200
```

### Greedy decoding (deterministic):
```bash
cargo run --release -- \
  -m meta-llama/Llama-2-7b-hf \
  -a llama \
  -p "The capital of France is" \
  --temperature 0.0
```

## License

This script is MIT licensed. mistral.rs is MIT licensed.
