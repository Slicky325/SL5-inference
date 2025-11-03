# llama.cpp Inference Script

This directory contains a basic LLM inference script using llama.cpp.

## File Structure
```
llamacpp-inf/
├── base-inf.cpp          # Main inference script
├── llama.cpp/            # llama.cpp repository (submodule)
└── README.md             # This file
```

## Prerequisites

1. **Build llama.cpp first:**
   ```bash
   cd llama.cpp
   mkdir build
   cd build
   cmake .. -DGGML_CUDA=ON  # For GPU support, or without for CPU only
   cmake --build . --config Release
   cd ../..
   ```

2. **Get a GGUF model:**
   Download a GGUF model from HuggingFace, for example:
   ```bash
   # Using huggingface-cli
   huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
   
   # Or manually download from:
   # https://huggingface.co/TheBloke/Llama-2-7B-GGUF
   ```

## Building the Inference Script

### Option 1: Direct compilation with g++
```bash
g++ -std=c++17 -O3 base-inf.cpp -o base-inf \
    -I./llama.cpp/include \
    -I./llama.cpp/ggml/include \
    -L./llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -pthread
```

### Option 2: Using the llama.cpp build system
Add the following to `llama.cpp/examples/CMakeLists.txt`:
```cmake
add_executable(base-inf ../../base-inf.cpp)
target_link_libraries(base-inf PRIVATE llama common)
install(TARGETS base-inf RUNTIME)
```

Then build:
```bash
cd llama.cpp/build
cmake ..
cmake --build . --target base-inf
```

## Running the Script

**Important:** Before running, set the library path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./llama.cpp/build/bin
```

### Basic usage:
```bash
./base-inf -m path/to/model.gguf
```

### With custom prompt:
```bash
./base-inf -m path/to/model.gguf "Tell me a story about AI"
```

### All options:
```bash
./base-inf -m <model.gguf> -n <tokens> -ngl <gpu_layers> [prompt]
```

**Options:**
- `-m <path>` - Path to GGUF model file (required)
- `-n <number>` - Number of tokens to generate (default: 128)
- `-ngl <number>` - Number of GPU layers to offload (default: 99)
- `[prompt]` - Text prompt (default: "Hello, my name is")

### Examples:

**CPU inference:**
```bash
./base-inf -m models/llama-2-7b.Q4_K_M.gguf -ngl 0 -n 50 "What is Rust?"
```

**GPU inference:**
```bash
./base-inf -m models/llama-2-7b.Q4_K_M.gguf -ngl 99 -n 200 "Explain quantum computing"
```

## Features

- ✅ Simple command-line interface
- ✅ GPU acceleration support (via -ngl parameter)
- ✅ Configurable generation length
- ✅ Greedy decoding (deterministic output)
- ✅ Performance statistics
- ✅ Support for GGUF quantized models

## Troubleshooting

**Library not found:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./llama.cpp/build/bin
```

**Cannot find llama.h:**
Make sure llama.cpp is properly built and the include path is correct.

**CUDA errors:**
Ensure you have CUDA installed and built llama.cpp with `-DGGML_CUDA=ON`.

## Performance Tips

1. Use quantized models (Q4_K_M, Q5_K_M) for better speed
2. Adjust `-ngl` based on your GPU VRAM
3. Use smaller batch sizes for lower latency
4. For CPU: compile with `-march=native` for better performance

## License

This script follows the same license as llama.cpp (MIT License).
