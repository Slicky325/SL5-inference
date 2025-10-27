// Basic LLM Inference Script using Candle
// Build: cargo build --release --features cuda (or without cuda for CPU)
// Run: cargo run --release -- -m <model_id> -p "Your prompt here"

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

use std::io::Write;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "Hello, my name is";

#[derive(Parser, Debug)]
#[command(
    name = "base-inf",
    about = "Basic LLM Inference with Candle",
    long_about = "A simple script to run LLM inference using the Candle framework"
)]
struct Args {
    /// Model ID from HuggingFace Hub (e.g., "meta-llama/Llama-2-7b-hf")
    #[arg(short = 'm', long)]
    model_id: String,

    /// The initial prompt for text generation
    #[arg(short = 'p', long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 128)]
    num_tokens: usize,

    /// Run on CPU instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Temperature for sampling (higher = more random)
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling (sample from top k tokens)
    #[arg(long)]
    top_k: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Data type (f16, bf16, f32)
    #[arg(long, default_value = "f16")]
    dtype: String,

    /// Penalty for repeating tokens (1.0 = no penalty)
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Context size for repeat penalty
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

    /// Disable key-value cache
    #[arg(long)]
    no_kv_cache: bool,

    /// Revision/branch to use from HuggingFace
    #[arg(long)]
    revision: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("\n=== Basic LLM Inference with Candle ===\n");
    println!("Model ID: {}", args.model_id);
    println!("Prompt: \"{}\"", args.prompt);
    println!("Tokens to generate: {}", args.num_tokens);
    println!("Device: {}", if args.cpu { "CPU" } else { "GPU (CUDA)" });
    println!("Temperature: {}", args.temperature);
    println!();

    // Set up device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("Using device: {:?}\n", device);

    // Parse dtype
    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        dtype => bail!("Unsupported dtype: {}", dtype),
    };

    // Download model from HuggingFace Hub
    println!("Downloading model files from HuggingFace Hub...");
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.unwrap_or("main".to_string()),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let weights_filename = repo.get("model.safetensors").or_else(|_| {
        println!("model.safetensors not found, trying pytorch_model.bin...");
        repo.get("pytorch_model.bin")
    })?;

    println!("Model files downloaded successfully!\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded!\n");

    // Load config
    println!("Loading model config...");
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    println!("Config loaded: {:?}\n", config);

    // Load model weights
    println!("Loading model weights...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)?
    };

    let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;
    let llama = Llama::load(vb, &config)?;
    println!("Model loaded successfully!\n");

    // Tokenize the prompt
    println!("Tokenizing prompt...");
    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
    let prompt_tokens = tokens.get_ids().to_vec();
    println!("Tokenized into {} tokens\n", prompt_tokens.len());

    // Convert tokens to tensor
    let mut tokens_tensor = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;

    // Set up the sampler
    let mut logits_processor = {
        let sampling = if args.temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature: args.temperature },
                (Some(k), None) => Sampling::TopK { k, temperature: args.temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature: args.temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: args.temperature,
                },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    // Generate tokens
    println!("=== Output ===\n{}", args.prompt);
    std::io::stdout().flush()?;

    let start_gen = std::time::Instant::now();
    let mut generated_tokens = 0usize;
    let mut pos = 0;

    for index in 0..args.num_tokens {
        let start_token = std::time::Instant::now();

        // Forward pass through the model
        let logits = llama.forward(&tokens_tensor, pos, &cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        // Apply repeat penalty
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = prompt_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &prompt_tokens[start_at..],
            )?
        };

        // Sample next token
        let next_token = logits_processor.sample(&logits)?;
        generated_tokens += 1;

        // Check for EOS token
        if let Some(text) = tokenizer.decode(&[next_token], true).ok() {
            if text == EOS_TOKEN || text.contains(EOS_TOKEN) {
                println!("\n[End of generation]");
                break;
            }
            print!("{}", text);
            std::io::stdout().flush()?;
        }

        // Update for next iteration
        pos += tokens_tensor.dim(1)?;
        tokens_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;

        let token_time = start_token.elapsed();
        if index % 10 == 0 && index > 0 {
            let tokens_per_sec = 10.0 / token_time.as_secs_f64();
            println!(" [{:.2} tok/s]", tokens_per_sec);
        }
    }

    let elapsed = start_gen.elapsed();
    println!("\n\n=== Statistics ===");
    println!("Tokens generated: {}", generated_tokens);
    println!("Time: {:.2?}", elapsed);
    println!(
        "Speed: {:.2} tokens/s",
        generated_tokens as f64 / elapsed.as_secs_f64()
    );
    println!("\n=== Inference Complete ===\n");

    Ok(())
}
