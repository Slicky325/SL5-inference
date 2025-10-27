// Basic LLM Inference Script using mistral.rs
// Build: cargo build --release --features cuda (or without cuda for CPU)
// Run: cargo run --release -- -m <model_id> -p "Your prompt here"

use anyhow::Result;
use clap::{Parser, ValueEnum};

use mistralrs::{
    Architecture, ChatCompletionRequest, Constraint, Device, DeviceMapMetadata, GGUFLoaderBuilder,
    GGUFSpecificConfig, MessageContent, MistralRs, MistralRsBuilder, NormalLoaderBuilder,
    NormalRequest, Request, RequestMessage, Response, SamplingParams, SchedulerMethod,
    TokenSource,
};

use std::sync::Arc;

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    /// Standard HuggingFace models
    Normal,
    /// GGUF quantized models
    Gguf,
}

#[derive(Debug, Clone, ValueEnum)]
enum ArchType {
    Llama,
    Mistral,
    Phi3,
    Qwen2,
    Gemma,
    Gemma2,
}

impl ArchType {
    fn to_architecture(&self) -> Architecture {
        match self {
            ArchType::Llama => Architecture::Llama,
            ArchType::Mistral => Architecture::Mistral,
            ArchType::Phi3 => Architecture::Phi3,
            ArchType::Qwen2 => Architecture::Qwen2,
            ArchType::Gemma => Architecture::Gemma,
            ArchType::Gemma2 => Architecture::Gemma2,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "base-inf",
    about = "Basic LLM Inference with mistral.rs",
    long_about = "A simple script to run LLM inference using the mistral.rs framework"
)]
struct Args {
    /// Model ID from HuggingFace Hub
    #[arg(short = 'm', long)]
    model_id: String,

    /// The initial prompt for text generation
    #[arg(short = 'p', long, default_value = "Hello, my name is")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Model type (normal or gguf)
    #[arg(short = 't', long, value_enum, default_value = "normal")]
    model_type: ModelType,

    /// Architecture type
    #[arg(short = 'a', long, value_enum, default_value = "llama")]
    arch: ArchType,

    /// GGUF filename (only for gguf model type)
    #[arg(long)]
    gguf_file: Option<String>,

    /// Tokenizer model ID (for GGUF models, defaults to model_id)
    #[arg(long)]
    tokenizer_id: Option<String>,

    /// Temperature for sampling (0.0 = greedy, higher = more random)
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    /// Top-k sampling
    #[arg(long)]
    top_k: Option<usize>,

    /// Penalty for repeating tokens (1.0 = no penalty)
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Presence penalty
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,

    /// Frequency penalty
    #[arg(long, default_value_t = 0.0)]
    frequency_penalty: f32,

    /// Use chat completion format
    #[arg(long)]
    chat: bool,

    /// Run on CPU instead of GPU
    #[arg(long)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("\n=== Basic LLM Inference with mistral.rs ===\n");
    println!("Model ID: {}", args.model_id);
    println!("Model Type: {:?}", args.model_type);
    println!("Architecture: {:?}", args.arch);
    println!("Prompt: \"{}\"", args.prompt);
    println!("Max Tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!("Device: {}", if args.cpu { "CPU" } else { "CUDA" });
    println!();

    // Build the model loader based on type
    let model = match args.model_type {
        ModelType::Normal => {
            println!("Loading normal HuggingFace model...");
            let loader = NormalLoaderBuilder::new(
                NormalSpecificConfig::default(),
                None,  // chat_template
                None,  // tokenizer_json
                Some(args.model_id.clone()),
            )
            .build()?;

            MistralRsBuilder::new(
                loader,
                args.arch.to_architecture(),
                TokenSource::CacheToken,
                None,
            )
            .with_logging()
            .with_device_mapping(if args.cpu {
                DeviceMapMetadata::dummy()
            } else {
                DeviceMapMetadata::from_num_device_layers(vec![(Device::Cuda(0), None)])
            })
            .build()
            .await?
        }
        ModelType::Gguf => {
            println!("Loading GGUF quantized model...");
            let gguf_file = args.gguf_file.ok_or_else(|| {
                anyhow::anyhow!("--gguf-file is required when using GGUF model type")
            })?;
            let tokenizer_id = args.tokenizer_id.unwrap_or_else(|| args.model_id.clone());

            let loader = GGUFLoaderBuilder::new(
                None,  // chat_template
                Some(tokenizer_id),
                Some(args.model_id.clone()),
                vec![gguf_file],
                GGUFSpecificConfig::default(),
            )
            .build()?;

            MistralRsBuilder::new(
                loader,
                args.arch.to_architecture(),
                TokenSource::CacheToken,
                None,
            )
            .with_logging()
            .with_device_mapping(if args.cpu {
                DeviceMapMetadata::dummy()
            } else {
                DeviceMapMetadata::from_num_device_layers(vec![(Device::Cuda(0), None)])
            })
            .build()
            .await?
        }
    };

    println!("Model loaded successfully!\n");

    // Set up sampling parameters
    let sampling_params = SamplingParams {
        temperature: Some(args.temperature),
        top_p: Some(args.top_p),
        top_k: args.top_k,
        frequency_penalty: Some(args.frequency_penalty),
        presence_penalty: Some(args.presence_penalty),
        max_len: Some(args.max_tokens),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    };

    // Create request
    let request = if args.chat {
        // Chat completion format
        let messages = vec![RequestMessage {
            content: MessageContent::Text(args.prompt.clone()),
            role: "user".to_string(),
            name: None,
        }];

        Request::Chat(ChatCompletionRequest {
            messages,
            model: "default".to_string(),
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            max_tokens: Some(args.max_tokens),
            n_choices: 1,
            presence_penalty: Some(args.presence_penalty),
            frequency_penalty: Some(args.frequency_penalty),
            stop_seqs: None,
            temperature: Some(args.temperature),
            top_p: Some(args.top_p),
            top_k: args.top_k,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            adapters: None,
            dry_params: None,
        })
    } else {
        // Normal text completion
        Request::Normal(NormalRequest {
            messages: RequestMessage {
                content: MessageContent::Text(args.prompt.clone()),
                role: "user".to_string(),
                name: None,
            },
            model: "default".to_string(),
            sampling_params: sampling_params.clone(),
            response: None,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint: Constraint::None,
            adapters: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
        })
    };

    println!("=== Output ===\n{}", args.prompt);

    let start_time = std::time::Instant::now();

    // Send the request
    let response = model.send_chat_completion_request(request).await?;

    // Print the response
    match response {
        Response::CompletionDone(completion) => {
            for choice in completion.choices {
                print!("{}", choice.message.content);
            }
            println!();

            let elapsed = start_time.elapsed();
            println!("\n=== Statistics ===");
            println!("Prompt tokens: {}", completion.usage.prompt_tokens);
            println!("Completion tokens: {}", completion.usage.completion_tokens);
            println!("Total tokens: {}", completion.usage.total_tokens);
            println!("Time: {:.2?}", elapsed);
            println!(
                "Speed: {:.2} tokens/s",
                completion.usage.completion_tokens as f64 / elapsed.as_secs_f64()
            );
        }
        Response::CompletionModelError(err, _) => {
            eprintln!("Error: {}", err);
        }
        Response::ValidationError(err) => {
            eprintln!("Validation Error: {}", err);
        }
        _ => {
            eprintln!("Unexpected response type");
        }
    }

    println!("\n=== Inference Complete ===\n");

    Ok(())
}

// Placeholder types - adjust based on actual mistral.rs API
struct NormalSpecificConfig {
    // Add fields as needed
}

impl Default for NormalSpecificConfig {
    fn default() -> Self {
        Self {}
    }
}
