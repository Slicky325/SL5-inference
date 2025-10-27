// Basic LLM Inference Script using llama.cpp
// Compile: g++ -std=c++17 -O3 base-inf.cpp -o base-inf -I./llama.cpp/include -L./llama.cpp/build/src -lllama -lggml
// Or use CMake from llama.cpp directory

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\n=== Basic LLM Inference with llama.cpp ===\n");
    printf("\nUsage:\n");
    printf("    %s -m <model.gguf> [-n tokens] [-ngl gpu_layers] [prompt]\n\n", argv[0]);
    printf("Options:\n");
    printf("    -m <path>      Path to GGUF model file (required)\n");
    printf("    -n <number>    Number of tokens to generate (default: 128)\n");
    printf("    -ngl <number>  Number of GPU layers to offload (default: 99)\n");
    printf("    [prompt]       Text prompt (default: 'Hello, my name is')\n\n");
    printf("Example:\n");
    printf("    %s -m ./models/llama-2-7b.Q4_K_M.gguf -n 50 \"Tell me a story\"\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    // Default parameters
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int ngl = 99;  // GPU layers to offload
    int n_predict = 128;  // Number of tokens to generate

    // Parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        fprintf(stderr, "Error: Invalid number for -n\n");
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        fprintf(stderr, "Error: Invalid number for -ngl\n");
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // Prompt starts here
                prompt = argv[i++];
                for (; i < argc; i++) {
                    prompt += " ";
                    prompt += argv[i];
                }
                break;
            }
        }
        
        if (model_path.empty()) {
            fprintf(stderr, "Error: Model path is required!\n");
            print_usage(argc, argv);
            return 1;
        }
    }

    printf("\n=== LLM Inference Starting ===\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Tokens to generate: %d\n", n_predict);
    printf("GPU layers: %d\n\n", ngl);

    // Load dynamic backends
    ggml_backend_load_all();

    // Initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    printf("Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "Error: Unable to load model from %s\n", model_path.c_str());
        return 1;
    }
    printf("Model loaded successfully!\n\n");

    // Get vocabulary
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize the prompt
    printf("Tokenizing prompt...\n");
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "Error: Failed to tokenize the prompt\n");
        llama_model_free(model);
        return 1;
    }
    printf("Tokenized into %d tokens\n\n", n_prompt);

    // Initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    printf("Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "Error: Failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }
    printf("Context created successfully!\n\n");

    // Initialize the sampler (greedy decoding for simplicity)
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Print the prompt
    printf("=== Output ===\n");
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "Error: Failed to convert token to piece\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
    fflush(stdout);

    // Prepare initial batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Handle encoder-decoder models
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "Error: Failed to encode\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // Main generation loop
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // Evaluate the current batch
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\nError: Failed to decode\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        n_pos += batch.n_tokens;

        // Sample the next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            printf("\n[End of generation]\n");
            break;
        }

        // Convert token to text and print
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "\nError: Failed to convert token to piece\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
        fflush(stdout);

        // Prepare next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);

        n_decode += 1;
    }

    printf("\n\n");

    const auto t_main_end = ggml_time_us();

    // Print performance statistics
    printf("=== Statistics ===\n");
    printf("Tokens generated: %d\n", n_decode);
    printf("Time: %.2f s\n", (t_main_end - t_main_start) / 1000000.0f);
    printf("Speed: %.2f tokens/s\n\n", n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    // Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("\n=== Inference Complete ===\n");
    return 0;
}
