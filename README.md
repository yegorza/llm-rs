# llm-rs

A from-scratch LLM inference engine written in Rust, with no ML framework (no PyTorch/tch, no ggml/candle). Tensors, matrix multiplication, attention, sampling, and tokenization are implemented; matmul is BLAS-backed via Apple's Accelerate framework (`cblas_sgemm`).

It loads weights directly from `.safetensors` files and supports two model architectures through a shared forward pass:

- **GPT-2** (`src/loader.rs::load_model`)
- **Llama / TinyLlama** (`src/loader.rs::load_llama`)

## Features

- Hand-rolled `Tensor` type (`src/tensor.rs`) with BLAS-backed matmul
- Flash attention with tiling (`forward.rs::flash_attention`)
- KV caching for incremental decoding
- RoPE positional encoding (`tensor.rs::apply_rope`), using the HF `rotate_half` convention (matches Llama/TinyLlama safetensors checkpoints)
- Grouped-query attention (GQA) for Llama
- RMSNorm (Llama) and LayerNorm (GPT-2)
- Byte-level BPE tokenizer for GPT-2, and a SentencePiece-BPE tokenizer with byte-fallback for Llama (`src/tokenizer.rs`)
- Top-p (nucleus) sampling
- Speculative decoding (GPT-2 only — requires a small draft model sharing the main model's vocab)

## Platform support

**macOS only, for now.** Matmul is linked against Apple's Accelerate framework (`build.rs`, `#[cfg(target_os = "macos")]` in `src/tensor.rs`). There's currently no Linux/Windows BLAS backend, so the crate won't compile on those platforms.

## Building

```sh
cargo build --release --bin llm-rs --features cli
```

There's also a `napi-binding` feature exposing a Node.js native addon (`src/lib.rs`) — it exists but is less polished/maintained than the CLI path.

## Running

```sh
./target/release/llm-rs --model llama        # TinyLlama-1.1B (default)
./target/release/llm-rs --model gpt2         # GPT-2-medium
./target/release/llm-rs --model gpt2 -s      # GPT-2-medium with speculative decoding
./target/release/llm-rs --prompt "The capital of France is" --tokens 50
```

| Flag                  | Default                            | Meaning                                                         |
| --------------------- | ---------------------------------- | --------------------------------------------------------------- |
| `--model`, `-m`       | `llama`                            | `gpt2` or `llama`                                               |
| `--prompt`, `-p`      | `"How many days in a week"`        | Prompt text                                                     |
| `--tokens`, `-n`      | `200`                              | Number of tokens to generate                                    |
| `--speculative`, `-s` | off                                | GPT-2 only — needs a draft model sharing the main model's vocab |
| `--weights`, `-w`     | model-dependent default path below | Override the weights `.safetensors` path                        |
| `--tokenizer`, `-t`   | `models/llama-tokenizer.json`      | Llama only — override the tokenizer JSON path                   |
| `--vocab`             | `models/vocab.json`                | GPT-2 only — override the BPE vocab path                        |
| `--merges`            | `models/merges.txt`                | GPT-2 only — override the BPE merges path                       |

## Model files

`models/` is gitignored and required at runtime — model weights aren't committed to this repo. Download the following and place them at these exact paths (filenames are hardcoded in `src/loader.rs`/`src/main.rs`):

**Llama path:**

- `models/tinyllama-1b.safetensors` — single-file `.safetensors` weights (not sharded)
- `models/llama-tokenizer.json`

Both are available from the public [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) repo on Hugging Face Hub (`model.safetensors` and `tokenizer.json` — rename to the paths above).

**GPT-2 path:**

- `models/gpt2-medium.safetensors`
- `models/vocab.json`
- `models/merges.txt`

Available from the public [`gpt2-medium`](https://huggingface.co/gpt2-medium) repo on Hugging Face Hub.

**Speculative decoding (optional, GPT-2 only):**

- `models/model.safetensors` — a small GPT-2 draft model (e.g. base `gpt2`), same vocab as `gpt2-medium`

## Benchmarks

Single-threaded, Apple M-series (see `benchmark.md` for details and history):

| Model               | Tokens/sec |
| ------------------- | ---------- |
| GPT-2-medium (350M) | ~6.6       |
| TinyLlama-1.1B      | ~3.2       |

## License

MIT — see [LICENSE](./LICENSE).
