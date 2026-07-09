# Benchmarks

## GPT-2 124M — Apple M-series, single-threaded

**2026-06-16**

- 6.12 tokens/sec
- Memory: ~655MB (f32), ~448MB (INT8 quantized)
- KV cache enabled

### GPT-2 Medium (350M)

- Tokens/sec: 2.31
- Config: 24 layers, 16 heads, 1024 embed dim

**2026-07-01**

- After adding

### GPT-2 Medium (350M)

- tokens: 200
- time: 30.25s
- tokens/sec: 6.61

## TinyLlama 1.1B — Apple M-series, single-threaded

**2026-07-09**

### TinyLlama-1.1B (22 layers, 32 heads / 4 KV heads, 2048 embed dim)

- tokens: 200
- time: 62.82s
- tokens/sec: 3.18
