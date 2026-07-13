# Benchmarks

## Apple M1, single-threaded, release build

### Optimization Progression — GPT-2 Small (124M)

| Date   | Optimization                                     | tok/s | Notes                          |
| ------ | ------------------------------------------------ | ----- | ------------------------------ |
| Jun 7  | Baseline (direct .data[] indexing, release mode) | 6.12  | First recorded benchmark       |
| Jun 18 | Loop reorder + unsafe get_unchecked              | 6.39  | Minor gain                     |
| Jun 18 | Flash attention (tiled, T=32)                    | 6.75  | O(n) memory, slight speed gain |
| Jun 26 | Apple Accelerate (cblas_sgemm)                   | 30.34 | +5x, M1 AMX unit               |

### Multi-Model Comparison (Apple Accelerate enabled)

| Model        | Parameters | Architecture                       | tok/s (50 tokens) | tok/s (200 tokens) |
| ------------ | ---------- | ---------------------------------- | ----------------- | ------------------ |
| GPT-2 Small  | 124M       | 12 layers, 12 heads, 768 dim       | 30.34             | 18.77              |
| GPT-2 Medium | 350M       | 24 layers, 16 heads, 1024 dim      | ~18               | 6.61               |
| TinyLlama    | 1.1B       | 22 layers, 32Q/4KV heads, 2048 dim | ~6                | 3.18               |

### Speculative Decoding (GPT-2 Small draft + Medium target)

| Metric             | Value                                             |
| ------------------ | ------------------------------------------------- |
| tok/s              | 0.82                                              |
| Acceptance rate    | 30.5%                                             |
| vs baseline Medium | Slower                                            |
| Notes              | CPU overhead exceeds gains at 30% acceptance rate |

### Memory Usage — GPT-2 Small

| Configuration  | Memory |
| -------------- | ------ |
| f32 weights    | ~655MB |
| INT8 quantized | ~448MB |
| Savings        | ~30%   |

### Key Findings

- Apple Accelerate BLAS was the single biggest optimization — 5x speedup from one change
- Flash attention saves O(n²) → O(n) memory with minimal speed impact at short sequences
- Speculative decoding requires >60% acceptance rate to be worthwhile on CPU
- tok/s degrades with sequence length due to growing KV cache attention cost
