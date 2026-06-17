use crate::tensor::{Tensor, QuantizedTensor};

pub struct ModelConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_embed: usize,
    pub n_vocab: usize,
    pub head_dim: usize,
}

pub struct Model {
    pub config: ModelConfig,
    pub wte: Tensor,
    pub wpe: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
}

pub struct TransformerBlock {
    pub ln_1_weight: Tensor,
    pub ln_1_bias: Tensor,
    pub c_attn_weight: QuantizedTensor,
    pub c_attn_bias: Tensor,
    pub c_proj_weight: QuantizedTensor,
    pub c_proj_bias: Tensor,
    pub ln_2_weight: Tensor,
    pub ln_2_bias: Tensor,
    pub mlp_fc_weight: QuantizedTensor,
    pub mlp_fc_bias: Tensor,
    pub mlp_proj_weight: QuantizedTensor,
    pub mlp_proj_bias: Tensor,
}

pub struct KVCache {
    pub k: Vec<Vec<Tensor>>,  // k[layer][head]
    pub v: Vec<Vec<Tensor>>,  // v[layer][head]
}