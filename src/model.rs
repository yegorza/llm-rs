use crate::tensor::Tensor;

pub enum ModelType {
    GPT2,
    Llama,
}

pub struct ModelConfig {
    pub model_type: ModelType,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_embed: usize,
    pub n_intermediate: usize,
    pub n_vocab: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

pub struct Model {
    pub config: ModelConfig,
    pub wte: Tensor,
    pub wpe: Option<Tensor>,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Option<Tensor>,
    pub lm_head: Option<Tensor>,
}

pub struct TransformerBlock {
    pub ln_1_weight: Tensor,
    pub ln_1_bias: Option<Tensor>,
    pub ln_2_weight: Tensor,
    pub ln_2_bias: Option<Tensor>,

    pub c_attn_weight: Option<Tensor>,
    pub c_attn_bias: Option<Tensor>,
    pub q_proj: Option<Tensor>,
    pub k_proj: Option<Tensor>,
    pub v_proj: Option<Tensor>,
    pub c_proj_weight: Tensor,
    pub c_proj_bias: Option<Tensor>,

    pub mlp_fc_weight: Option<Tensor>,
    pub mlp_fc_bias: Option<Tensor>,
    pub mlp_proj_weight: Option<Tensor>,
    pub mlp_proj_bias: Option<Tensor>,
    pub gate_proj: Option<Tensor>,
    pub up_proj: Option<Tensor>,
    pub down_proj: Option<Tensor>,
}

pub struct KVCache {
    pub k: Vec<Vec<Tensor>>,
    pub v: Vec<Vec<Tensor>>,
}