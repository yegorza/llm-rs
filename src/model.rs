use crate::Tensor;
pub struct Model {
    pub wte: Tensor,      // [50257, 768]
    pub wpe: Tensor,      // [1024, 768]

    pub blocks: Vec<TransformerBlock>,

    // final layer norm
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
}

pub struct TransformerBlock {
    pub ln_1_weight: Tensor,
    pub ln_1_bias: Tensor,
    pub c_attn_weight: Tensor,
    pub c_attn_bias: Tensor,
    pub c_proj_weight: Tensor,
    pub c_proj_bias: Tensor,
    pub ln_2_weight: Tensor,
    pub ln_2_bias: Tensor,
    pub mlp_fc_weight: Tensor,
    pub mlp_fc_bias: Tensor,
    pub mlp_proj_weight: Tensor,
    pub mlp_proj_bias: Tensor,
}