use serde_json::Value;

use crate::tensor::Tensor;
use crate::model::{Model, ModelConfig, ModelType, TransformerBlock};


pub fn load_model(file_path: &str) -> Model{
    let file_data: Vec<u8> = std::fs::read(file_path).unwrap();

    let header_size = u64::from_le_bytes([
        file_data[0], file_data[1], file_data[2], file_data[3],
        file_data[4], file_data[5], file_data[6], file_data[7],
    ]);

    let header_str = String::from_utf8(file_data[8..8 + header_size as usize].to_vec()).unwrap();
    let header: Value = serde_json::from_str(&header_str).unwrap();

    // load top level
    let wte = load_tensor(&file_data, &header, header_size, "wte.weight");
    let wpe = load_tensor(&file_data, &header, header_size, "wpe.weight");
    let ln_f_weight = load_tensor(&file_data, &header, header_size, "ln_f.weight");
    let ln_f_bias = load_tensor(&file_data, &header, header_size, "ln_f.bias");

    let n_vocab = wte.shape[0];
    let n_embed = wte.shape[1];
    let n_layers = header.as_object().unwrap().keys()
        .filter(|k| k.starts_with("h.") && k.ends_with(".ln_1.weight"))
        .count();
    let head_dim = 64; // GPT-2 always uses 64
    let n_heads = n_embed / head_dim;

    let mut blocks: Vec<TransformerBlock> = Vec::new();

    for i in 0..n_layers {
    let block = TransformerBlock {
        ln_1_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_1.weight", i)),
        ln_1_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_1.bias", i))),
        c_attn_weight: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_attn.weight", i))),
        c_attn_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_attn.bias", i))),
        c_proj_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_proj.weight", i)),
        c_proj_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_proj.bias", i))),
        ln_2_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_2.weight", i)),
        ln_2_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_2.bias", i))),
        mlp_fc_weight: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_fc.weight", i))),
        mlp_fc_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_fc.bias", i))),
        mlp_proj_weight: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_proj.weight", i))),
        mlp_proj_bias: Some(load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_proj.bias", i))),
        q_proj: None,
        k_proj: None,
        v_proj: None,
        gate_proj: None,
        up_proj: None,
        down_proj: None,
        };
        blocks.push(block);
    }
    
    let config = ModelConfig { model_type: ModelType::GPT2, n_layers, n_heads, n_embed, n_vocab, head_dim, n_kv_heads: n_heads, n_intermediate: 4 * n_embed, rope_theta: 10000.0, rms_norm_eps: 1e-5 };
    let model: Model = Model { config, wte, wpe: Some(wpe), blocks, ln_f_weight, ln_f_bias: Some(ln_f_bias), lm_head: None };

    return model;
}


pub fn load_llama(path: &str) -> Model {
    let file_data = std::fs::read(path).unwrap();
    let header_size = u64::from_le_bytes([
        file_data[0], file_data[1], file_data[2], file_data[3],
        file_data[4], file_data[5], file_data[6], file_data[7],
    ]);

    let header_str = String::from_utf8(file_data[8..8 + header_size as usize].to_vec()).unwrap();
    let header: Value = serde_json::from_str(&header_str).unwrap();

    let wte = load_tensor(&file_data, &header, header_size, "model.embed_tokens.weight");
    let ln_f_weight = load_tensor(&file_data, &header, header_size, "model.norm.weight");

    let mut blocks = Vec::new();
    for i in 0..22 {
        let prefix = format!("model.layers.{}", i);
        let block = TransformerBlock {
            ln_1_weight: load_tensor(&file_data, &header, header_size, &format!("{}.input_layernorm.weight", prefix)),
            ln_1_bias: None,
            ln_2_weight: load_tensor(&file_data, &header, header_size, &format!("{}.post_attention_layernorm.weight", prefix)),
            ln_2_bias: None,
            q_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.self_attn.q_proj.weight", prefix))),
            k_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.self_attn.k_proj.weight", prefix))),
            v_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.self_attn.v_proj.weight", prefix))),
            c_proj_weight: load_tensor(&file_data, &header, header_size, &format!("{}.self_attn.o_proj.weight", prefix)),
            c_proj_bias: None,
            gate_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.mlp.gate_proj.weight", prefix))),
            up_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.mlp.up_proj.weight", prefix))),
            down_proj: Some(load_tensor(&file_data, &header, header_size, &format!("{}.mlp.down_proj.weight", prefix))),
            c_attn_weight: None,
            c_attn_bias: None,
            mlp_fc_weight: None,
            mlp_fc_bias: None,
            mlp_proj_weight: None,
            mlp_proj_bias: None,
        };
        blocks.push(block);
    }

    Model {
        config: ModelConfig {
            model_type: ModelType::Llama,
            n_layers: 22,
            n_heads: 32,
            n_kv_heads: 4,
            n_embed: 2048,
            n_intermediate: 5632,
            n_vocab: 32000,
            head_dim: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        },
        wte,
        wpe: None,
        blocks,
        ln_f_weight,
        ln_f_bias: None,
        lm_head: Some(load_tensor(&file_data, &header, header_size, "lm_head.weight")),
    }
}

fn load_tensor(file_data: &[u8], header: &Value, header_size: u64, name: &str) -> Tensor{
    let tensor_info = &header[name];
    let offsets = &tensor_info["data_offsets"];
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;
    let shape: Vec<usize> = tensor_info["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let data_start = 8 + header_size as usize + start;
    let data_end = 8 + header_size as usize + end;
    let bytes = &file_data[data_start..data_end];

    let dtype = tensor_info["dtype"].as_str().unwrap();
    let data: Vec<f32> = if dtype == "F32" {
        bytes.chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else if dtype == "BF16" {
        bytes.chunks(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()
    } else {
        panic!("Unsupported dtype: {}", dtype);
    };


    return Tensor { data: data, shape: shape }

}