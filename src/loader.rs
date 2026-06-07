use serde_json::Value;

use crate::tensor::{self, Tensor};
use crate::model::{Model, TransformerBlock};


pub fn load_model() -> Model{
    let file_data: Vec<u8> = std::fs::read("models/model.safetensors").unwrap();

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
    let mut blocks: Vec<TransformerBlock> = Vec::new();
    
    // loop through the 12 transfomers 
    for i in 0..12 {
    let block = TransformerBlock {
        ln_1_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_1.weight", i)),
        ln_1_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_1.bias", i)),
        c_attn_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_attn.weight", i)),
        c_attn_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_attn.bias", i)),
        c_proj_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_proj.weight", i)),
        c_proj_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.attn.c_proj.bias", i)),
        ln_2_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_2.weight", i)),
        ln_2_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.ln_2.bias", i)),
        mlp_fc_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_fc.weight", i)),
        mlp_fc_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_fc.bias", i)),
        mlp_proj_weight: load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_proj.weight", i)),
        mlp_proj_bias: load_tensor(&file_data, &header, header_size, &format!("h.{}.mlp.c_proj.bias", i)),
        };
        blocks.push(block);
    }
    
    let model: Model = Model { wte, wpe, blocks, ln_f_weight, ln_f_bias };

    return model;
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
    let data: Vec<f32> = bytes
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();


    return Tensor { data: data, shape: shape }

}