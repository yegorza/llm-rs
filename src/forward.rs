use crate::{model::{KVCache, Model}, tensor::{QuantizedTensor, Tensor, add, matmul, matmul_quantized, mul}};

pub fn forward(model: &Model, token_ids: &[usize], cache: &mut Option<KVCache>, wte_t: &QuantizedTensor) -> Tensor{
    let cfg = &model.config;

    let position_offset = if cache.is_some() {
        cache.as_ref().unwrap().k[0][0].shape[0]
    } else {
        0
    };

    let mut input: Vec<f32> = Vec::new();
    for (i, id) in token_ids.iter().enumerate(){
        let start = id * cfg.n_embed;
        let end = start + cfg.n_embed;
        let meaning_embedding = &model.wte.data[start..end];
        let pos_i = i + position_offset;
        let position_embedding = &model.wpe.data[pos_i*cfg.n_embed..(pos_i+1)*cfg.n_embed];
        for j in 0..cfg.n_embed {
            input.push(meaning_embedding[j] + position_embedding[j]);
        }
    }
    let mut hidden = Tensor::new(input, vec![token_ids.len(), cfg.n_embed]);

    if cache.is_none() {
        *cache = Some(KVCache {
            k: vec![Vec::new(); cfg.n_layers],
            v: vec![Vec::new(); cfg.n_layers],
        });
    }

    let scale = 1.0 / (cfg.head_dim as f32).sqrt();

    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let ln1_out = hidden.layer_norm(&block.ln_1_weight, &block.ln_1_bias, 1e-5);

        let qkv = add(&matmul_quantized(&ln1_out, &block.c_attn_weight), &block.c_attn_bias);
        let (q, k, v) = split_into_qkv(&qkv);
        let q_heads = split_into_heads(&q, cfg.n_heads);
        let k_heads = split_into_heads(&k, cfg.n_heads);
        let v_heads = split_into_heads(&v, cfg.n_heads);

        if cache.as_ref().unwrap().k[layer_idx].is_empty() {
            for head_idx in 0..cfg.n_heads {
                let c = cache.as_mut().unwrap();
                c.k[layer_idx].push(k_heads[head_idx].clone());
                c.v[layer_idx].push(v_heads[head_idx].clone());
            }
        } else {
            for head_idx in 0..cfg.n_heads {
                let c = cache.as_mut().unwrap();
                c.k[layer_idx][head_idx] = concat(&c.k[layer_idx][head_idx], &k_heads[head_idx]);
                c.v[layer_idx][head_idx] = concat(&c.v[layer_idx][head_idx], &v_heads[head_idx]);
            }
        }

        let mut head_outputs: Vec<Tensor> = Vec::new();

        for i in 0..cfg.n_heads {
            let head_out = flash_attention(
                &q_heads[i],
                &cache.as_ref().unwrap().k[layer_idx][i],
                &cache.as_ref().unwrap().v[layer_idx][i],
                scale
            );
            head_outputs.push(head_out);
        }

        let concatenated = concatenate_heads(&head_outputs);
        let attention_out = add(&matmul_quantized(&concatenated, &block.c_proj_weight), &block.c_proj_bias);
        hidden = add(&hidden, &attention_out);

        let ln2_out = hidden.layer_norm(&block.ln_2_weight, &block.ln_2_bias, 1e-5);

        let fc_out_mul = &matmul_quantized(&ln2_out, &block.mlp_fc_weight);
        let fc_out = add(&fc_out_mul, &block.mlp_fc_bias);
        let gelu_out = fc_out.gelu();
        let mut proj_out = matmul_quantized(&gelu_out, &block.mlp_proj_weight);
        proj_out = add(&proj_out, &block.mlp_proj_bias);
        hidden = add(&hidden, &proj_out);
    }

    hidden = hidden.layer_norm(&model.ln_f_weight, &model.ln_f_bias, 1e-5);
    let logits = matmul_quantized(&hidden, &wte_t);
    let last_row = &logits.data[logits.data.len() - cfg.n_vocab..];
    Tensor::new(last_row.to_vec(), vec![cfg.n_vocab])
}

fn split_into_qkv(qkv_tensor: &Tensor) -> (Tensor, Tensor, Tensor){
    let mut q = Tensor{data: vec![0.0; qkv_tensor.shape[0] * qkv_tensor.shape[1]/3], shape: vec![qkv_tensor.shape[0], qkv_tensor.shape[1]/3]};
    let mut k = Tensor{data: vec![0.0; qkv_tensor.shape[0] * qkv_tensor.shape[1]/3], shape: vec![qkv_tensor.shape[0], qkv_tensor.shape[1]/3]};
    let mut v = Tensor{data: vec![0.0; qkv_tensor.shape[0] * qkv_tensor.shape[1]/3], shape: vec![qkv_tensor.shape[0], qkv_tensor.shape[1]/3]};

    let last_dim = qkv_tensor.shape[qkv_tensor.shape.len() - 1];
    let segment = last_dim / 3;
    for i in 0..qkv_tensor.data.iter().len() / last_dim{
        let start = i * last_dim;
        let end = start + last_dim;
        let row = &qkv_tensor.data[start..end];
        q.data[i*segment..(i+1)*segment].copy_from_slice(&row[0..segment]);
        k.data[i*segment..(i+1)*segment].copy_from_slice(&row[segment..segment*2]);
        v.data[i*segment..(i+1)*segment].copy_from_slice(&row[segment*2..last_dim]);
    }

    (q,k,v)
}

fn split_into_heads(tensor: &Tensor, num_heads: usize) -> Vec<Tensor> {
    let head_dim = tensor.shape[1] / num_heads;
    let mut heads: Vec<Vec<f32>> = vec![Vec::new(); num_heads];
    let last_dim = tensor.shape[tensor.shape.len() - 1];
    for i in 0..tensor.data.iter().len() / last_dim{
        let start = i * last_dim;
        let end = start + last_dim;
        let row = &tensor.data[start..end];
        for j in 0..num_heads{
            heads[j].extend_from_slice(&row[j*head_dim..(j+1)*head_dim]);
        }
    }
    heads.iter()
        .map(|h| Tensor::new(h.clone(), vec![tensor.shape[0], head_dim]))
        .collect()
}

fn concatenate_heads(heads: &Vec<Tensor>) -> Tensor {
    let seq_len = heads[0].shape[0];
    let head_dim = heads[0].shape[1];
    let mut data: Vec<f32> = Vec::new();

    for row in 0..seq_len {
        for head in heads {
            let start = row * head_dim;
            data.extend_from_slice(&head.data[start..start + head_dim]);
        }
    }

    Tensor::new(data, vec![seq_len, heads.len() * head_dim])
}

fn concat(a: &Tensor, b: &Tensor) -> Tensor {
    let mut data = a.data.clone();
    data.extend(&b.data);
    Tensor::new(data, vec![a.shape[0] + b.shape[0], a.shape[1]])
}


fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor{
    let mut result_data: Vec<f32> = Vec::new();
    for i in 0..q.shape[0]{
        let row = &q.data[i*q.shape[1]..(i+1)*q.shape[1]];
        let mut running_max = f32::NEG_INFINITY;
        let mut d_i = 0.0;
        let mut output = vec![0.0; q.shape[1]];

        for j in 0..k.shape[0]{
            if q.shape[0] > 1 && j > i {
                continue;
            }
            let k_row = &k.data[j*k.shape[1]..(j+1)*k.shape[1]];
            let score: f32 = row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
            let new_max = running_max.max(score);
            let old_d_i = d_i;
            d_i = d_i*(running_max - new_max).exp() + (score - new_max).exp();
            let correction = (old_d_i / d_i) * (running_max - new_max).exp();
            let new_weight = (score - new_max).exp() / d_i;
            let v_row = &v.data[j * v.shape[1]..(j + 1) * v.shape[1]];
            for e in 0..q.shape[1] {
                output[e] = output[e] * correction + new_weight * v_row[e];
            }
            running_max = new_max;
        }
        result_data.extend_from_slice(&output);
    }
    return Tensor { data: result_data, shape: vec![q.shape[0], k.shape[1]] };
}