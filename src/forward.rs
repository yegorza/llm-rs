use crate::{model::{KVCache, Model}, tensor::{Tensor, add, matmul, mul}};

pub fn forward(model: &Model, token_ids: &[usize], cache: &mut Option<KVCache>, wte_t: &Tensor) -> Tensor{
    // look up token ids in wte and position wpe and then add
    let mut input: Vec<f32> = Vec::new();
    let position_offset = if cache.is_some() {
        cache.as_ref().unwrap().k[0][0].shape[0] // how many tokens already cached
    } else {
        0
    };
    for (i, id) in token_ids.iter().enumerate(){
        let start = id * 768;
        let end = start + 768;
        let meaning_embedding = &model.wte.data[start..end];
        let pos_i = i + position_offset; // figure out true position of new token after cached values
        let position_embedding = &model.wpe.data[pos_i*768..(pos_i+1)*768];
        for j in 0..768 {
            input.push(meaning_embedding[j] + position_embedding[j]);
        }
    }
    let mut hidden = Tensor::new(input, vec![token_ids.len(), 768]);

    // init the cache
    if cache.is_none() {
        *cache = Some(KVCache {
            k: vec![Vec::new(); 12],
            v: vec![Vec::new(); 12],
        });
    }

    for (layer_idx, block) in model.blocks.iter().enumerate() {
        // layer norm 1
        let ln1_out = hidden.layer_norm(&block.ln_1_weight, &block.ln_1_bias, 1e-5);
        
        // attention
        let qkv = add(&matmul(&ln1_out, &block.c_attn_weight), &block.c_attn_bias);
        let (q, k, v) = split_into_qkv(&qkv);
        let q_heads = split_into_heads(&q, 12);
        let k_heads = split_into_heads(&k, 12);
        let v_heads = split_into_heads(&v, 12);

        if cache.as_ref().unwrap().k[layer_idx].is_empty() {
            for head_idx in 0..12 {
                let c = cache.as_mut().unwrap();
                c.k[layer_idx].push(k_heads[head_idx].clone());
                c.v[layer_idx].push(v_heads[head_idx].clone());
            }
        } else {
            for head_idx in 0..12 {
                let c = cache.as_mut().unwrap();
                c.k[layer_idx][head_idx] = concat(&c.k[layer_idx][head_idx], &k_heads[head_idx]);
                c.v[layer_idx][head_idx] = concat(&c.v[layer_idx][head_idx], &v_heads[head_idx]);
            }
        }

        let mut head_outputs: Vec<Tensor> = Vec::new();

        for i in 0..12{
            let mut scores = matmul(&q_heads[i], &cache.as_ref().unwrap().k[layer_idx][i].transpose());
            let div = Tensor{data: vec![1.0/8.0; scores.shape[0]*scores.shape[1]], shape: scores.shape.clone()};
            scores = mul(&scores, &div);
            if scores.shape[0] > 1 {
                scores = scores.apply_causal_mask();
            }
            let weights = scores.softmax();
            let head_out = matmul(&weights, &cache.as_ref().unwrap().v[layer_idx][i]);
            head_outputs.push(head_out);
        }

        let concatenated = concatenate_heads(&head_outputs);
        let attention_out = add(&matmul(&concatenated, &block.c_proj_weight), &block.c_proj_bias);
        hidden = add(&hidden, &attention_out);
        
        // layer norm 2
        // println!("hidden shape: {:?}", hidden.shape);
        // println!("&block.ln_2_weight shape: {:?}", &block.ln_2_weight.shape);
        let ln2_out = hidden.layer_norm(&block.ln_2_weight, &block.ln_2_bias, 1e-5);
        
        // feedforward
        // println!("ln2_out shape: {:?}", &ln2_out.shape);
        // println!("&block.mlp_fc_weight shape: {:?}", &block.mlp_fc_weight.shape);
        let fc_out_mul = &matmul(&ln2_out, &block.mlp_fc_weight);
        // println!("&fc_out_mul shape: {:?}", &fc_out_mul.shape);
        let fc_out = add(&fc_out_mul, &block.mlp_fc_bias);
        let gelu_out = fc_out.gelu();
        let mut proj_out = matmul(&gelu_out, &block.mlp_proj_weight);
        proj_out = add(&proj_out, &block.mlp_proj_bias);
        hidden = add(&hidden, &proj_out);
    }

    hidden = hidden.layer_norm(&model.ln_f_weight, &model.ln_f_bias, 1e-5);
    let logits = matmul(&hidden, &wte_t);
    let last_row = &logits.data[logits.data.len() - 50257..];
    return Tensor::new(last_row.to_vec(), vec![50257]);

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

    return (q,k,v);
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
    return heads.iter()
        .map(|h| Tensor::new(h.clone(), vec![tensor.shape[0], head_dim]))
        .collect();

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