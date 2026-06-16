mod tensor;
mod loader;
mod model;
mod forward;
mod tokenizer;
use tensor::{Tensor, matmul, add, mul};
use std::time::Instant;
use rand::Rng;


use crate::{forward::forward, model::KVCache, tokenizer::Tokenizer};

fn main() {

    // loading initial data
    let model = loader::load_model();
    let tokenizer = Tokenizer::new("models/vocab.json", "models/merges.txt");
    let start = Instant::now();
    let mut cache: Option<KVCache> = None;
    let wte_t = model.wte.transpose();

    // inference loop
    let mut token_ids = tokenizer.encode("How many days in a week");
    let mut logits = forward(&model, &token_ids, &mut cache, &wte_t);
    let temperature = 0.8;
    for _ in 0..50 {
        let scaled: Vec<f32> = logits.data.iter().map(|x| x / temperature).collect();
        let scaled_tensor = Tensor::new(scaled, vec![50257]);
        let probs = scaled_tensor.softmax();
        
        // random weighted sample
        let mut rng = rand::thread_rng();
        let roll: f32 = rng.r#gen();  // random number between 0 and 1
        let k = 50;
        let mut indexed: Vec<(usize, f32)> = probs.data.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // zero out everything after top k
        let mut filtered = vec![0.0f32; 50257];
        for j in 0..k {
            filtered[indexed[j].0] = indexed[j].1;
        }
        let sum: f32 = filtered.iter().sum();
        let filtered: Vec<f32> = filtered.iter().map(|x| x / sum).collect();
        let mut cumulative = 0.0;
        let mut next_token = 0;
        for (i, p) in filtered.iter().enumerate() {
            cumulative += p;
            if cumulative > roll {
                next_token = i;
                break;
            }
        }
        
        print!("{}", tokenizer.decode(&next_token));
        token_ids.push(next_token);
        logits = forward(&model, &[*token_ids.last().unwrap()], &mut cache, &wte_t);
    }
    println!("took: {:?}", start.elapsed());
}