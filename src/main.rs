use std::time::Instant;
use rand::Rng;
use llm_rs::tensor::{Tensor, matmul, add, mul};
use llm_rs::forward::forward;
use llm_rs::model::KVCache;
use llm_rs::tensor::quantize;
use llm_rs::tokenizer::Tokenizer;
use llm_rs::loader;

fn main() {

    // loading initial data
    let model = loader::load_model();
    let tokenizer = Tokenizer::new("models/vocab.json", "models/merges.txt");
    let start = Instant::now();
    let mut cache: Option<KVCache> = None;
    let wte_t = quantize(&model.wte.transpose());

    // inference loop
    let mut token_ids = tokenizer.encode("How many days in a week");
    let mut logits = forward(&model, &token_ids, &mut cache, &wte_t);
    let temperature = 0.8;
    let token_count = 50;

    
    for _ in 0..token_count {
        let scaled: Vec<f32> = logits.data.iter().map(|x| x / temperature).collect();
        let scaled_tensor = Tensor::new(scaled, vec![50257]);
        let probs = scaled_tensor.softmax();
        
        // random weighted sample
        let mut rng = rand::thread_rng();
        let roll: f32 = rng.r#gen();  // random number between 0 and 1
        let mut indexed: Vec<(usize, f32)> = probs.data.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // zero out everything after top k
        let mut filtered = vec![0.0f32; 50257];
        let p = 0.9;
        let mut cumulative = 0.0;
        for i in 0..50257 {
            filtered[indexed[i].0] = indexed[i].1;
            cumulative += indexed[i].1;
            if cumulative > p {
                break;
            }
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
    let elapsed = start.elapsed().as_secs_f32();
    println!("tokens: {}", token_count);
    println!("time: {:.2}s", elapsed);
    println!("tokens/sec: {:.2}", token_count as f32 / elapsed);
}