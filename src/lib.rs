use rand::Rng;
use crate::{model::KVCache, tensor::Tensor, tokenizer::Tokenizer};

#[cfg(feature = "napi-binding")]
#[macro_use]
extern crate napi_derive;

pub mod tensor;
pub mod model;
pub mod loader;
pub mod forward;
pub mod tokenizer;

#[cfg_attr(feature = "napi-binding", napi)]
pub fn generate(prompt: String, max_tokens: i32) -> String {
    let model = loader::load_model("models/gpt2-medium.safetensors");
    let tokenizer = Tokenizer::new("models/vocab.json", "models/merges.txt");
    let mut cache: Option<KVCache> = None;
    let wte_t = model.wte.transpose();

    let mut token_ids = tokenizer.encode(&prompt);
    let mut logits = forward::forward(&model, &token_ids, &mut cache, &wte_t, false);
    let temperature = 0.8;
    let mut output = String::new();

    for _ in 0..max_tokens {
        let scaled: Vec<f32> = logits.data.iter().map(|x| x / temperature).collect();
        let scaled_tensor = Tensor::new(scaled, vec![50257]);
        let probs = scaled_tensor.softmax();

        let mut rng = rand::thread_rng();
        let roll: f32 = rng.r#gen();
        let mut indexed: Vec<(usize, f32)> = probs.data.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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

        output.push_str(&tokenizer.decode(&next_token));
        token_ids.push(next_token);
        logits = forward::forward(&model, &[*token_ids.last().unwrap()], &mut cache, &wte_t,false);
    }

    output
}