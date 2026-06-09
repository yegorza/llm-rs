mod tensor;
mod loader;
mod model;
mod forward;
mod tokenizer;
use tensor::{Tensor, matmul, add, mul};
use std::time::Instant;


use crate::{forward::forward, tokenizer::Tokenizer};

fn main() {

    // loading initial data
    let model = loader::load_model();
    let tokenizer = Tokenizer::new("models/vocab.json");
    let start = Instant::now();

    // inference loop
    let mut token_ids: Vec<usize> = vec![15496, 995]; 
    for _ in 0..10 {
        let logits = forward(&model, &token_ids);
        let next_token = logits.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        print!("{}", tokenizer.decode(&next_token));
        token_ids.push(next_token);
    }
    
}