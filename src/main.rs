mod tensor;
mod loader;
mod model;
mod forward;
mod tokenizer;
use tensor::{Tensor, matmul, add, mul};
use std::time::Instant;


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
    for _ in 0.. 50{
        let next_token = logits.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        print!("{}", tokenizer.decode(&next_token));
        token_ids.push(next_token);
        logits = forward(&model, &[*token_ids.last().unwrap()], &mut cache, &wte_t);
    }
    println!("took: {:?}", start.elapsed());
}