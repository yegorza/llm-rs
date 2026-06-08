mod tensor;
mod loader;
mod model;
mod forward;
use tensor::{Tensor, matmul, add, mul};

use crate::forward::forward;

fn main() {
    let model = loader::load_model();
    println!("wte shape: {:?}", model.wte.shape);
    println!("wpe shape: {:?}", model.wpe.shape);
    println!("blocks: {}", model.blocks.len());
    println!("first 5 wte values: {:?}", &model.wte.data[..5]);
    let token_ids: Vec<usize> = vec![15496, 995];
    let logits = forward(&model, &token_ids);
    println!("logits shape: {:?}", logits.shape);
    let max_index = logits.data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!("predicted token id: {}", max_index);
}