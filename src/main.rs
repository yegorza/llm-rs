mod tensor;
mod loader;
mod model;
use tensor::{Tensor, matmul, add, mul};

fn main() {
    let model = loader::load_model();
    println!("wte shape: {:?}", model.wte.shape);
    println!("blocks: {}", model.blocks.len());
    println!("first 5 wte values: {:?}", &model.wte.data[..5]);
}