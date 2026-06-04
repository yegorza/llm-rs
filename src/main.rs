mod tensor;
use tensor::{Tensor, matmul, add, mul};

fn main() {
    // matmul
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = matmul(&a, &b);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    println!("matmul passed");

    // add
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let c = add(&a, &b);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
    println!("add passed");

    // mul
    let c = mul(&a, &b);
    assert_eq!(c.data, vec![10.0, 40.0, 90.0, 160.0]);
    println!("mul passed");

    // transpose
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = a.transpose();
    assert_eq!(b.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(b.shape, vec![3, 2]);
    println!("transpose passed");

    // softmax
    let a = Tensor::new(vec![2.0, 1.0, 0.1], vec![3]);
    let b = a.softmax();
    assert!((b.data.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    println!("softmax passed");

    // gelu
    let a = Tensor::new(vec![0.0], vec![1]);
    let b = a.gelu();
    assert!((b.data[0] - 0.0).abs() < 1e-5);
    println!("gelu passed");

    // layer norm
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let gamma = Tensor::new(vec![1.0; 5], vec![5]);
    let beta = Tensor::new(vec![0.0; 5], vec![5]);
    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!((result.data[2] - 0.0).abs() < 1e-4);
    println!("layer norm passed");

    println!("all tests passed!");
}