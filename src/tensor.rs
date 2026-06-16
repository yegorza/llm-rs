use core::panic;
use std::f32::NEG_INFINITY;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>
}

pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}

impl Tensor{
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape
        }
    }

    pub fn zeros(&mut self) {
        self.data = vec![0.0; self.shape.iter().product()]
    }

    pub fn reshape(&mut self, shape: Vec<usize>){
        let product: usize= shape.iter().product();
        if product != self.data.len() {
            panic!("Cannot reshape tensor of size {} into shape {:?}", self.data.len(), shape);
        }
        self.shape = shape
    }

    pub fn get(&self, indices: Vec<usize>) -> f32{
        let mut index = 0;
        let mut product = 1;
        for (i, val) in self.shape.iter().enumerate().rev() {
            index += indices[i] * product;
            product = product * val;
        }
        return self.data[index];
    }

    pub fn set(&mut self, indices: Vec<usize>, value: f32){
        let mut index = 0;
        let mut product = 1;
        for (i, val) in self.shape.iter().enumerate().rev() {
            index += indices[i] * product;
            product = product * val;
        }
        self.data[index] = value;
    }

    pub fn transpose(&self) -> Tensor{
        let mut result = Tensor::new(vec![0.0; self.shape[0] * self.shape[1]], vec![self.shape[1], self.shape[0]]);
        for row in 0..self.shape[0]{
            for col in 0..self.shape[1]{
                result.set(vec![col, row], self.get(vec![row,col]));
            }
        }
        return result;
    }

    pub fn softmax(&self) -> Tensor {
        let mut result = Tensor::new(self.data.clone(), self.shape.clone());
        let last_dim = self.shape[self.shape.len() - 1];
        for i in 0..self.data.len() / last_dim {
            let start = i * last_dim;
            let end = start + last_dim;
            let row = &self.data[start..end];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut d = 0.0;
            for val in row.iter() {
                d += (val - max).exp();
            }
            for (j, val) in row.iter().enumerate() {
                result.data[start + j] = (val - max).exp() / d;
            }
        }
        return result;
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        let last_dim = self.shape[self.shape.len() - 1];
        let n = last_dim as f32;
        for i in 0..self.data.len() / last_dim{
            let start = i * last_dim;
            let end = start + last_dim;
            let row: &[f32] = &self.data[start..end];
            let mean: f32 = row.iter().sum::<f32>() / n;
            let mut variance = 0.0;
            for val in row.iter(){
                variance += (val - mean).powi(2);
            }
            variance = variance / n;
    
            for (j, _val) in row.iter().enumerate(){
                result.data[j+start] = ((self.data[j+start] - mean)/(variance + epsilon).sqrt())*gamma.data[j] + beta.data[j];
            }
        }

        return result;

    }

    pub fn gelu(&self) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        for (i, val) in self.data.iter().enumerate(){
            result.data[i] = 0.5 * val * (1.0 + (0.7978845608 * (val + 0.044715 * val.powi(3))).tanh());
        }

        return result;
    }

    pub fn apply_causal_mask(&self) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        let last_dim = self.shape[self.shape.len() - 1];
         for i in 0..self.shape[0]{
            let start = i * last_dim;
            for j in (i+1)..last_dim{
                result.data[start + j] = f32::NEG_INFINITY;
            }
        }

        return result;
    }

}


pub fn quantize(tensor: &Tensor) -> QuantizedTensor {
    let absmax = tensor.data.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let data: Vec<i8> = tensor.data.iter().map(|x| (x / absmax * 127.0).round() as i8).collect();
    QuantizedTensor { data, scale: absmax, shape: tensor.shape.clone() }

}
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut result = Tensor::new(vec![0.0], vec![a.shape[0], b.shape[1]]);
    result.zeros();
    for row in 0..a.shape[0] {
        for k in 0..a.shape[1] {
            let a_val = a.data[row * a.shape[1] + k];
            for col in 0..b.shape[1] {
                result.data[row * b.shape[1] + col] += a_val * b.data[k * b.shape[1] + col];
            }
        }
    }
    return result;
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    if a.shape.len() == 2 && b.shape.len() == 1 {
        let mut result = Tensor::new(a.data.clone(), a.shape.clone());
        let last_dim = a.shape[a.shape.len() - 1];
        for i in 0..a.data.iter().len() / last_dim{
            let start = i * last_dim;
            let end = start + last_dim;
            let row = &a.data[start..end];
            for (j,_val) in row.iter().enumerate(){
                result.data[start + j] = a.data[start + j] + b.data[j];
            }
        }
        return result;
    }else{
        if a.shape != b.shape {
            panic!("Shapes do not match");
        }
        let data: Vec<f32> = a.data.iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Tensor::new(data, a.shape.clone())
    }
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    if a.shape != b.shape {
        panic!("Shapes do not match");
    }
    let data: Vec<f32> = a.data.iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    Tensor::new(data, a.shape.clone())
}

pub fn matmul_quantized(a: &Tensor, b: &QuantizedTensor) -> Tensor{
    let mut result = Tensor::new(vec![0.0], vec![a.shape[0], b.shape[1]]);
    let factor = b.scale / 127.0;
    result.zeros();
    for row in 0..a.shape[0] {
        for k in 0..a.shape[1] {
            let a_val = a.data[row * a.shape[1] + k];
            for col in 0..b.shape[1] {
                result.data[row * b.shape[1] + col] += a_val * b.data[k * b.shape[1] + col] as f32 * factor;
            }
        }
    }
    return result;
}