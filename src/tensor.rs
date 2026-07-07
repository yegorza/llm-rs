use core::panic;
use std::f32::NEG_INFINITY;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>
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

    pub fn rms_norm(&self, gamma: &Tensor, epsilon: f32) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        let last_dim = self.shape[self.shape.len() - 1];
        let n = last_dim as f32;
        for i in 0..self.data.len() / last_dim{
            let start = i * last_dim;
            let end = start + last_dim;
            let row: &[f32] = &self.data[start..end];
            let mut mean_sq = 0.0;
            for val in row.iter(){
                mean_sq += (val).powi(2);
            }
            mean_sq = mean_sq / n;
    
            for (j, _val) in row.iter().enumerate(){
                result.data[j+start] = (self.data[j+start]/(mean_sq + epsilon).sqrt())*gamma.data[j];
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


pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0] as i32;
    let n = b.shape[1] as i32;
    let k = a.shape[1] as i32;
    let mut result = vec![0.0f32; (m * n) as usize];
    
    unsafe {
        cblas_sgemm(
            101,        // CblasRowMajor
            111,        // CblasNoTrans
            111,        // CblasNoTrans
            m, n, k,
            1.0,        // alpha
            a.data.as_ptr(), k,
            b.data.as_ptr(), n,
            0.0,        // beta
            result.as_mut_ptr(), n,
        );
    }
    
    Tensor::new(result, vec![m as usize, n as usize])
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
