use core::panic;

#[derive(Debug)]
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

    pub fn softmax(&self) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut d = 0.0;
        for val in self.data.iter(){
            d += (val - max).exp()
        }
        for (i, val) in self.data.iter().enumerate(){
            result.data[i] = (val - max).exp() / d;
        }
        
        return result;
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor{
        let mut result: Tensor = Tensor::new(self.data.clone(), self.shape.clone());
        let n = self.data.len() as f32;
        let mean: f32 = self.data.iter().sum::<f32>() / n;
        let mut variance = 0.0;
        for val in self.data.iter(){
            variance += (val - mean).powi(2);
        }
        variance = variance / n;

        for (i, _val) in self.data.iter().enumerate(){
            result.data[i] = ((self.data[i] - mean)/(variance + epsilon).sqrt())*gamma.data[i] + beta.data[i];
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

}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut result = Tensor::new(vec![0.0], vec![a.shape[0], b.shape[1]]);
    result.zeros();
    for row in 0..a.shape[0]{
        for col in 0..b.shape[1]{
            let mut sum: f32 = 0.0;
            for i in 0..a.shape[1]{
                sum += a.get(vec![row,i]) * b.get(vec![i, col]); 
            }
            result.set(vec![row, col], sum);
        }
    }
    return result;
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    if a.shape != b.shape {
        panic!("Shapes do not match");
    }
    let data: Vec<f32> = a.data.iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    Tensor::new(data, a.shape.clone())
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
