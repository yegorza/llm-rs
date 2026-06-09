use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
}

impl Tokenizer{
    pub fn new(vocab_path: &str) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut reverse_vocab: HashMap<usize, String> = HashMap::new();        
        let file_data = std::fs::read_to_string(vocab_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&file_data).unwrap();

        for (k,v) in json.as_object().unwrap(){
            let id = v.as_f64().unwrap() as usize;
            vocab.insert(k.clone(), id);
            reverse_vocab.insert(id, k.clone());
        }
        
        Self { vocab, reverse_vocab }
    }

    pub fn decode(&self, token_id: &usize) -> String {
        let token = self.reverse_vocab[token_id].clone();
        token.replace("Ġ", " ").replace("Ċ", "\n")
    }
}