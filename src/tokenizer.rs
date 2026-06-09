use std::{collections::HashMap, f32::INFINITY};

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    merge_ranks: HashMap<(String, String), usize>
}

impl Tokenizer{
    pub fn new(vocab_path: &str, merges_path: &str) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut reverse_vocab: HashMap<usize, String> = HashMap::new();

        let file_data = std::fs::read_to_string(vocab_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&file_data).unwrap();
        let merges_data = std::fs::read_to_string(merges_path).unwrap();
        let merges: Vec<(String, String)> = merges_data
            .lines()
            .skip(1)
            .map(|line| {
                let parts: Vec<&str> = line.split(' ').collect();
                (parts[0].to_string(), parts[1].to_string())
            })
            .collect();

        let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
        for (i, merge) in merges.iter().enumerate() {
            merge_ranks.insert(merge.clone(), i);
        }

        for (k,v) in json.as_object().unwrap(){
            let id = v.as_f64().unwrap() as usize;
            vocab.insert(k.clone(), id);
            reverse_vocab.insert(id, k.clone());
        }
        
        Self { vocab, reverse_vocab, merge_ranks}
    }

    pub fn decode(&self, token_id: &usize) -> String {
        let token = self.reverse_vocab[token_id].clone();
        token.replace("Ġ", " ").replace("Ċ", "\n")
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let words: Vec<String> = text.split(' ')
            .enumerate()
            .map(|(i, w)| if i == 0 { w.to_string() } else { format!("Ġ{}", w) })
            .collect();

        let mut all_ids: Vec<usize> = Vec::new();

        for word in &words {
            let mut letters: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            loop {
                let mut cur_rank = usize::MAX;
                let mut cur_pair: Option<usize> = None;
                for i in 0..letters.len() - 1 {
                    let pair = (letters[i].clone(), letters[i + 1].clone());
                    if let Some(&rank) = self.merge_ranks.get(&pair) {
                        if rank < cur_rank {
                            cur_rank = rank;
                            cur_pair = Some(i);
                        }
                    }
                }
                if cur_pair.is_none() {
                    break;
                }
                let i = cur_pair.unwrap();
                let merged = format!("{}{}", letters[i], letters[i + 1]);
                letters[i] = merged;
                letters.remove(i + 1);
            }

            for token in &letters {
                all_ids.push(self.vocab[token]);
            }
        }

        all_ids
    }
}