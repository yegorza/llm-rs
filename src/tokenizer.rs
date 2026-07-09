use std::collections::HashMap;

pub struct LlamaTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    merge_ranks: HashMap<(String, String), usize>,
    bos_id: usize,
    special_ids: std::collections::HashSet<usize>,
}

impl LlamaTokenizer {
    pub fn new(tokenizer_json_path: &str) -> Self {
        let file_data = std::fs::read_to_string(tokenizer_json_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&file_data).unwrap();

        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut reverse_vocab: HashMap<usize, String> = HashMap::new();
        for (k, v) in json["model"]["vocab"].as_object().unwrap() {
            let id = v.as_u64().unwrap() as usize;
            vocab.insert(k.clone(), id);
            reverse_vocab.insert(id, k.clone());
        }

        let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
        for (i, merge) in json["model"]["merges"].as_array().unwrap().iter().enumerate() {
            let merge = merge.as_str().unwrap();
            let mut parts = merge.splitn(2, ' ');
            let a = parts.next().unwrap().to_string();
            let b = parts.next().unwrap().to_string();
            merge_ranks.insert((a, b), i);
        }

        let bos_id = vocab["<s>"];

        let mut special_ids = std::collections::HashSet::new();
        if let Some(added) = json["added_tokens"].as_array() {
            for t in added {
                if t["special"].as_bool().unwrap_or(false) {
                    special_ids.insert(t["id"].as_u64().unwrap() as usize);
                }
            }
        }

        Self { vocab, reverse_vocab, merge_ranks, bos_id, special_ids }
    }

    /// Splits a single Unicode character into its raw-byte vocab tokens
    /// (e.g. 'é' -> ["<0xC3>", "<0xA9>"]), used when a char isn't in the base alphabet.
    fn byte_fallback(&self, c: char) -> Vec<String> {
        let mut buf = [0u8; 4];
        c.encode_utf8(&mut buf)
            .bytes()
            .map(|b| format!("<0x{:02X}>", b))
            .collect()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let normalized = format!("▁{}", text.replace(' ', "▁"));

        let mut symbols: Vec<String> = Vec::new();
        for c in normalized.chars() {
            let s = c.to_string();
            if self.vocab.contains_key(&s) {
                symbols.push(s);
            } else {
                symbols.extend(self.byte_fallback(c));
            }
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(i) = best_idx else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        let mut ids: Vec<usize> = Vec::with_capacity(symbols.len() + 1);
        ids.push(self.bos_id);
        ids.extend(symbols.iter().map(|s| self.vocab[s]));
        ids
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        let flush = |byte_buf: &mut Vec<u8>, out: &mut String| {
            if !byte_buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(byte_buf));
                byte_buf.clear();
            }
        };

        for id in token_ids {
            if self.special_ids.contains(id) {
                continue;
            }
            let token = self.reverse_vocab[id].replace('▁', " ");
            if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                    byte_buf.push(byte);
                    continue;
                }
            }
            flush(&mut byte_buf, &mut out);
            out.push_str(&token);
        }
        flush(&mut byte_buf, &mut out);
        out
    }
}

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

    pub fn decode(&self, token_ids: &[usize]) -> String {
        token_ids.iter()
            .map(|id| self.reverse_vocab[id].replace("Ġ", " ").replace("Ċ", "\n"))
            .collect()
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

/// Common interface over the GPT-2 and Llama tokenizers so callers (e.g. the CLI)
/// can pick one at runtime without caring which scheme is behind it.
pub trait TextTokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, token_ids: &[usize]) -> String;
}

impl TextTokenizer for Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize> { self.encode(text) }
    fn decode(&self, token_ids: &[usize]) -> String { self.decode(token_ids) }
}

impl TextTokenizer for LlamaTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> { self.encode(text) }
    fn decode(&self, token_ids: &[usize]) -> String { self.decode(token_ids) }
}