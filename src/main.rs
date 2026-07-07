use std::io::Write;
use std::time::Instant;
use llm_rs::tensor::Tensor;
use llm_rs::forward::forward;
use llm_rs::model::{KVCache, Model};
use llm_rs::tokenizer::Tokenizer;
use llm_rs::loader;
use rand::Rng;

fn main() {

    // Speculative decoding is opt-in behind the --speculative (-s) flag; otherwise
    // we run top-p sampling with the main model only.
    let speculative = std::env::args().any(|a| a == "--speculative" || a == "-s");

    // loading initial data
    let main_model = loader::load_llama("models/tinyllama-1b.safetensors");

    let tokenizer = Tokenizer::new("models/vocab.json", "models/merges.txt");

    let main_wte_t = main_model.wte.transpose();

    let token_ids = tokenizer.encode("How many days in a week");
    let initial_len = token_ids.len();
    let token_count = 200;

    if speculative {
        run_speculative(&main_model, &main_wte_t, &tokenizer, token_ids, initial_len, token_count);
    } else {
        run_sample(&main_model, &main_wte_t, &tokenizer, token_ids, initial_len, token_count);
    }
}


fn run_sample(
    main_model: &Model,
    main_wte_t: &Tensor,
    tokenizer: &Tokenizer,
    mut token_ids: Vec<usize>,
    initial_len: usize,
    token_count: usize,
) {
    let start = Instant::now();
    let mut main_cache: Option<KVCache> = None;
    let n_vocab = main_model.config.n_vocab;

    let temperature = 0.8;
    let p = 0.9; // nucleus (top-p) threshold
    let mut rng = rand::thread_rng();

    // Prefill with all but the last prompt token; the loop re-feeds the last token.
    let prefill = &token_ids[..token_ids.len() - 1];
    let _ = forward(main_model, prefill, &mut main_cache, main_wte_t, false);

    while token_ids.len() < initial_len + token_count {
        let last_token = *token_ids.last().unwrap();
        let logits = forward(main_model, &[last_token], &mut main_cache, main_wte_t, false);

        // temperature-scaled softmax
        let scaled: Vec<f32> = logits.data.iter().map(|x| x / temperature).collect();
        let probs = Tensor::new(scaled, vec![n_vocab]).softmax();

        // sort descending and keep the smallest set whose mass exceeds p (nucleus)
        let mut indexed: Vec<(usize, f32)> =
            probs.data.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered = vec![0.0f32; n_vocab];
        let mut cumulative = 0.0;
        for &(idx, prob) in &indexed {
            filtered[idx] = prob;
            cumulative += prob;
            if cumulative > p {
                break;
            }
        }

        // renormalize the kept mass and sample from it
        let sum: f32 = filtered.iter().sum();
        let roll: f32 = rng.r#gen();
        let mut cumulative = 0.0;
        let mut next = 0;
        for (i, prob) in filtered.iter().enumerate() {
            cumulative += prob / sum;
            if cumulative > roll {
                next = i;
                break;
            }
        }

        print!("{}", tokenizer.decode(&next));
        std::io::stdout().flush().unwrap();
        token_ids.push(next);
    }

    let elapsed = start.elapsed().as_secs_f32();
    let generated = token_ids.len() - initial_len;
    println!();
    println!("tokens: {}", generated);
    println!("time: {:.2}s", elapsed);
    println!("tokens/sec: {:.2}", generated as f32 / elapsed);
}


fn run_speculative(
    main_model: &Model,
    main_wte_t: &Tensor,
    tokenizer: &Tokenizer,
    mut token_ids: Vec<usize>,
    initial_len: usize,
    token_count: usize,
) {
    let small_model = loader::load_model("models/model.safetensors");
    let start = Instant::now();

    let mut small_cache: Option<KVCache> = None;
    let mut main_cache: Option<KVCache> = None;

    let small_wte_t = small_model.wte.transpose();

    let k = 5; // draft length
    let n_vocab = main_model.config.n_vocab;
    let temperature = 0.8;
    let top_p = 0.9;
    let mut rng = rand::thread_rng();

    // Invariant maintained across the loop: each cache holds exactly the committed
    // tokens *except the last one* (length == token_ids.len() - 1). Every iteration
    // re-feeds the last committed token, so both models stay perfectly aligned in
    // position and we never need to carry stale logits between iterations.
    //
    // Prefill both caches with all but the last prompt token.
    let prefill = &token_ids[..token_ids.len() - 1];
    let _ = forward(&small_model, prefill, &mut small_cache, &small_wte_t, false);
    let _ = forward(main_model, prefill, &mut main_cache, main_wte_t, false);

    let mut total_draft = 0;
    let mut total_accepted = 0;

    while token_ids.len() < initial_len + token_count {
        let last_token = *token_ids.last().unwrap();

        // Draft k tokens with the small model. This re-feeds last_token (regenerating
        // its position) and then each drafted token, leaving small_cache holding
        // [last_token, draft[0..k]] on top of the previously committed prefix.
        let draft = draft_tokens(&small_model, &mut small_cache, last_token, k, &small_wte_t, temperature, top_p, &mut rng);

        // Verify with a single main forward over [last_token, draft...]. Row j of the
        // output is the main model's distribution for the token at draft position j,
        // so draft[j] is accepted iff it matches the sampled token from the main model.
        let mut verify_seq = Vec::with_capacity(k + 1);
        verify_seq.push(last_token);
        verify_seq.extend_from_slice(&draft);
        let all_logits = forward(main_model, &verify_seq, &mut main_cache, main_wte_t, true);

        let mut committed: Vec<usize> = Vec::new();
        let mut n_accept = 0;
        for j in 0..k {
            let target = sample(&all_logits.data[j * n_vocab..(j + 1) * n_vocab], temperature, top_p, &mut rng);
            if draft[j] == target {
                committed.push(draft[j]);
                n_accept += 1;
            } else {
                // first mismatch: take the target's correction and stop
                committed.push(target);
                break;
            }
        }
        // All k drafts accepted -> the final row gives a free bonus token.
        if n_accept == k {
            let bonus = sample(&all_logits.data[k * n_vocab..(k + 1) * n_vocab], temperature, top_p, &mut rng);
            committed.push(bonus);
        }

        total_draft += k;
        total_accepted += n_accept;

        for &tok in &committed {
            print!("{}", tokenizer.decode(&tok));
            std::io::stdout().flush().unwrap();
            token_ids.push(tok);
        }

        // Drop the speculative tail and keep both caches at len-1; the last committed
        // token is re-fed at the top of the next iteration.
        truncate_cache(&mut small_cache, token_ids.len() - 1);
        truncate_cache(&mut main_cache, token_ids.len() - 1);
    }

    let elapsed = start.elapsed().as_secs_f32();
    let generated = token_ids.len() - initial_len;
    println!();
    println!("tokens: {}", generated);
    println!("time: {:.2}s", elapsed);
    println!("tokens/sec: {:.2}", generated as f32 / elapsed);
    println!("acceptance rate: {:.1}%", total_accepted as f32 / total_draft as f32 * 100.0);
}


fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0
}

fn sample(logits: &[f32], temperature: f32, top_p: f32, rng: &mut impl Rng) -> usize {
    let n = logits.len();
    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
    let probs = Tensor::new(scaled, vec![n]).softmax();

    let mut indexed: Vec<(usize, f32)> =
        probs.data.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0.0f32; n];
    let mut cumulative = 0.0f32;
    for &(idx, prob) in &indexed {
        filtered[idx] = prob;
        cumulative += prob;
        if cumulative > top_p {
            break;
        }
    }

    let sum: f32 = filtered.iter().sum();
    let roll: f32 = rng.r#gen();
    let mut cumulative = 0.0f32;
    for (i, prob) in filtered.iter().enumerate() {
        cumulative += prob / sum;
        if cumulative > roll {
            return i;
        }
    }
    argmax(logits)
}


fn draft_tokens(model: &Model, cache: &mut Option<KVCache>, last_token: usize, k: usize, wte_t: &Tensor, temperature: f32, top_p: f32, rng: &mut impl Rng) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(k);
    let mut current = last_token;
    for _ in 0..k {
        let logits = forward(model, &[current], cache, wte_t, false);
        let next = sample(&logits.data, temperature, top_p, rng);
        tokens.push(next);
        current = next;
    }
    // Feed the final drafted token too so the draft cache covers [last_token, draft[0..k]],
    // matching what the main model caches during verification (its logits are unused).
    let _ = forward(model, &[current], cache, wte_t, false);
    tokens
}


fn truncate_cache(cache: &mut Option<KVCache>, target_len: usize) {
    if let Some(c) = cache {
        for layer in 0..c.k.len() {
            for head in 0..c.k[layer].len() {
                let head_dim = c.k[layer][head].shape[1];
                c.k[layer][head].data.truncate(target_len * head_dim);
                c.k[layer][head].shape[0] = target_len;
                c.v[layer][head].data.truncate(target_len * head_dim);
                c.v[layer][head].shape[0] = target_len;
            }
        }
    }
}
