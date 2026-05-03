pub struct Qwen36DecodeLoopState {
    pub generated_ids: Vec<u32>,
    pub last_logits_bytes: Vec<u8>,
    pub current_token: u32,
    pub position: i32,
    pub total_steps: usize,
    max_new: usize,
}

impl Qwen36DecodeLoopState {
    pub fn new(prompt_ids: &[u32], max_new: usize) -> Self {
        Self {
            generated_ids: Vec::with_capacity(max_new),
            last_logits_bytes: Vec::new(),
            current_token: prompt_ids[0],
            position: 0,
            total_steps: prompt_ids.len() + max_new - 1,
            max_new,
        }
    }

    pub fn reached_max_new(&self) -> bool {
        self.generated_ids.len() >= self.max_new
    }

    pub fn remaining_generation_slots(&self) -> usize {
        self.max_new.saturating_sub(self.generated_ids.len())
    }

    pub fn record_last_logits(&mut self, logits: &[u8]) {
        self.last_logits_bytes.clear();
        self.last_logits_bytes.extend_from_slice(logits);
    }
}
