use std::time::{SystemTime, UNIX_EPOCH};

pub fn chat_completion_id() -> String {
    format!("chatcmpl-{:x}{:04x}", epoch_secs(), rand::random::<u16>())
}

pub fn completion_id() -> String {
    format!("cmpl-{:x}{:04x}", epoch_secs(), rand::random::<u16>())
}

pub fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
