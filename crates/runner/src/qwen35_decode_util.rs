pub(crate) fn token_history(prompt_ids: &[u32], generated_ids: &[u32]) -> Vec<u32> {
    prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect()
}

pub(crate) fn token_history_with_next(
    prompt_ids: &[u32],
    generated_ids: &[u32],
    next_token: u32,
) -> Vec<u32> {
    prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .chain(std::iter::once(next_token))
        .collect()
}
