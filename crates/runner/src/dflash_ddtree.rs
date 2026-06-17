use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::tensor_bytes::bf16_bytes_to_f32;

#[derive(Debug, Clone, PartialEq)]
pub struct DDTree {
    pub token_ids: Vec<u32>,
    pub depths: Vec<usize>,
    pub parents: Vec<isize>,
    pub child_maps: Vec<HashMap<u32, usize>>,
    pub visibility: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DDTreeVerifyPlan {
    pub flat_tokens: Vec<u32>,
    pub parent_ids: Vec<i32>,
    pub depths: Vec<usize>,
    pub positions: Vec<usize>,
    pub visibility: Vec<u8>,
}

impl DDTree {
    pub fn n_nodes(&self) -> usize {
        self.token_ids.len()
    }

    pub fn width(&self) -> usize {
        1 + self.n_nodes()
    }
}

#[derive(Debug, Clone)]
struct HeapEntry {
    logw: f32,
    parent_index: usize,
    depth: usize,
    rank: usize,
}

impl Eq for HeapEntry {}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.logw == other.logw
            && self.parent_index == other.parent_index
            && self.depth == other.depth
            && self.rank == other.rank
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.logw
            .partial_cmp(&other.logw)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.rank.cmp(&self.rank))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn build_verify_plan(tree: &DDTree, root_token: u32, pos_offset: usize) -> DDTreeVerifyPlan {
    let width = tree.width();
    assert_eq!(tree.parents.len(), width);
    assert_eq!(tree.visibility.len(), width * width);

    let mut flat_tokens = Vec::with_capacity(width);
    flat_tokens.push(root_token);
    flat_tokens.extend_from_slice(&tree.token_ids);

    let mut depths = Vec::with_capacity(width);
    depths.push(0);
    depths.extend_from_slice(&tree.depths);

    let positions = depths
        .iter()
        .map(|depth| pos_offset + depth)
        .collect::<Vec<_>>();
    let parent_ids = tree
        .parents
        .iter()
        .map(|&parent| parent as i32)
        .collect::<Vec<_>>();

    DDTreeVerifyPlan {
        flat_tokens,
        parent_ids,
        depths,
        positions,
        visibility: tree.visibility.clone(),
    }
}

pub fn accepted_tokens_for_path(
    root_token: u32,
    tree: &DDTree,
    accepted_indices: &[usize],
) -> Vec<u32> {
    accepted_indices
        .iter()
        .map(|&idx| {
            if idx == 0 {
                root_token
            } else {
                tree.token_ids[idx - 1]
            }
        })
        .collect()
}

pub fn extract_draft_topk_bf16(
    logits_bf16: &[u8],
    n_positions: usize,
    vocab: usize,
    top_k: usize,
    temperature: f32,
) -> (Vec<f32>, Vec<u32>) {
    assert!(top_k > 0);
    assert!(top_k <= vocab);
    assert!(logits_bf16.len() >= n_positions * vocab * 2);

    let inv_t = 1.0f32 / temperature.max(1.0e-3);
    let mut out_log_probs = vec![0.0f32; n_positions * top_k];
    let mut out_token_ids = vec![0u32; n_positions * top_k];

    for pos in 0..n_positions {
        let row_start = pos * vocab * 2;
        let row = bf16_bytes_to_f32(&logits_bf16[row_start..row_start + vocab * 2]);
        let mut best_vals = vec![f32::NEG_INFINITY; top_k];
        let mut best_ids = vec![0u32; top_k];
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum_exp = 0.0f32;

        for (token_id, raw_logit) in row.into_iter().enumerate() {
            let logit = raw_logit * inv_t;
            if logit > running_max {
                if running_max.is_finite() {
                    running_sum_exp *= (running_max - logit).exp();
                }
                running_sum_exp += 1.0;
                running_max = logit;
            } else {
                running_sum_exp += (logit - running_max).exp();
            }

            let token_id = token_id as u32;
            if !is_better_topk_entry(logit, token_id, best_vals[top_k - 1], best_ids[top_k - 1]) {
                continue;
            }
            let mut insert_at = top_k - 1;
            while insert_at > 0
                && is_better_topk_entry(
                    logit,
                    token_id,
                    best_vals[insert_at - 1],
                    best_ids[insert_at - 1],
                )
            {
                best_vals[insert_at] = best_vals[insert_at - 1];
                best_ids[insert_at] = best_ids[insert_at - 1];
                insert_at -= 1;
            }
            best_vals[insert_at] = logit;
            best_ids[insert_at] = token_id;
        }

        let log_z = running_max + running_sum_exp.ln();
        let out_base = pos * top_k;
        for rank in 0..top_k {
            out_log_probs[out_base + rank] = best_vals[rank] - log_z;
            out_token_ids[out_base + rank] = best_ids[rank];
        }
    }

    (out_log_probs, out_token_ids)
}

fn is_better_topk_entry(lhs_logit: f32, lhs_id: u32, rhs_logit: f32, rhs_id: u32) -> bool {
    lhs_logit > rhs_logit || (lhs_logit == rhs_logit && lhs_id < rhs_id)
}

pub fn build_ddtree(
    top_log_probs: &[f32],
    top_token_ids: &[u32],
    depth_limit: usize,
    top_k: usize,
    budget: usize,
    chain_seed: bool,
) -> DDTree {
    let mut tree = DDTree {
        token_ids: Vec::with_capacity(budget),
        depths: Vec::with_capacity(budget),
        parents: vec![-1],
        child_maps: vec![HashMap::new()],
        visibility: Vec::new(),
    };
    if budget == 0 || depth_limit == 0 || top_k == 0 {
        tree.visibility = vec![1];
        return tree;
    }
    assert!(top_log_probs.len() >= depth_limit * top_k);
    assert!(top_token_ids.len() >= depth_limit * top_k);

    let mut heap = BinaryHeap::new();
    if chain_seed {
        let chain_depth = depth_limit.min(budget);
        let mut cum_logw = 0.0f32;
        let mut prev_idx = 0usize;
        for depth in 1..=chain_depth {
            let base = (depth - 1) * top_k;
            let tok_id = top_token_ids[base];
            cum_logw += top_log_probs[base];

            let cur_idx = tree.n_nodes() + 1;
            tree.token_ids.push(tok_id);
            tree.depths.push(depth);
            tree.parents.push(prev_idx as isize);
            tree.child_maps.push(HashMap::new());
            tree.child_maps[prev_idx].insert(tok_id, cur_idx);

            if top_k > 1 {
                let sibling_logw = cum_logw - top_log_probs[base] + top_log_probs[base + 1];
                heap.push(HeapEntry {
                    logw: sibling_logw,
                    parent_index: prev_idx,
                    depth,
                    rank: 1,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        heap.push(HeapEntry {
            logw: top_log_probs[0],
            parent_index: 0,
            depth: 1,
            rank: 0,
        });
    }

    while let Some(entry) = heap.pop() {
        if tree.n_nodes() >= budget {
            break;
        }
        let base = (entry.depth - 1) * top_k;
        let tok_id = top_token_ids[base + entry.rank];
        let cur_idx = tree.n_nodes() + 1;
        tree.token_ids.push(tok_id);
        tree.depths.push(entry.depth);
        tree.parents.push(entry.parent_index as isize);
        tree.child_maps.push(HashMap::new());
        tree.child_maps[entry.parent_index].insert(tok_id, cur_idx);

        if entry.rank + 1 < top_k {
            let sibling_logw = entry.logw - top_log_probs[base + entry.rank]
                + top_log_probs[base + entry.rank + 1];
            heap.push(HeapEntry {
                logw: sibling_logw,
                parent_index: entry.parent_index,
                depth: entry.depth,
                rank: entry.rank + 1,
            });
        }

        if entry.depth < depth_limit {
            let child_base = entry.depth * top_k;
            heap.push(HeapEntry {
                logw: entry.logw + top_log_probs[child_base],
                parent_index: cur_idx,
                depth: entry.depth + 1,
                rank: 0,
            });
        }
    }

    build_visibility(&mut tree);
    tree
}

pub fn follow_verified_tree(tree: &DDTree, posterior: &[u32]) -> (Vec<usize>, u32, usize) {
    assert!(posterior.len() >= tree.width());
    let mut accepted = Vec::with_capacity(tree.width());
    accepted.push(0);

    let mut current_index = 0usize;
    let mut next_token = posterior[current_index];
    while let Some(&child_index) = tree.child_maps[current_index].get(&next_token) {
        current_index = child_index;
        accepted.push(current_index);
        next_token = posterior[current_index];
    }
    (accepted, next_token, current_index)
}

fn build_visibility(tree: &mut DDTree) {
    let n = tree.width();
    tree.visibility = vec![0; n * n];
    tree.visibility[0] = 1;
    for i in 1..n {
        let parent = tree.parents[i] as usize;
        for j in 0..i {
            tree.visibility[i * n + j] = tree.visibility[parent * n + j];
        }
        tree.visibility[i * n + i] = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_bytes::f32_to_bf16_bytes;

    #[test]
    fn extract_topk_returns_sorted_log_probs() {
        let logits = f32_to_bf16_bytes(&[1.0, 3.0, 2.0, -1.0, 0.5, 0.25]);
        let (log_probs, token_ids) = extract_draft_topk_bf16(&logits, 2, 3, 2, 1.0);

        assert_eq!(token_ids, vec![1, 2, 1, 2]);
        assert!(log_probs[0] > log_probs[1]);
        assert!(log_probs[2] > log_probs[3]);
    }

    #[test]
    fn extract_topk_ties_keep_lower_token_id() {
        let logits = f32_to_bf16_bytes(&[2.0, 2.0, 1.0]);
        let (_log_probs, token_ids) = extract_draft_topk_bf16(&logits, 1, 3, 2, 1.0);

        assert_eq!(token_ids, vec![0, 1]);
    }

    #[test]
    fn chain_seed_builds_top1_spine() {
        let tree = build_ddtree(&[0.0, 0.0, 0.0], &[10, 11, 12], 3, 1, 3, true);
        assert_eq!(tree.token_ids, vec![10, 11, 12]);
        assert_eq!(tree.depths, vec![1, 2, 3]);
        assert_eq!(tree.parents, vec![-1, 0, 1, 2]);

        let (accepted, next, node) = follow_verified_tree(&tree, &[10, 11, 12, 99]);
        assert_eq!(accepted, vec![0, 1, 2, 3]);
        assert_eq!(next, 99);
        assert_eq!(node, 3);
    }

    #[test]
    fn sibling_branch_can_be_followed() {
        let top_log_probs = [-0.1, -0.2, -0.1, -0.4];
        let top_token_ids = [10, 20, 11, 21];
        let tree = build_ddtree(&top_log_probs, &top_token_ids, 2, 2, 4, true);

        assert_eq!(tree.token_ids, vec![10, 11, 20, 11]);
        assert_eq!(tree.parents, vec![-1, 0, 1, 0, 3]);

        let (accepted, next, node) = follow_verified_tree(&tree, &[20, 0, 0, 11, 99]);
        assert_eq!(accepted, vec![0, 3, 4]);
        assert_eq!(next, 99);
        assert_eq!(node, 4);
    }

    #[test]
    fn visibility_is_ancestor_only() {
        let top_log_probs = [-0.1, -0.2, -0.1, -0.4];
        let top_token_ids = [10, 20, 11, 21];
        let tree = build_ddtree(&top_log_probs, &top_token_ids, 2, 2, 4, true);
        let n = tree.width();

        let row = |i: usize| &tree.visibility[i * n..(i + 1) * n];
        assert_eq!(row(0), &[1, 0, 0, 0, 0]);
        assert_eq!(row(2), &[1, 1, 1, 0, 0]);
        assert_eq!(row(4), &[1, 0, 0, 1, 1]);
    }

    #[test]
    fn verify_plan_flattens_root_and_tree_nodes() {
        let top_log_probs = [-0.1, -0.2, -0.1, -0.4];
        let top_token_ids = [10, 20, 11, 21];
        let tree = build_ddtree(&top_log_probs, &top_token_ids, 2, 2, 4, true);
        let plan = build_verify_plan(&tree, 99, 125);

        assert_eq!(plan.flat_tokens, vec![99, 10, 11, 20, 11]);
        assert_eq!(plan.parent_ids, vec![-1, 0, 1, 0, 3]);
        assert_eq!(plan.depths, vec![0, 1, 2, 1, 2]);
        assert_eq!(plan.positions, vec![125, 126, 127, 126, 127]);
        assert_eq!(plan.visibility, tree.visibility);
    }

    #[test]
    fn accepted_path_maps_back_to_committed_tokens() {
        let top_log_probs = [-0.1, -0.2, -0.1, -0.4];
        let top_token_ids = [10, 20, 11, 21];
        let tree = build_ddtree(&top_log_probs, &top_token_ids, 2, 2, 4, true);

        assert_eq!(
            accepted_tokens_for_path(99, &tree, &[0, 3, 4]),
            vec![99, 20, 11]
        );
    }
}
