use crate::tasks;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct TaskDef {
    pub id: &'static str,
    pub family: &'static str,
    pub description: &'static str,
    pub tags: &'static [&'static str],
    pub backend_support: &'static [&'static str],
    pub correctness: &'static str,
    pub required: bool,
    pub run: fn(&crate::run::KernelLabConfig) -> anyhow::Result<crate::run::TaskResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskMetadata {
    pub id: String,
    pub family: String,
    pub description: String,
    pub tags: Vec<String>,
    pub backend_support: Vec<String>,
    pub correctness: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TagMetadata {
    pub tag: String,
    pub task_count: usize,
    pub required_task_count: usize,
}

const HIP_BACKEND: &[&str] = &["hip"];

const TASKS: &[TaskDef] = &[
    TaskDef {
        id: "qwen35.full_attention_prefill",
        family: "qwen3.5",
        description: "Qwen3.5 full-attention prefill microbenchmark over short and longer KV contexts.",
        tags: &["qwen35", "attention", "prefill", "required"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP output against the CPU attention reference with max_abs/max_rel/min_cos tolerances.",
        required: true,
        run: tasks::qwen35_full_attention_prefill,
    },
    TaskDef {
        id: "qwen35.int4_matvec",
        family: "qwen3.5",
        description: "Qwen3.5 INT4 matvec baseline without AWQ sidecar correction.",
        tags: &["qwen35", "int4", "awq", "quant"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP BF16 matvec output against the CPU INT4 dequant reference.",
        required: false,
        run: tasks::qwen35_int4_matvec,
    },
    TaskDef {
        id: "qwen35.int4_awq_dense_matvec",
        family: "qwen3.5",
        description: "Qwen3.5 INT4 matvec with dense AWQ inverse-scale sidecar.",
        tags: &["qwen35", "int4", "awq", "quant"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP BF16 matvec output against the CPU INT4 plus dense AWQ reference.",
        required: false,
        run: tasks::qwen35_int4_awq_dense_matvec,
    },
    TaskDef {
        id: "qwen35.int4_awq_sparse_outlier_matvec",
        family: "qwen3.5",
        description: "Qwen3.5 INT4 matvec with sparse outlier sidecar accumulation.",
        tags: &["qwen35", "int4", "awq", "quant"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP BF16 matvec output against the CPU INT4 plus sparse outlier reference.",
        required: false,
        run: tasks::qwen35_int4_awq_sparse_outlier_matvec,
    },
    TaskDef {
        id: "qwen36.batched_prefill_attn_full",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE batched full-attention prefill over representative decode-prefill shapes.",
        tags: &["qwen36", "attention", "prefill", "required"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP output against the CPU attention reference with max_abs/max_rel/min_cos tolerances.",
        required: true,
        run: tasks::qwen36_batched_prefill_attn_full,
    },
    TaskDef {
        id: "qwen36.router_permute",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE router top-k permutation and token routing layout benchmark.",
        tags: &["qwen36", "moe", "router", "required"],
        backend_support: HIP_BACKEND,
        correctness: "Checks exact router indices, expert counts, token offsets, and routing weights against the CPU reference.",
        required: true,
        run: tasks::qwen36_router_permute,
    },
    TaskDef {
        id: "qwen36.grouped_expert_int4",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE grouped INT4 expert projection benchmark.",
        tags: &["qwen36", "moe", "int4", "required"],
        backend_support: HIP_BACKEND,
        correctness: "Compares grouped HIP expert output against the CPU INT4 grouped-expert reference.",
        required: true,
        run: tasks::qwen36_grouped_expert_int4,
    },
    TaskDef {
        id: "qwen36.unpermute_combine",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE unpermute and top-k weighted combine benchmark.",
        tags: &["qwen36", "moe", "combine", "required"],
        backend_support: HIP_BACKEND,
        correctness: "Compares combined HIP token output against the CPU unpermute/combine reference.",
        required: true,
        run: tasks::qwen36_unpermute_combine,
    },
    TaskDef {
        id: "qwen36.batched_prefill_attn_full.stress",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE batched full-attention prefill stress cases with larger contexts.",
        tags: &["qwen36", "attention-stress", "stress"],
        backend_support: HIP_BACKEND,
        correctness: "Compares HIP output against the CPU attention reference with max_abs/max_rel/min_cos tolerances.",
        required: false,
        run: tasks::qwen36_batched_prefill_attn_full_stress,
    },
    TaskDef {
        id: "qwen36.router_permute.stress",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE router permutation stress cases at larger token counts.",
        tags: &["qwen36", "router-stress", "stress"],
        backend_support: HIP_BACKEND,
        correctness: "Checks exact router indices, expert counts, token offsets, and routing weights against the CPU reference.",
        required: false,
        run: tasks::qwen36_router_permute_stress,
    },
    TaskDef {
        id: "qwen36.grouped_expert_int4.stress",
        family: "qwen3.6-moe",
        description: "Qwen3.6 MoE grouped INT4 expert stress cases at larger token counts.",
        tags: &["qwen36", "int4-stress", "stress"],
        backend_support: HIP_BACKEND,
        correctness: "Compares grouped HIP expert output against the CPU INT4 grouped-expert reference.",
        required: false,
        run: tasks::qwen36_grouped_expert_int4_stress,
    },
];

pub fn all_tasks() -> &'static [TaskDef] {
    TASKS
}

pub fn find_task(id: &str) -> Option<&'static TaskDef> {
    TASKS.iter().find(|task| task.id == id)
}

pub fn task_metadata(task: &TaskDef) -> TaskMetadata {
    TaskMetadata {
        id: task.id.to_string(),
        family: task.family.to_string(),
        description: task.description.to_string(),
        tags: task.tags.iter().map(|tag| tag.to_string()).collect(),
        backend_support: task
            .backend_support
            .iter()
            .map(|backend| backend.to_string())
            .collect(),
        correctness: task.correctness.to_string(),
        required: task.required,
    }
}

pub fn all_task_metadata() -> Vec<TaskMetadata> {
    TASKS.iter().map(task_metadata).collect()
}

pub fn describe_task_selector(selector: &str) -> anyhow::Result<Vec<TaskMetadata>> {
    if let Some(tag) = selector.strip_prefix("tag:") {
        let matches: Vec<_> = TASKS
            .iter()
            .filter(|task| task.tags.iter().any(|task_tag| *task_tag == tag))
            .map(task_metadata)
            .collect();
        if matches.is_empty() {
            anyhow::bail!("unknown task tag: {tag}");
        }
        return Ok(matches);
    }

    let task = find_task(selector).ok_or_else(|| anyhow::anyhow!("unknown task: {selector}"))?;
    Ok(vec![task_metadata(task)])
}

pub fn all_tag_metadata() -> Vec<TagMetadata> {
    let mut counts = std::collections::BTreeMap::<String, (usize, usize)>::new();
    for task in TASKS {
        for tag in task.tags {
            let entry = counts.entry((*tag).to_string()).or_default();
            entry.0 += 1;
            if task.required {
                entry.1 += 1;
            }
        }
    }
    counts
        .into_iter()
        .map(|(tag, (task_count, required_task_count))| TagMetadata {
            tag,
            task_count,
            required_task_count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn task_metadata_is_complete_and_well_formed() {
        let mut ids = BTreeSet::new();
        for task in all_tasks() {
            assert!(ids.insert(task.id), "duplicate task id {}", task.id);
            assert!(
                !task.family.trim().is_empty(),
                "empty family for {}",
                task.id
            );
            assert!(
                !task.description.trim().is_empty(),
                "empty description for {}",
                task.id
            );
            assert!(
                !task.correctness.trim().is_empty(),
                "empty correctness for {}",
                task.id
            );
            assert!(
                !task.backend_support.is_empty(),
                "empty backend support for {}",
                task.id
            );
            assert!(!task.tags.is_empty(), "empty tags for {}", task.id);
            let mut tags = BTreeSet::new();
            for tag in task.tags {
                assert!(!tag.trim().is_empty(), "empty tag for {}", task.id);
                assert!(
                    tag.chars().all(|ch| ch.is_ascii_lowercase()
                        || ch.is_ascii_digit()
                        || ch == '-'
                        || ch == '_'),
                    "invalid tag {tag} for {}",
                    task.id
                );
                assert!(tags.insert(*tag), "duplicate tag {tag} for {}", task.id);
            }
        }
    }

    #[test]
    fn task_metadata_snapshots_are_derived_from_registry() {
        let meta = all_task_metadata();
        assert_eq!(meta.len(), all_tasks().len());
        assert_eq!(meta[0].id, all_tasks()[0].id);
        assert_eq!(meta[0].description, all_tasks()[0].description);
        assert_eq!(meta[0].backend_support, vec!["hip"]);
    }

    #[test]
    fn tag_metadata_counts_tasks() {
        let tags = all_tag_metadata();
        let required = tags.iter().find(|tag| tag.tag == "required").unwrap();
        assert_eq!(required.task_count, 5);
        assert_eq!(required.required_task_count, 5);

        let stress = tags.iter().find(|tag| tag.tag == "stress").unwrap();
        assert_eq!(stress.task_count, 3);
        assert_eq!(stress.required_task_count, 0);
    }

    #[test]
    fn describe_selector_accepts_task_id_and_tag() {
        let task = describe_task_selector("qwen35.full_attention_prefill").unwrap();
        assert_eq!(task.len(), 1);
        assert_eq!(task[0].id, "qwen35.full_attention_prefill");

        let tagged = describe_task_selector("tag:stress").unwrap();
        assert_eq!(tagged.len(), 3);
        assert!(tagged.iter().all(|task| !task.required));
    }
}
