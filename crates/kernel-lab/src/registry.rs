use crate::tasks;

#[derive(Debug, Clone, Copy)]
pub struct TaskDef {
    pub id: &'static str,
    pub family: &'static str,
    pub tags: &'static [&'static str],
    pub required: bool,
    pub run: fn(&crate::run::KernelLabConfig) -> anyhow::Result<crate::run::TaskResult>,
}

const TASKS: &[TaskDef] = &[
    TaskDef {
        id: "qwen35.full_attention_prefill",
        family: "qwen3.5",
        tags: &["qwen35", "attention", "prefill", "required"],
        required: true,
        run: tasks::qwen35_full_attention_prefill,
    },
    TaskDef {
        id: "qwen35.int4_matvec",
        family: "qwen3.5",
        tags: &["qwen35", "int4", "awq", "quant"],
        required: false,
        run: tasks::qwen35_int4_matvec,
    },
    TaskDef {
        id: "qwen35.int4_awq_dense_matvec",
        family: "qwen3.5",
        tags: &["qwen35", "int4", "awq", "quant"],
        required: false,
        run: tasks::qwen35_int4_awq_dense_matvec,
    },
    TaskDef {
        id: "qwen35.int4_awq_sparse_outlier_matvec",
        family: "qwen3.5",
        tags: &["qwen35", "int4", "awq", "quant"],
        required: false,
        run: tasks::qwen35_int4_awq_sparse_outlier_matvec,
    },
    TaskDef {
        id: "qwen36.batched_prefill_attn_full",
        family: "qwen3.6-moe",
        tags: &["qwen36", "attention", "prefill", "required"],
        required: true,
        run: tasks::qwen36_batched_prefill_attn_full,
    },
    TaskDef {
        id: "qwen36.router_permute",
        family: "qwen3.6-moe",
        tags: &["qwen36", "moe", "router", "required"],
        required: true,
        run: tasks::qwen36_router_permute,
    },
    TaskDef {
        id: "qwen36.grouped_expert_int4",
        family: "qwen3.6-moe",
        tags: &["qwen36", "moe", "int4", "required"],
        required: true,
        run: tasks::qwen36_grouped_expert_int4,
    },
    TaskDef {
        id: "qwen36.unpermute_combine",
        family: "qwen3.6-moe",
        tags: &["qwen36", "moe", "combine", "required"],
        required: true,
        run: tasks::qwen36_unpermute_combine,
    },
    TaskDef {
        id: "qwen36.batched_prefill_attn_full.stress",
        family: "qwen3.6-moe",
        tags: &["qwen36", "attention-stress", "stress"],
        required: false,
        run: tasks::qwen36_batched_prefill_attn_full_stress,
    },
    TaskDef {
        id: "qwen36.router_permute.stress",
        family: "qwen3.6-moe",
        tags: &["qwen36", "router-stress", "stress"],
        required: false,
        run: tasks::qwen36_router_permute_stress,
    },
    TaskDef {
        id: "qwen36.grouped_expert_int4.stress",
        family: "qwen3.6-moe",
        tags: &["qwen36", "int4-stress", "stress"],
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
