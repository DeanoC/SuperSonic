pub mod registry;
pub mod run;
pub mod tasks;

pub use registry::{
    all_tag_metadata, all_task_metadata, all_tasks, describe_task_selector, find_task,
    task_metadata, TagMetadata, TaskDef, TaskMetadata,
};
pub use run::{
    diff_exit_code, diff_runs, diff_runs_with_min_speedup, render_diff_markdown, render_markdown,
    run_tasks, CaseResult, DiffReport, KernelLabConfig, RunSummary, TaskResult,
};
