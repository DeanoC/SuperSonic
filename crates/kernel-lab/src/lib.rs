pub mod registry;
pub mod run;
pub mod tasks;

pub use registry::{all_tasks, find_task, TaskDef};
pub use run::{
    diff_exit_code, diff_runs, render_diff_markdown, render_markdown, run_tasks, CaseResult,
    DiffReport, KernelLabConfig, RunSummary, TaskResult,
};
