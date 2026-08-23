#![recursion_limit = "512"]

//! Public runner library surface for the Qwen3.8 GQH contract.
//!
//! The production binary owns its private decode modules. Integration tests
//! use only the parser and host-side artifact validator exported here.

pub mod cli;
pub use cli::{parse_cli_from, Cli};

pub mod model_files;
pub use model_files::validate_input_contract;
