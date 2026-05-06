use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Profile {
    pub schema_version: u32,
}

impl Profile {
    pub fn empty() -> Self {
        Self { schema_version: 1 }
    }
}
