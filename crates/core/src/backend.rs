use gpu_hal::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Explicit(Backend),
}

impl BackendChoice {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "hip" => Some(Self::Explicit(Backend::Hip)),
            _ => None,
        }
    }
}

pub const BACKEND_CHOICES: &str = "auto | hip";

pub fn compiled_backends_display() -> String {
    Backend::Hip.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_choice_parse_accepts_supported_values() {
        assert_eq!(BackendChoice::parse("auto"), Some(BackendChoice::Auto));
        assert_eq!(BackendChoice::parse(""), Some(BackendChoice::Auto));
        assert_eq!(
            BackendChoice::parse("hip"),
            Some(BackendChoice::Explicit(Backend::Hip))
        );
        assert_eq!(BackendChoice::parse("vulkan"), None);
    }
}
