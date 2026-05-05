use crate::errors::ApiError;

pub fn validate_model(request_model: Option<&str>, loaded_model: &str) -> Result<(), ApiError> {
    let Some(model) = request_model else {
        return Ok(());
    };
    let model = model.trim();
    if model.is_empty()
        || model == loaded_model
        || matches!(model, "default" | "local" | "supersonic" | "any")
    {
        return Ok(());
    }
    Err(ApiError::bad_request(format!(
        "requested model '{model}' does not match loaded model '{loaded_model}'"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_loaded_model_and_local_aliases() {
        validate_model(Some("qwen3.5-0.8b"), "qwen3.5-0.8b").unwrap();
        validate_model(Some("local"), "qwen3.5-0.8b").unwrap();
        validate_model(Some("any"), "qwen3.5-0.8b").unwrap();
        validate_model(None, "qwen3.5-0.8b").unwrap();
    }

    #[test]
    fn rejects_unrelated_model() {
        let err = validate_model(Some("gpt-4.1"), "qwen3.5-0.8b").expect_err("model mismatch");
        assert!(err.body.message.contains("does not match"));
    }
}
