//! Local tokenizer utility endpoints. These are not OpenAI core endpoints,
//! but local harnesses commonly use them for prompt budgeting and debugging.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::compat::validate_model;
use crate::errors::ApiError;
use crate::schemas::{DetokenizeRequest, DetokenizeResponse, TokenizeRequest, TokenizeResponse};
use crate::state::ServerState;

pub async fn tokenize(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    validate_model(req.model.as_deref(), &state.model_id)?;
    let add_special_tokens = req.add_special_tokens.unwrap_or(true);
    let encoding = state
        .tokenizer
        .encode(req.input.as_str(), add_special_tokens)
        .map_err(|e| ApiError::bad_request(format!("tokenize failed: {e}")))?;
    let tokens = encoding.get_ids().to_vec();
    Ok(Json(TokenizeResponse {
        object: "tokenization",
        model: state.model_id.clone(),
        tokens,
    }))
}

pub async fn detokenize(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    validate_model(req.model.as_deref(), &state.model_id)?;
    let skip_special_tokens = req.skip_special_tokens.unwrap_or(true);
    let text = state
        .tokenizer
        .decode(&req.tokens, skip_special_tokens)
        .map_err(|e| ApiError::bad_request(format!("detokenize failed: {e}")))?;
    Ok(Json(DetokenizeResponse {
        object: "detokenization",
        model: state.model_id.clone(),
        text,
    }))
}
