//! `GET /v1/models` — returns the single model loaded by this process.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::Json;

use crate::errors::ApiError;
use crate::schemas::{ListModelsResponse, ModelObject};
use crate::state::ServerState;

pub async fn list(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<ListModelsResponse>, ApiError> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(Json(ListModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model",
            created,
            owned_by: "supersonic",
        }],
    }))
}
