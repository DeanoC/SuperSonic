//! OpenAI-shaped error envelopes. Every non-streaming error response looks
//! like `{ "error": { "message", "type", "param", "code" } }` with an
//! appropriate HTTP status.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorBody,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ApiErrorBody,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ApiErrorBody {
                message: msg.into(),
                type_: "invalid_request_error".into(),
                param: None,
                code: None,
            },
        }
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: ApiErrorBody {
                message: msg.into(),
                type_: "authentication_error".into(),
                param: None,
                code: None,
            },
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ApiErrorBody {
                message: msg.into(),
                type_: "internal_error".into(),
                param: None,
                code: None,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let env = ApiErrorEnvelope { error: self.body };
        (self.status, Json(env)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        Self::internal(err.to_string())
    }
}
