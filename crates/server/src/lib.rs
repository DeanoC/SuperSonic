//! SuperSonic HTTP server — an OpenAI-compatible inference endpoint
//! wrapping the `runner` crate's decode engines.

pub mod chat_template;
pub mod errors;
pub mod generate;
pub mod routes;
pub mod sampling;
pub mod schemas;
pub mod session;
pub mod state;

use std::sync::Arc;

/// Run the HTTP server on an already-bound `TcpListener`. Callers own the
/// listener so tests can bind to port 0 and discover the ephemeral address
/// before handing it over.
pub async fn serve(
    state: Arc<state::ServerState>,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    let app = routes::router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
