//! HTTP API and the embedded Svelte UI.
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::{Path as UrlPath, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use rust_embed::RustEmbed;
use serde_json::json;

use crate::jobs::{Queue, RunRequest};
use crate::store::{Store, scan_models};

#[derive(RustEmbed)]
#[folder = "web/dist"]
struct Ui;

#[derive(Clone)]
pub struct App {
    pub store: Store,
    pub queue: Queue,
    pub model_dirs: Arc<Vec<PathBuf>>,
}

fn err(code: StatusCode, msg: impl ToString) -> Response {
    (code, Json(json!({"error": msg.to_string()}))).into_response()
}

async fn health(State(a): State<App>) -> Response {
    let runs = a.queue.runs.lock().unwrap();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "traces": a.store.list().len(),
        "runs_active": runs.iter().filter(|r| r.state == "running" || r.state == "queued").count(),
        "model_dirs": *a.model_dirs,
    })).into_response()
}

async fn traces(State(a): State<App>) -> Response {
    Json(a.store.list()).into_response()
}

async fn trace(State(a): State<App>, UrlPath(name): UrlPath<String>) -> Response {
    match a.store.viz_path(&name) {
        Some(p) => match tokio::fs::read(p).await {
            Ok(b) => ([(header::CONTENT_TYPE, "application/json")], b).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        },
        None => err(StatusCode::NOT_FOUND, format!("no trace {name:?}")),
    }
}

async fn models(State(a): State<App>) -> Response {
    let dirs = a.model_dirs.clone();
    Json(tokio::task::spawn_blocking(move || scan_models(&dirs)).await.unwrap_or_default()).into_response()
}

async fn runs(State(a): State<App>) -> Response {
    Json(a.queue.runs.lock().unwrap().clone()).into_response()
}

async fn submit(State(a): State<App>, Json(req): Json<RunRequest>) -> Response {
    if let Err(e) = a.store.dir(&req.name) {
        return err(StatusCode::BAD_REQUEST, e);
    }
    if req.prompt.trim().is_empty() || req.n > 4096 {
        return err(StatusCode::BAD_REQUEST, "prompt must be non-empty and n <= 4096");
    }
    // Only models found under the configured directories may be run.
    let dirs = a.model_dirs.clone();
    let wanted = req.model.clone();
    let allowed = tokio::task::spawn_blocking(move || {
        scan_models(&dirs).iter().any(|m| m["path"].as_str() == Some(&wanted.to_string_lossy()))
    }).await.unwrap_or(false);
    if !allowed {
        return err(StatusCode::BAD_REQUEST, "model is not in the server's model directories (see GET /api/models)");
    }
    Json(json!({"id": a.queue.submit(req)})).into_response()
}

async fn static_file(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let file = Ui::get(path).or_else(|| (!path.starts_with("api/")).then(|| Ui::get("index.html")).flatten());
    match file {
        Some(f) => {
            let mime = mime_guess::from_path(if Ui::get(path).is_some() { path } else { "index.html" }).first_or_octet_stream();
            Response::builder().header(header::CONTENT_TYPE, mime.as_ref()).body(Body::from(f.data.into_owned())).unwrap()
        }
        None => err(StatusCode::NOT_FOUND, "not found"),
    }
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/traces", get(traces))
        .route("/api/traces/{name}", get(trace))
        .route("/api/models", get(models))
        .route("/api/runs", get(runs).post(submit))
        .fallback(static_file)
        .with_state(app)
}
