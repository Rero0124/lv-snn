use crate::network::Network;
use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

pub struct AppState {
    pub net: Mutex<Network>,
    // lock 없이 읽을 수 있는 상태 카운터
    pub stat_neurons: AtomicUsize,
    pub stat_synapses: AtomicUsize,
    pub stat_cached: AtomicUsize,
    pub stat_fire_count: AtomicU64,
    pub stat_patterns: AtomicUsize,
    pub stat_token_vocab: AtomicUsize,
}

impl AppState {
    pub fn new(net: Network) -> Self {
        let neurons = net.neuron_count();
        let synapses = net.synapse_count();
        let cached = net.cached_count();
        let fire_count = net.fire_count();
        let patterns = net.pattern_count();
        let token_vocab = net.token_vocab_count();
        Self {
            net: Mutex::new(net),
            stat_neurons: AtomicUsize::new(neurons),
            stat_synapses: AtomicUsize::new(synapses),
            stat_cached: AtomicUsize::new(cached),
            stat_fire_count: AtomicU64::new(fire_count),
            stat_patterns: AtomicUsize::new(patterns),
            stat_token_vocab: AtomicUsize::new(token_vocab),
        }
    }

    fn update_stats(&self, net: &Network) {
        self.stat_neurons.store(net.neuron_count(), Ordering::Relaxed);
        self.stat_synapses.store(net.synapse_count(), Ordering::Relaxed);
        self.stat_cached.store(net.cached_count(), Ordering::Relaxed);
        self.stat_fire_count.store(net.fire_count(), Ordering::Relaxed);
        self.stat_patterns.store(net.pattern_count(), Ordering::Relaxed);
        self.stat_token_vocab.store(net.token_vocab_count(), Ordering::Relaxed);
    }
}

#[derive(Deserialize)]
pub struct FireReq {
    pub text: String,
}

#[derive(Serialize)]
pub struct FireResp {
    pub fire_id: u64,
    pub output: String,
    pub path_length: usize,
}

#[derive(Deserialize)]
pub struct TeachReq {
    pub input: String,
    pub target: String,
}

#[derive(Serialize)]
pub struct TeachResp {
    pub output: String,
    pub semantic_score: f64,
    pub efficiency_score: f64,
    pub combined_score: f64,
    pub path_length: usize,
}

#[derive(Deserialize)]
pub struct FeedbackReq {
    pub fire_id: u64,
    pub positive: bool,
    #[serde(default = "default_strength")]
    pub strength: f64,
}

fn default_strength() -> f64 { 1.0 }

#[derive(Deserialize)]
pub struct FeedbackPartialReq {
    pub fire_id: u64,
    pub token_scores: Vec<(String, f64)>,  // [("단어", 점수), ...] 양수=강화, 음수=약화
}

async fn fire(state: web::Data<AppState>, req: web::Json<FireReq>) -> HttpResponse {
    let mut net = state.net.lock().unwrap();
    let fire_id = net.fire(&req.text);
    let output = net.get_last_output();
    let path_length = net.get_last_path_length();
    state.update_stats(&net);
    drop(net);

    HttpResponse::Ok().json(FireResp {
        fire_id,
        output,
        path_length,
    })
}

async fn teach(state: web::Data<AppState>, req: web::Json<TeachReq>) -> HttpResponse {
    let mut net = state.net.lock().unwrap();
    let result = net.teach_api(&req.input, &req.target);
    state.update_stats(&net);
    drop(net);

    HttpResponse::Ok().json(result)
}

async fn feedback(state: web::Data<AppState>, req: web::Json<FeedbackReq>) -> HttpResponse {
    let mut net = state.net.lock().unwrap();
    net.feedback(req.fire_id, req.positive, req.strength);
    drop(net);
    HttpResponse::Ok().json(serde_json::json!({"ok": true}))
}

async fn feedback_partial(state: web::Data<AppState>, req: web::Json<FeedbackPartialReq>) -> HttpResponse {
    let mut net = state.net.lock().unwrap();
    net.feedback_partial(req.fire_id, &req.token_scores);
    state.update_stats(&net);
    drop(net);
    HttpResponse::Ok().json(serde_json::json!({"ok": true}))
}

async fn status(state: web::Data<AppState>) -> HttpResponse {
    // lock 없이 atomic 읽기 — fire 중에도 즉시 응답
    HttpResponse::Ok().json(serde_json::json!({
        "neurons": state.stat_neurons.load(Ordering::Relaxed),
        "synapses": state.stat_synapses.load(Ordering::Relaxed),
        "cached": state.stat_cached.load(Ordering::Relaxed),
        "fire_count": state.stat_fire_count.load(Ordering::Relaxed),
        "patterns": state.stat_patterns.load(Ordering::Relaxed),
        "token_vocab": state.stat_token_vocab.load(Ordering::Relaxed),
    }))
}

async fn save(state: web::Data<AppState>) -> HttpResponse {
    let net = state.net.lock().unwrap();
    net.save_state();
    HttpResponse::Ok().json(serde_json::json!({"ok": true, "message": "저장 완료"}))
}

pub async fn run_server(state: web::Data<AppState>, port: u16) -> std::io::Result<()> {
    eprintln!("  [서버] http://127.0.0.1:{port} 에서 시작");
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(1048576))
            .route("/fire", web::post().to(fire))
            .route("/teach", web::post().to(teach))
            .route("/feedback", web::post().to(feedback))
            .route("/feedback_partial", web::post().to(feedback_partial))
            .route("/status", web::get().to(status))
            .route("/save", web::post().to(save))
    })
    .keep_alive(Duration::from_secs(300))
    .client_request_timeout(Duration::from_secs(300))
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
