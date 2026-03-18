use crate::network::Network;
use actix_web::{web, App, HttpServer, HttpResponse};
use crossbeam::channel;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::oneshot;

// ── 요청/응답 타입 ──

enum NetRequest {
    Fire { text: String, reply: oneshot::Sender<FireResp> },
    FireSequential { text: String, reply: oneshot::Sender<FireResp> },
    FireDebug { text: String, reply: oneshot::Sender<serde_json::Value> },
    Teach { input: String, target: String, reply: oneshot::Sender<serde_json::Value> },
    Feedback { fire_id: u64, positive: bool, strength: f64, reply: oneshot::Sender<()> },
    FeedbackPartial { fire_id: u64, token_scores: Vec<(String, f64)>, reply: oneshot::Sender<()> },
    Save { reply: oneshot::Sender<()> },
}

pub struct AppState {
    tx: channel::Sender<NetRequest>,
    pub stat_neurons: AtomicUsize,
    pub stat_synapses: AtomicUsize,
    pub stat_cached: AtomicUsize,
    pub stat_fire_count: AtomicU64,
    pub stat_patterns: AtomicUsize,
    pub stat_token_vocab: AtomicUsize,
}

fn update_stats(state: &AppState, net: &Network) {
    state.stat_neurons.store(net.neuron_count(), Ordering::Relaxed);
    state.stat_synapses.store(net.synapse_count(), Ordering::Relaxed);
    state.stat_cached.store(net.cached_count(), Ordering::Relaxed);
    state.stat_fire_count.store(net.fire_count(), Ordering::Relaxed);
    state.stat_patterns.store(net.pattern_count(), Ordering::Relaxed);
    state.stat_token_vocab.store(net.token_vocab_count(), Ordering::Relaxed);
}

#[derive(Deserialize)]
pub struct FireReq { pub text: String }

#[derive(Serialize, Clone)]
pub struct FireResp {
    pub fire_id: u64,
    pub output: String,
    pub path_length: usize,
    pub output_tokens: Vec<OutputTokenInfo>,
}

#[derive(Serialize, Clone)]
pub struct OutputTokenInfo {
    pub token: String,
    pub length: usize,
    pub weight: f64,
    pub synapse_id: String,
}

#[derive(Deserialize)]
pub struct TeachReq { pub input: String, pub target: String }

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
    pub token_scores: Vec<(String, f64)>,
}

// ── 핸들러: crossbeam 큐에 넣고 oneshot으로 응답 대기 ──

async fn fire(state: web::Data<AppState>, req: web::Json<FireReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Fire { text: req.text.clone(), reply: reply_tx });
    match reply_rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn fire_sequential(state: web::Data<AppState>, req: web::Json<FireReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::FireSequential { text: req.text.clone(), reply: reply_tx });
    match reply_rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn fire_debug(state: web::Data<AppState>, req: web::Json<FireReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::FireDebug { text: req.text.clone(), reply: reply_tx });
    match reply_rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn teach(state: web::Data<AppState>, req: web::Json<TeachReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Teach { input: req.input.clone(), target: req.target.clone(), reply: reply_tx });
    match reply_rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn feedback(state: web::Data<AppState>, req: web::Json<FeedbackReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Feedback { fire_id: req.fire_id, positive: req.positive, strength: req.strength, reply: reply_tx });
    match reply_rx.await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn feedback_partial(state: web::Data<AppState>, req: web::Json<FeedbackPartialReq>) -> HttpResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::FeedbackPartial { fire_id: req.fire_id, token_scores: req.token_scores.clone(), reply: reply_tx });
    match reply_rx.await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn status(state: web::Data<AppState>) -> HttpResponse {
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
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Save { reply: reply_tx });
    match reply_rx.await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({"ok": true, "message": "저장 완료"})),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

pub async fn run_server(net: Network, port: u16) -> std::io::Result<()> {
    let (tx, rx) = channel::bounded::<NetRequest>(256);

    let state = web::Data::new(AppState {
        tx,
        stat_neurons: AtomicUsize::new(net.neuron_count()),
        stat_synapses: AtomicUsize::new(net.synapse_count()),
        stat_cached: AtomicUsize::new(net.cached_count()),
        stat_fire_count: AtomicU64::new(net.fire_count()),
        stat_patterns: AtomicUsize::new(net.pattern_count()),
        stat_token_vocab: AtomicUsize::new(net.token_vocab_count()),
    });
    let state_for_worker = state.clone();

    // 전용 OS 스레드: Network 단독 소유, crossbeam 큐에서 요청 순차 처리
    // 5분마다 자동 저장 (요청 대기 중 타임아웃 활용)
    std::thread::spawn(move || {
        let mut net = net;
        let mut last_save = std::time::Instant::now();
        let auto_save_interval = Duration::from_secs(300); // 5분
        let mut request_count: u64 = 0;

        loop {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(req) => {
                    match req {
                        NetRequest::Fire { text, reply } => {
                            if rx.is_empty() {
                                net.run_pending_post_process();
                            }
                            let fire_id = net.fire_parallel(&text);
                            let output = net.get_last_output();
                            let path_length = net.get_last_path_length();
                            let tokens = net.get_last_output_tokens();
                            let output_tokens = tokens.into_iter().map(|(token, length, weight, synapse_id)| {
                                OutputTokenInfo { token, length, weight, synapse_id }
                            }).collect();
                            let _ = reply.send(FireResp { fire_id, output, path_length, output_tokens });
                            update_stats(&state_for_worker, &net);
                        }
                        NetRequest::FireSequential { text, reply } => {
                            if rx.is_empty() {
                                net.run_pending_post_process();
                            }
                            let fire_id = net.fire(&text);
                            let output = net.get_last_output();
                            let path_length = net.get_last_path_length();
                            let tokens = net.get_last_output_tokens();
                            let output_tokens = tokens.into_iter().map(|(token, length, weight, synapse_id)| {
                                OutputTokenInfo { token, length, weight, synapse_id }
                            }).collect();
                            let _ = reply.send(FireResp { fire_id, output, path_length, output_tokens });
                            update_stats(&state_for_worker, &net);
                        }
                        NetRequest::FireDebug { text, reply } => {
                            let result = net.fire_debug(&text);
                            let _ = reply.send(serde_json::to_value(result).unwrap_or_default());
                        }
                        NetRequest::Teach { input, target, reply } => {
                            let result = net.teach_api(&input, &target);
                            update_stats(&state_for_worker, &net);
                            let _ = reply.send(serde_json::to_value(result).unwrap_or_default());
                        }
                        NetRequest::Feedback { fire_id, positive, strength, reply } => {
                            net.feedback(fire_id, positive, strength);
                            let _ = reply.send(());
                        }
                        NetRequest::FeedbackPartial { fire_id, token_scores, reply } => {
                            net.feedback_partial(fire_id, &token_scores);
                            update_stats(&state_for_worker, &net);
                            let _ = reply.send(());
                        }
                        NetRequest::Save { reply } => {
                            net.save_state();
                            last_save = std::time::Instant::now();
                            let _ = reply.send(());
                        }
                    }
                    request_count += 1;
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    // 타임아웃: 밀린 후처리 실행 + 자동 저장 체크
                    net.run_pending_post_process();
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    eprintln!("  [워커] 채널 끊김, 저장 후 종료...");
                    net.save_state();
                    break;
                }
            }

            // 자동 저장: 5분 경과 && 요청이 있었을 때
            if last_save.elapsed() >= auto_save_interval && request_count > 0 {
                eprintln!("  [자동저장] {}건 처리 후 자동 저장...", request_count);
                net.save_state();
                last_save = std::time::Instant::now();
                request_count = 0;
            }
        }
    });

    eprintln!("  [서버] http://127.0.0.1:{port} 에서 시작 (큐 기반)");
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(1048576))
            .route("/fire", web::post().to(fire))
            .route("/fire_sequential", web::post().to(fire_sequential))
            .route("/fire_debug", web::post().to(fire_debug))
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
