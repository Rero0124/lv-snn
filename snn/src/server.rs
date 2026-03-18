use crate::network::Network;
use actix_web::{web, App, HttpServer, HttpResponse};
use crossbeam::channel;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

const AUTO_SAVE_INTERVAL: Duration = Duration::from_secs(300); // 5분

// ── 요청 타입 ──

enum NetRequest {
    Fire { text: String, reply: oneshot::Sender<serde_json::Value> },
    Teach { input: String, target: String, reply: oneshot::Sender<serde_json::Value> },
    Feedback { fire_id: u64, positive: bool, strength: f64, reply: oneshot::Sender<()> },
    Status { reply: oneshot::Sender<serde_json::Value> },
    Save { reply: oneshot::Sender<()> },
}

struct AppState {
    tx: channel::Sender<NetRequest>,
}

#[derive(Deserialize)]
pub struct FireReq { pub text: String }

#[derive(Deserialize)]
pub struct TeachReq { pub input: String, pub target: String }

#[derive(Deserialize)]
pub struct FeedbackReq {
    pub fire_id: u64,
    pub positive: bool,
    #[serde(default = "default_strength")]
    pub strength: f64,
}
fn default_strength() -> f64 { 1.0 }

// ── 핸들러 ──

async fn fire(state: web::Data<AppState>, req: web::Json<FireReq>) -> HttpResponse {
    let (tx, rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Fire { text: req.text.clone(), reply: tx });
    match rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn teach(state: web::Data<AppState>, req: web::Json<TeachReq>) -> HttpResponse {
    let (tx, rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Teach { input: req.input.clone(), target: req.target.clone(), reply: tx });
    match rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn feedback(state: web::Data<AppState>, req: web::Json<FeedbackReq>) -> HttpResponse {
    let (tx, rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Feedback { fire_id: req.fire_id, positive: req.positive, strength: req.strength, reply: tx });
    match rx.await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn status(state: web::Data<AppState>) -> HttpResponse {
    let (tx, rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Status { reply: tx });
    match rx.await {
        Ok(resp) => HttpResponse::Ok().json(resp),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

async fn save(state: web::Data<AppState>) -> HttpResponse {
    let (tx, rx) = oneshot::channel();
    let _ = state.tx.send(NetRequest::Save { reply: tx });
    match rx.await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err(_) => HttpResponse::InternalServerError().json(serde_json::json!({"error": "worker died"})),
    }
}

pub async fn run_server(net: Network, port: u16, save_path: PathBuf, debug: bool) -> std::io::Result<()> {
    let (tx, rx) = channel::bounded::<NetRequest>(256);
    let state = web::Data::new(AppState { tx });
    let save_path_display = save_path.display().to_string();

    std::thread::Builder::new()
        .name("worker".into())
        .spawn(move || {
            let mut net = net;
            net.debug = debug;
            let mut last_save = Instant::now();
            let mut requests_since_save = 0u64;
            loop {
                match rx.recv_timeout(Duration::from_secs(10)) {
                    Ok(req) => {
                        requests_since_save += 1;
                        match req {
                            NetRequest::Fire { text, reply } => {
                                let fire_id = net.fire(&text);
                                let output = net.get_last_output();
                                let mut resp = serde_json::json!({
                                    "fire_id": fire_id,
                                    "output": output,
                                });
                                if net.debug {
                                    if let Some(rec) = net.get_last_record() {
                                        let traces: Vec<serde_json::Value> = rec.traces.iter()
                                            .map(|t| serde_json::json!({
                                                "from": net.nid_label(t.from),
                                                "to": net.nid_label(t.to),
                                                "from_potential": format!("{:.2}", t.from_potential),
                                                "to_before": format!("{:.2}", t.to_before),
                                                "weight": format!("{:.3}", t.weight),
                                                "delivered": format!("{:.3}", t.delivered),
                                                "tick": t.tick,
                                            }))
                                            .collect();
                                        resp["traces"] = serde_json::json!(traces);
                                    }
                                }
                                let _ = reply.send(resp);
                            }
                            NetRequest::Teach { input, target: _, reply } => {
                                // teach 제거 — fire만 수행
                                let fire_id = net.fire(&input);
                                let output = net.get_last_output();
                                let _ = reply.send(serde_json::json!({
                                    "fire_id": fire_id,
                                    "output": output,
                                }));
                            }
                            NetRequest::Feedback { fire_id, positive, strength, reply } => {
                                net.feedback(fire_id, positive, strength);
                                let _ = reply.send(());
                            }
                            NetRequest::Status { reply } => {
                                let _ = reply.send(net.get_status());
                            }
                            NetRequest::Save { reply } => {
                                net.save(&save_path);
                                last_save = Instant::now();
                                requests_since_save = 0;
                                let _ = reply.send(());
                            }
                        }
                    },
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {}
                    Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                        // 종료 전 저장
                        if requests_since_save > 0 {
                            net.save(&save_path);
                        }
                        break;
                    }
                }

                // 자동 저장
                if last_save.elapsed() >= AUTO_SAVE_INTERVAL && requests_since_save > 0 {
                    net.save(&save_path);
                    last_save = Instant::now();
                    requests_since_save = 0;
                }
            }
        })
        .expect("워커 스레드 생성 실패");

    eprintln!("  [서버] http://127.0.0.1:{port} 에서 시작");
    eprintln!("  [저장 경로] {}", save_path_display);
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(1048576))
            .route("/fire", web::post().to(fire))
            .route("/teach", web::post().to(teach))
            .route("/feedback", web::post().to(feedback))
            .route("/status", web::get().to(status))
            .route("/save", web::post().to(save))
    })
    .keep_alive(Duration::from_secs(300))
    .client_request_timeout(Duration::from_secs(300))
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
