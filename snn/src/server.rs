use crate::network::Network;
use actix_web::{web, App, HttpServer, HttpResponse};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use crossbeam::channel;

const AUTO_SAVE_INTERVAL: Duration = Duration::from_secs(300);

enum NetRequest {
    Fire { text: String, reply: oneshot::Sender<serde_json::Value> },
    Teach { input: String, target: String, reply: oneshot::Sender<serde_json::Value> },
    Feedback { fire_id: u64, positive: bool, strength: f64, reply: oneshot::Sender<()> },
    Save { reply: oneshot::Sender<()> },
}

struct AppState {
    tx: channel::Sender<NetRequest>,
    cached_status: Arc<Mutex<serde_json::Value>>,
    busy: Arc<AtomicBool>,
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
    let mut s = state.cached_status.lock().unwrap().clone();
    s["busy"] = serde_json::json!(state.busy.load(Ordering::Relaxed));
    HttpResponse::Ok().json(s)
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
    let cached_status = Arc::new(Mutex::new(net.get_status()));
    let busy = Arc::new(AtomicBool::new(false));

    let state = web::Data::new(AppState {
        tx,
        cached_status: Arc::clone(&cached_status),
        busy: Arc::clone(&busy),
    });

    let save_path_display = save_path.display().to_string();

    std::thread::Builder::new()
        .name("tick-loop".into())
        .spawn(move || {
            let mut net = net;
            net.debug = debug;
            let mut last_save = Instant::now();
            let mut last_status_update = Instant::now();
            let mut requests_since_save = 0u64;

            loop {
                match rx.try_recv() {
                    Ok(req) => {
                        busy.store(true, Ordering::Relaxed);
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
                            NetRequest::Teach { input, target, reply } => {
                                let fire_id = net.teach(&input, &target);
                                let output = net.get_last_output();
                                let _ = reply.send(serde_json::json!({
                                    "fire_id": fire_id,
                                    "output": output,
                                    "target": target,
                                }));
                            }
                            NetRequest::Feedback { fire_id, positive, strength, reply } => {
                                net.feedback(fire_id, positive, strength);
                                let _ = reply.send(());
                            }
                            NetRequest::Save { reply } => {
                                net.save(&save_path);
                                last_save = Instant::now();
                                requests_since_save = 0;
                                let _ = reply.send(());
                            }
                        }
                        // 요청 처리 후 status 캐시 갱신
                        *cached_status.lock().unwrap() = net.get_status();
                        busy.store(false, Ordering::Relaxed);
                    }
                    Err(crossbeam::channel::TryRecvError::Empty) => {
                        net.idle_tick();
                        // 1초마다 status 캐시 갱신
                        if last_status_update.elapsed() >= Duration::from_secs(1) {
                            *cached_status.lock().unwrap() = net.get_status();
                            last_status_update = Instant::now();
                        }
                    }
                    Err(crossbeam::channel::TryRecvError::Disconnected) => {
                        if requests_since_save > 0 { net.save(&save_path); }
                        break;
                    }
                }

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
