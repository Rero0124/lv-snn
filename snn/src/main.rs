mod bitmap;
mod network;
mod neuron;
mod region;
mod server;
mod synapse;
mod tokenizer;

use network::Network;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

const DEFAULT_SAVE_PATH: &str = "data/snn.json";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let serve_mode = args.iter().any(|a| a == "--serve");
    let debug_mode = args.iter().any(|a| a == "--debug");
    let port: u16 = args.iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let save_path = PathBuf::from(
        args.iter()
            .position(|a| a == "--data")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
            .unwrap_or(DEFAULT_SAVE_PATH)
    );

    // 저장 디렉토리 생성
    if let Some(parent) = save_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // 기존 스냅샷 불러오기, 없으면 새로 생성
    let net = if save_path.exists() {
        Network::load(&save_path).unwrap_or_else(|| {
            eprintln!("  [경고] 스냅샷 로드 실패, 새로 생성");
            Network::new()
        })
    } else {
        Network::new()
    };

    if serve_mode {
        run_server(net, port, save_path, debug_mode);
    } else {
        run_interactive(net, save_path);
    }
}

#[tokio::main]
async fn run_server(net: Network, port: u16, save_path: PathBuf, debug: bool) {
    println!("=== LV-SNN v2 (Spike-based) ===");
    println!("  POST /fire     {{\"text\": \"...\"}}");
    println!("  POST /teach    {{\"input\": \"...\", \"target\": \"...\"}}");
    println!("  POST /feedback {{\"fire_id\": N, \"positive\": bool}}");
    println!("  POST /save");
    println!("  GET  /status");
    println!();

    if let Err(e) = server::run_server(net, port, save_path, debug).await {
        eprintln!("서버 오류: {e}");
    }
}

fn run_interactive(mut net: Network, save_path: PathBuf) {
    println!("=== LV-SNN v2 (Spike-based) ===");
    net.print_summary();
    println!();
    println!("명령어: 텍스트→발화, /teach 입력|목표, +N/-N 피드백, /save, /status, /quit");
    println!();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }
        let line = line.trim();
        if line.is_empty() { continue; }

        if line == "/quit" || line == "/q" {
            net.save(&save_path);
            return;
        }
        if line == "/save" {
            net.save(&save_path);
            continue;
        }
        if line == "/status" || line == "/s" {
            net.print_summary();
            continue;
        }
        if line.starts_with("/teach") {
            println!("  teach 제거됨 — fire + feedback으로 학습");
            continue;
        }
        if let Some(rest) = line.strip_prefix('+') {
            if let Ok(id) = rest.trim().parse::<u64>() {
                net.feedback(id, true, 1.0);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix('-') {
            if let Ok(id) = rest.trim().parse::<u64>() {
                net.feedback(id, false, 1.0);
            }
            continue;
        }

        net.fire(line);
    }
}
