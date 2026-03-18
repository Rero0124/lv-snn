mod fire_engine;
mod hippocampus;
mod network;
mod neuron;
mod region;
mod server;
mod synapse;
mod tokenizer;

use network::Network;
use region::RegionType;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let serve_mode = args.iter().any(|a| a == "--serve");
    let reset_mode = args.iter().any(|a| a == "--reset");
    let port: u16 = args.iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    // SIGTERM, SIGINT 종료 플래그
    let shutdown = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&shutdown)).ok();
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&shutdown)).ok();

    let db_path = PathBuf::from("data/network.redb");
    let mut net = Network::new(db_path.clone(), 50000, Arc::clone(&shutdown));

    if reset_mode || !net.try_load_state() {
        if reset_mode {
            println!("  [리셋] 네트워크 구조 재생성 (시냅스 DB는 유지)...");
        } else {
            println!("  새 네트워크 생성 (기존 시냅스 DB 유지)...");
        }
        net.add_region(RegionType::Input, 256);
        net.add_region(RegionType::Emotion, 128);
        net.add_region(RegionType::Reason, 128);
        net.add_region(RegionType::Storage, 512);
        net.add_region(RegionType::Output, 128);
        // 초기 랜덤 빈 시냅스: 구조적 연결 씨앗 (해마가 학습으로 다듬음)
        net.seed_random_synapses(30);
    }

    if serve_mode {
        run_server(net, port);
    } else {
        run_interactive(net, shutdown);
    }
}

#[tokio::main]
async fn run_server(net: Network, port: u16) {
    println!("=== LV-SNN 서버 모드 ===");

    println!("  POST /fire       {{\"text\": \"...\"}})");
    println!("  POST /teach      {{\"input\": \"...\", \"target\": \"...\"}}");
    println!("  POST /feedback   {{\"fire_id\": N, \"positive\": bool}}");
    println!("  GET  /status");
    println!("  POST /save");
    println!();

    if let Err(e) = server::run_server(net, port).await {
        eprintln!("서버 오류: {e}");
    }
}

fn run_interactive(mut net: Network, shutdown: Arc<AtomicBool>) {
    println!("=== LV-SNN 대화형 모드 (토큰 기반) ===");
    net.print_summary();
    println!();
    println!("명령어:");
    println!("  텍스트    → 발화");
    println!("  +N [0~1] → 발화 #N 강화 (예: +1, +1 0.8)");
    println!("  -N [0~1] → 발화 #N 약화 (예: -1, -1 0.3)");
    println!("  /teach 입력|목표 → 목표 기반 학습 (예: /teach 안녕|안녕하세요)");
    println!("  /train N → N초간 자동 학습");
    println!("  /status  → 상태");
    println!("  /quit    → 종료");
    println!();

    let stdin = io::stdin();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            eprintln!("  [시그널] 종료 요청 감지, 상태 저장 중...");
            net.save_state();
            eprintln!("  [시그널] 저장 완료, 종료합니다.");
            break;
        }

        print!("> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            net.save_state();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line == "/quit" || line == "/q" {
            net.save_state();
            let _ = writeln!(io::stdout(), "종료합니다.");
            return;
        }
        if let Some(args) = line.strip_prefix("/train") {
            let secs: u64 = args.trim().parse().unwrap_or(300);
            let pairs: Vec<(&str, &str)> = vec![
                ("안녕하세요", "안녕하세요"),
                ("안녕", "안녕"),
                ("반갑습니다", "반갑습니다"),
                ("반가워", "반가워요"),
                ("하이", "하이요"),
                ("좋은 아침", "좋은 아침이에요"),
                ("좋은 하루", "좋은 하루 보내세요"),
                ("잘 지내?", "잘 지내요"),
                ("오랜만이야", "오랜만이에요"),
                ("만나서 반가워", "만나서 반갑습니다"),
                ("감사합니다", "감사합니다"),
                ("고마워", "고마워요"),
                ("수고하세요", "수고하세요"),
                ("안녕히 가세요", "안녕히 가세요"),
                ("또 만나요", "또 만나요"),
                ("잘 부탁해", "잘 부탁드려요"),
                ("좋은 저녁", "좋은 저녁이에요"),
                ("어서오세요", "어서오세요"),
                ("환영합니다", "환영합니다"),
            ];
            net.train(&pairs, secs);
            continue;
        }
        if let Some(args) = line.strip_prefix("/teach") {
            let args = args.trim();
            if let Some((input, target)) = args.split_once('|') {
                let input = input.trim();
                let target = target.trim();
                if input.is_empty() || target.is_empty() {
                    println!("  사용법: /teach 입력|목표 (예: /teach 안녕하세요|반갑습니다)");
                } else {
                    net.teach(input, target);
                }
            } else {
                println!("  사용법: /teach 입력|목표 (예: /teach 안녕하세요|반갑습니다)");
            }
            continue;
        }
        if line == "/status" || line == "/s" {
            net.print_summary();
            continue;
        }
        if let Some(rest) = line.strip_prefix('+') {
            let parts: Vec<&str> = rest.trim().splitn(2, ' ').collect();
            if let Ok(id) = parts[0].parse::<u64>() {
                let strength: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                net.feedback(id, true, strength);
            } else {
                println!("  사용법: +N [강도] (예: +1, +1 0.8)");
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix('-') {
            let parts: Vec<&str> = rest.trim().splitn(2, ' ').collect();
            if let Ok(id) = parts[0].parse::<u64>() {
                let strength: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                net.feedback(id, false, strength);
            } else {
                println!("  사용법: -N [강도] (예: -1, -1 0.3)");
            }
            continue;
        }

        net.fire(line);
    }
}
