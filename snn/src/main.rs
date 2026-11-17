mod bitmap;
mod network;
mod neuron;
mod plasticity;
mod region;
mod server;
mod synapse;
mod tokenizer;
mod typecheck;

use network::{NetParams, Network};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

const DEFAULT_SAVE_PATH: &str = "data/snn.json";

fn main() {
    // 실행 기록 첫 줄: 어느 코드로 빌드한 것인지 (build.rs 가 빌드 시점에 넣음)
    println!("[빌드] lv-snn {}", env!("BUILD_ID"));
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
    let stdp_scale: Option<f64> = args.iter()
        .position(|a| a == "--stdp-scale")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok());

    // --typecheck: 세포 종류별 특성 점검 (네트워크를 만들지 않음)
    if args.iter().any(|a| a == "--typecheck") {
        typecheck::run();
        return;
    }
    let num = |name: &str| -> Option<f64> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok())
    };
    // 네트워크 구성: --profile connectome (커넥톰 비율), 세부 값은 --fanin/--gain/--inh/--input-pop 로 덮어씀
    let mut params = if args.iter().any(|a| a == "connectome") && args.iter().any(|a| a == "--profile") {
        NetParams::connectome()
    } else {
        NetParams::default()
    };
    if let Some(v) = num("--fanin") { params.synapses_per_neuron = v as usize; }
    if let Some(v) = num("--gain") { params.hidden_gain = v; }
    if let Some(v) = num("--inh") { params.inhibitory_ratio = v; }
    if let Some(v) = num("--input-pop") { params.input_pop = v as usize; }
    if let Some(v) = num("--out-frac") { params.output_frac = v; }
    if args.iter().any(|a| a == "--lognormal") { params.lognormal = true; }
    // --validated: 초파리 커넥톰에서 검증한 학습 구성
    if args.iter().any(|a| a == "--validated") { params.validated = true; }
    // --morph: 추체세포의 두 구획(세포체 + 수상돌기) 모델. 세포 종류 구분이 필요하므로 --typed 를 함께 켠다
    if args.iter().any(|a| a == "--morph") { params.morph = true; params.typed = true; if num("--inh").is_none() { params.inhibitory_ratio = 0.20; } }
    // --typed: 세포 종류 구분(추체 / Pv / Sst / Vip). 억제성 비율은 따로 주지 않으면 생쥐 피질 기준 20%
    if args.iter().any(|a| a == "--typed") {
        params.typed = true;
        if num("--inh").is_none() { params.inhibitory_ratio = 0.20; }
    }

    // --dopamine: 도파민 뉴런 방식 (feedback 이 보상 입력 뉴런을 발화시키고, 학습은 조절 신호의 기준선 대비 변화로 일어남)
    if args.iter().any(|a| a == "--dopamine" || a == "--da-test") { params.dopamine = true; }
    if let Some(v) = num("--gate-base") { params.gate_base = v; }
    if let Some(v) = num("--da-gain") { params.da_gain = v; }
    // --short-tau T: STDP 의 변화는 T 틱에 걸쳐 사라지는 일시적인 것으로 하고, 도파민이 와야 오래가는 변화가 되게 한다
    if let Some(v) = num("--short-tau") { params.short_tau = Some(v); params.dopamine = true; }
    // --predict: 보상 예측 경로 (예측된 보상에는 도파민이 덜 나온다)
    if args.iter().any(|a| a == "--predict" || a == "--rpe-test") { params.predict = true; params.dopamine = true; }
    if let Some(v) = num("--pred-eta") { params.pred_eta = v; }
    if let Some(v) = num("--val-eta") { params.val_eta = v; }
    // --v2da W / --p2da W: 가치 뉴런 → 도파민, 예측 뉴런 → 도파민 시냅스의 세기
    if let Some(v) = num("--v2da") { params.v2da = v; }
    if let Some(v) = num("--p2da") { params.p2da = Some(v); }
    // --value: 가치 경로 (예고 자극이 스스로 도파민을 일으킴 → 2차, 3차 조건화)
    if args.iter().any(|a| a == "--value" || a == "--order2-test") { params.value = true; params.dopamine = true; }
    // --release P: 확률적 전달 (은닉 뉴런의 시냅스가 확률 P 로만 전달, 평균 전달량은 유지)
    if let Some(v) = num("--release") { params.release = Some(v); }
    // --needs: 1차 보상(먹이, 물)과 몸의 상태(배고픔, 목마름)
    if args.iter().any(|a| a == "--needs" || a == "--needs-test") { params.needs = true; params.dopamine = true; }
    // 아래 옵션들은 시험 분기(--needs-test, --op-test 등)보다 먼저 읽는다. 시험 분기는 실행하고 바로 끝나므로, 그 뒤에서 읽으면 시험에 적용되지 않는다
    // --end-on-output: 1대1 방식에서 출력 뉴런이 조용해지면 반응이 끝난 것으로 본다 (내부 활동은 계속돼도 됨)
    if args.iter().any(|a| a == "--end-on-output") { params.end_on_output = true; }
    // --spont X: 자발 발화 배수 (검증 구성에서는 기본 0)
    if let Some(v) = num("--spont") { params.spont = Some(v); }
    // --spont-fixed P: 자발 발화 확률을 입력 가중치와 무관하게 고정 (틱당 확률 P × 종류 배수. --spont 의 배수는 쓰지 않는다) / --spont-flat: 종류별 자발 배수를 끈다
    //   (--spont-flat 은 --spont 나 --spont-fixed 가 없으면, --unit-med-fix 는 --grow 가 없으면 효과가 없다)
    if let Some(v) = num("--spont-fixed") { params.spont_fixed = Some(v); }
    if args.iter().any(|a| a == "--spont-flat") { params.spont_flat = true; }
    // --unit-med-fix: 새 접점 세기 기준에서 조절(도파민) 뉴런이 보내는 입력을 뺀다
    if args.iter().any(|a| a == "--unit-med-fix") { params.unit_med_fix = true; }
    // --topo: 입력마다 정해진 자리 주변으로만 연결 / --local F: 영역 내부 연결 중 가까운 뉴런으로 가는 비율
    if args.iter().any(|a| a == "--topo") { params.topo = true; }
    if let Some(v) = num("--local") { params.local_frac = v; }
    // --scale X: 은닉 영역 크기 배수 / --grow R --target N --prune-tau T: 발달 과정 (성장 + 가지치기) / --min-weight W: 가지치기 기준
    if let Some(v) = num("--scale") { params.scale = v; }
    if let Some(v) = num("--grow") { params.grow = Some(v); }
    if let Some(v) = num("--target") { params.target_syn = v as usize; }
    if let Some(v) = num("--prune-tau") { params.prune_tau = v; }
    if let Some(v) = num("--grow-sigma") { params.grow_sigma = v; }
    if let Some(v) = num("--new-weight") { params.new_weight = v; }
    // --grow-da B: 성장 속도를 도파민에 맡긴다 (기본 성장 B + 조절 수준이 기준선보다 높은 정도)
    if let Some(v) = num("--grow-da") { params.grow_da = Some(v); }
    // --compete K: 접점끼리의 경쟁 (받는 뉴런의 감응도가 낮을수록 굳지 않은 접점의 수명이 짧아짐)
    if let Some(v) = num("--compete") { params.compete = v; }
    if let Some(v) = num("--new-spread") { params.new_spread = v; }
    if let Some(v) = num("--min-weight") { params.min_weight = v; }
    // --seed N: 배선 씨앗 (초기 배선까지만 같아진다. 워밍업과 실행 중의 난수에는 씨앗이 없다) / --scatter-co: 처음부터 있던 접점의 기한을 흩는다
    if let Some(v) = num("--seed") { params.seed = Some(v as u64); }
    if args.iter().any(|a| a == "--scatter-co") { params.scatter_co = true; }
    // --act-target K: 활동 목표 = K × 자기 자발 발화 확률 / --grow-gate fast|slow|both: 받기 문 / --prune-act E: 활동 의존 소멸의 지수
    if let Some(v) = num("--act-target") { params.act_target = Some(v); }
    if let Some(i) = args.iter().position(|a| a == "--grow-gate") {
        params.grow_gate = match args.get(i + 1).map(|s| s.as_str()) {
            Some("fast") => Some(network::GateMode::Fast),
            Some("slow") => Some(network::GateMode::Slow),
            Some("both") => Some(network::GateMode::Both),
            other => { eprintln!("[오류] --grow-gate 는 fast, slow, both 중 하나여야 한다 (받은 값: {:?})", other); std::process::exit(2); }
        };
    }
    if let Some(v) = num("--prune-act") { params.prune_act = v; }
    // --slow-events: 지연 시냅스를 "발화했을 때만 알리기"로 계산 (규칙은 같다) / --slow-check: 옛 방식의 값도 함께 계산해 차이를 1000틱마다 찍는다
    // --warmup-quiet: 기준선을 잡는 준비 구간 동안만 은닉 뉴런의 자발 발화를 끈다
    if args.iter().any(|a| a == "--warmup-quiet") { params.warmup_quiet = true; }
    // --load P: 저장된 망을 불러온다 (시험 분기들이 망을 새로 만드는 자리에서 그 파일을 읽는다. 설정은 저장된 것을 쓴다)
    if let Some(i) = args.iter().position(|a| a == "--load") { params.load_from = args.get(i + 1).cloned(); }
    // --wake-check: 깨어남 확인 (동작은 그대로. "변동 있는 뉴런만 판정"의 규칙이 놓치는 발화가 있는지 센다)
    if args.iter().any(|a| a == "--wake-check") { params.wake_check = true; }
    if args.iter().any(|a| a == "--slow-events") { params.slow_events = true; }
    if args.iter().any(|a| a == "--slow-check") {
        params.slow_check = true;
        if !params.slow_events { println!("[경고] --slow-check 는 --slow-events 와 함께 써야 한다. --slow-events 를 켠다"); params.slow_events = true; }
    }
    if params.act_target.is_some() && params.spont_fixed.is_none() {
        println!("[경고] --act-target 을 --spont-fixed 없이 주면 목표(K × 자발 발화 확률)가 1만 틱마다의 갱신에서 움직인다");
    }
    if (params.grow_gate.is_some() || params.prune_act > 0.0) && params.act_target.is_none() {
        println!("[경고] --grow-gate 와 --prune-act 는 --act-target 이 없으면 효과가 없다 (목표가 없다)");
    }
    if (params.act_target.is_some() || params.scatter_co) && params.grow.is_none() {
        println!("[경고] --act-target, --grow-gate, --prune-act, --scatter-co 는 --grow 가 없으면 효과가 없다");
    }
    if params.prune_act > 0.0 && params.compete > 0.0 {
        println!("[경고] --prune-act 와 --compete 는 같이 쓰지 않는다 (둘 다 받는 쪽 점검의 수명을 줄인다)");
    }
    // --needs-test: 결핍 정도에 따라 같은 감각 입력이 도파민을 일으키는 정도가 달라지는지
    if args.iter().any(|a| a == "--needs-test") {
        let mut net = Network::with_params(params.clone());
        let da = net.da_count() as f64;
        println!("{:?}", params);
        println!("\n감각 입력을 20틱 켰을 때 그 20틱 동안 도파민 뉴런 한 개의 발화 수 (평소 약 0.6), 그리고 결핍 정도의 변화");
        println!("{:<34} {:>8} {:>20} {:>20}", "상황", "도파민", "배고픔 (전 → 후)", "목마름 (전 → 후)");
        let mut probe = |net: &mut Network, name: &str, hunger: f64, thirst: f64, kind: usize| {
            net.rest(300);
            net.set_need(0, hunger); net.set_need(1, thirst);
            net.rest(60);
            let (h0, t0, d0) = (net.need_level(0), net.need_level(1), net.da_spikes);
            net.drive_primary(kind, 20);
            net.rest(20);
            let d = (net.da_spikes - d0) as f64 / da;
            net.rest(60);
            println!("{:<34} {:>8.1} {:>12.2} → {:>4.2} {:>12.2} → {:>4.2}", name, d, h0, net.need_level(0), t0, net.need_level(1));
        };
        probe(&mut net, "배고픔 1.0, 먹이", 1.0, 0.0, 0);
        probe(&mut net, "배고픔 0.7, 먹이", 0.7, 0.0, 0);
        probe(&mut net, "배고픔 0.0 (배부름), 먹이", 0.0, 0.0, 0);
        probe(&mut net, "배고픔 1.0, 물 (목마르지 않음)", 1.0, 0.0, 1);
        probe(&mut net, "목마름 1.0, 물", 0.0, 1.0, 1);
        probe(&mut net, "목마름 1.0, 먹이 (배고프지 않음)", 0.0, 1.0, 0);
        // 먹이를 연달아 줄 때: 배가 차면서 도파민이 줄어드는지
        net.set_need(0, 1.0); net.set_need(1, 0.0);
        net.rest(100);
        print!("\n배고픔 1.0 에서 먹이를 연달아 10번 (도파민 / 배고픔):");
        for _ in 0..10 {
            let d0 = net.da_spikes;
            net.drive_primary(0, 20);
            net.rest(20);
            let d = (net.da_spikes - d0) as f64 / da;
            net.rest(80);
            print!("  {:.1}/{:.2}", d, net.need_level(0));
        }
        println!();
        return;
    }
    // --op-test N: 기능 시험. 입력 A 에 정해 둔 출력이 나오면 보상. 그 출력이 나오는 비율이 올라가는지 (출력만 보고 보상을 준다)
    if let Some(n) = num("--op-test") {
        params.dopamine = true;
        op_test(&params, n as usize);
        return;
    }
    // --val-drift N: 보상 없이 A 만 N번 반복하면서 가치 뉴런과 도파민이 언제부터 벌어지는지
    if let Some(n) = num("--val-drift") {
        params.value = true; params.dopamine = true;
        println!("{:?}", params);
        let mut net = Network::with_params(params.clone());
        net.rest(300);
        // 먼저 쉬는 상태에서 가치 뉴런이 읽는 도파민의 흔들림: 평균, 표준편차, +50% 를 넘은 비율, −50% 아래로 내려간 비율
        let (mut sum, mut sq, mut hi, mut lo, mut cnt) = (0.0, 0.0, 0usize, 0usize, 0usize);
        for _ in 0..5000 { net.step(); for r in net.value_deviations() { sum += r; sq += r * r; if r > 0.5 { hi += 1; } if r < -0.5 { lo += 1; } cnt += 1; } }
        let mean = sum / cnt.max(1) as f64;
        println!("\n쉬는 상태 5000틱, 가치 뉴런이 읽는 도파민의 기준선 대비 벗어남: 평균 {:+.3}, 표준편차 {:.3}, +50% 초과 {:.2}%, −50% 미만 {:.2}%",
            mean, (sq / cnt.max(1) as f64 - mean * mean).max(0.0).sqrt(), hi as f64 * 100.0 / cnt.max(1) as f64, lo as f64 * 100.0 / cnt.max(1) as f64);
        let (_, _, set_a) = run_window(&mut net, "안녕", 330, None, None);
        println!("\n보상 없이 A = 안녕 만 반복. 구간 평균: 가치 뉴런 발화 / 예측 뉴런 발화 / 자극 뒤 0~20, 20~40, 40~60틱의 도파민 (평소 0.6) / 가치 뉴런 입력 가중치 (A 에 반응한 뉴런에서 오는 것, 그 밖)");
        println!("{:>8} {:>10} {:>10} {:>22} {:>12} {:>12}", "시행", "가치뉴런", "예측뉴런", "도파민 (세 구간)", "A 쪽 입력", "그 밖 입력");
        let block = ((n as usize) / 16).max(1);
        let (mut v, mut pr, mut b, mut k) = (0u64, 0u64, [0.0f64; 3], 0usize);
        for t in 1..=(n as usize) {
            let p0 = net.pred_spikes;
            let (bins, vs, _) = run_window(&mut net, "안녕", 330, None, None);
            v += vs; pr += net.pred_spikes - p0;
            for i in 0..3 { b[i] += bins[i]; }
            k += 1;
            if t % block == 0 {
                let (wa, wo) = net.value_input_weights(&set_a);
                println!("{:>8} {:>10.1} {:>10.1} {:>8.2} {:>6.2} {:>6.2} {:>12.5} {:>12.5}", t, v as f64 / k as f64, pr as f64 / k as f64, b[0] / k as f64, b[1] / k as f64, b[2] / k as f64, wa, wo);
                v = 0; pr = 0; b = [0.0; 3]; k = 0;
            }
        }
        return;
    }
    // --order2-test N: 2차 조건화 점검. (1) A → 보상 N번, (2) B → A (보상 없음) N번. 비교: (1) 없이 (2)만 한 네트워크
    if let Some(n) = num("--op2-test") {
        params.dopamine = true;
        op2_test(&params, n as usize);
        return;
    }
    if let Some(n) = num("--order2-test") {
        order2_test(&params, n as usize);
        return;
    }
    // --spont-test TICKS: 입력 없이 자발 발화만으로 돌렸을 때 활동이 번지는지 (배수별)
    //   --spont-list A,B,...: 시험할 배수 목록 (주지 않으면 1, 100, 1000, 10000, 30000)
    if let Some(n) = num("--spont-test") {
        let list: Vec<f64> = args.iter().position(|a| a == "--spont-list").and_then(|i| args.get(i + 1))
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1.0, 100.0, 1000.0, 10000.0, 30000.0]);
        // --spont-fixed 를 주면 배수가 쓰이지 않아 모든 행이 같은 설정이 된다: 알리고 한 행만 돈다
        let list = if params.spont_fixed.is_some() {
            println!("[경고] --spont-fixed 가 있으면 자발 확률이 고정되어 --spont-list 의 배수는 쓰이지 않는다. 한 행만 돈다 (배수 열은 뜻이 없다)");
            list[..1.min(list.len())].to_vec()
        } else { list };
        spont_test(&params, n as u64, &list);
        return;
    }
    // --spread-test: 자극 한 번 뒤 틱별·영역별 발화 수 (활동이 어디서부터 번지는지)
    if args.iter().any(|a| a == "--spread-test") {
        params.dopamine = true;
        let mut net = Network::with_params(params.clone());
        println!("{:?}", params);
        let names = ["감정Main", "감정Map", "이성Main", "이성Map", "기억", "해마", "출력"];
        let sizes = net.area_sizes();
        println!("{:>4} {}", "틱", names.iter().zip(sizes.iter()).map(|(n, s)| format!("{:>14}", format!("{n}({s})"))).collect::<String>());
        net.rest(100);
        net.drive_text("안녕", 4);
        for t in 1..=16 {
            net.step();
            let mut c = [0usize; 8];
            for &id in &net.tick_ids { c[net.area_of(id as usize)] += 1; }
            println!("{:>4} {}", t, c[..7].iter().map(|v| format!("{:>14}", v)).collect::<String>());
        }
        return;
    }
    // --dev-test TICKS: 발달 과정 점검. 자발 발화와 가끔 들어오는 자극만으로 돌리며 시냅스 수의 변화를 본다
    if let Some(n) = num("--dev-test") {
        params.dopamine = true;
        // --report N: 보고 간격을 N틱으로 고정 (없으면 전체의 1/60, 최소 1000틱) / --dev-no-stim: 단어 자극과 먹이·물을 넣지 않는다
        // --save-every N --save-path P: N틱마다 P.t<틱> 으로 저장 (최신 둘과 이정표 200만·1000만·2500만·5000만 틱만 남긴다) / --load P: 저장된 망에서 이어서
        let save = match (num("--save-every"), args.iter().position(|a| a == "--save-path").and_then(|i| args.get(i + 1))) {
            (Some(n), Some(p)) if n >= 1.0 => Some((n as u64, p.clone())),
            (None, None) => None,
            _ => { eprintln!("[오류] --save-every N 과 --save-path P 는 함께 줘야 한다"); std::process::exit(2); }
        };
        dev_test(&params, n as u64, num("--report").map(|v| (v as u64).max(1)), args.iter().any(|a| a == "--dev-no-stim"), save);
        return;
    }
    // --persist-test N: 학습 전후로 자극이 끝난 뒤 활동이 얼마나 오래 남는지 (A 를 N번 반복 / A 에 N번 보상)
    if let Some(n) = num("--persist-test") {
        params.dopamine = true;
        persist_test(&params, n as usize);
        return;
    }
    // --stream-test N: 연속 모드 점검. 자극의 끝을 기다리지 않고 A, B 를 불규칙한 간격으로 N번씩 넣고, A 가 시작된 지 30틱 뒤에 보상(처벌)을 켠다
    if let Some(n) = num("--stream-test") {
        params.dopamine = true;
        stream_test(&params, n as usize);
        return;
    }
    // --rpe-test N: 입력 A 뒤에 매번 보상을 N번 줄 때 도파민 반응이 줄어드는지 (예측 경로가 있을 때와 없을 때)
    if let Some(n) = num("--rpe-test") {
        rpe_test(&params, n as usize);
        return;
    }

    // --da-test N: 도파민 방식 점검. (1) 기준선 발화와 보상/처벌 때의 변화, (2) 두 입력 중 하나에만 보상(또는 처벌)을 줬을 때 그 입력만 달라지는지
    if let Some(n) = num("--da-test") {
        da_test(&params, n as usize);
        return;
    }

    // --calibrate N: 새 네트워크로 피드백 없이 N번 fire 하며 활동량이 유지되는지 출력
    if let Some(n) = num("--calibrate") {
        let mut net = Network::with_params(params.clone());
        if let Some(s) = stdp_scale { net.set_stdp_scale(s); }
        let words = ["안녕", "고마워", "사랑해", "잘자", "배고파"];
        let marks = [1usize, 2, 5, 10, 30, 100, 300, 1000, 3000];
        let mut acc = (0usize, 0usize, 0usize, 0usize);
        // 입력별로 직전에 켜진 은닉 뉴런 집합: 같은 입력의 일관성과 다른 입력과의 겹침을 본다
        let mut last_sets: Vec<Vec<u32>> = vec![Vec::new(); words.len()];
        let (mut same_sum, mut same_n) = (0.0f64, 0usize);
        // 같은 입력에 같은 출력 문자열이 나오는 비율, 서로 다른 입력의 출력이 서로 다른지
        let mut last_out: Vec<Option<String>> = vec![None; words.len()];
        let (mut out_same, mut out_n) = (0usize, 0usize);
        let jaccard = |a: &[u32], b: &[u32]| -> f64 {
            if a.is_empty() && b.is_empty() { return 0.0; }
            let sb: std::collections::HashSet<u32> = b.iter().cloned().collect();
            let inter = a.iter().filter(|x| sb.contains(x)).count();
            inter as f64 / (a.len() + b.len() - inter) as f64
        };
        println!("{:?}", params);
        println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>10}   (구간 평균: 총 발화, 출력 발화, 발화 뉴런 수 / 같은 입력 일관성, 다른 입력과의 겹침)",
            "fire", "spikes", "output", "neurons", "일관성", "다른입력겹침");
        println!("        (뒤의 두 열: 같은 입력에 직전과 같은 출력 문자열이 나온 비율, 서로 다른 출력 문자열 수 / 입력 수)");
        for i in 1..=(n as usize) {
            let wi = i % words.len();
            net.fire(words[wi]);
            let (s, o, d) = net.last_stats();
            acc = (acc.0 + s, acc.1 + o, acc.2 + d, acc.3 + 1);
            let set = net.last_active_hidden();
            if !last_sets[wi].is_empty() { same_sum += jaccard(&last_sets[wi], &set); same_n += 1; }
            last_sets[wi] = set;
            let out = net.get_last_output();
            if let Some(prev) = &last_out[wi] { out_n += 1; if *prev == out { out_same += 1; } }
            last_out[wi] = Some(out);
            if marks.contains(&i) || i == n as usize {
                let (mut cross, mut pairs) = (0.0, 0usize);
                for a in 0..words.len() { for b in a + 1..words.len() {
                    if !last_sets[a].is_empty() && !last_sets[b].is_empty() { cross += jaccard(&last_sets[a], &last_sets[b]); pairs += 1; }
                } }
                let distinct: std::collections::HashSet<&String> = last_out.iter().flatten().filter(|o| !o.is_empty()).collect();
                println!("{:>6} {:>8} {:>8} {:>8} {:>8.2} {:>10.2} {:>9.2} {:>6}/{}", i, acc.0 / acc.3, acc.1 / acc.3, acc.2 / acc.3,
                    if same_n > 0 { same_sum / same_n as f64 } else { 0.0 }, if pairs > 0 { cross / pairs as f64 } else { 0.0 },
                    if out_n > 0 { out_same as f64 / out_n as f64 } else { 0.0 }, distinct.len(), words.len());
                out_same = 0; out_n = 0;
                acc = (0, 0, 0, 0);
                same_sum = 0.0; same_n = 0;
            }
        }
        return;
    }

    let mut net = if save_path.exists() {
        Network::load(&save_path).unwrap_or_else(|| {
            eprintln!("  [경고] 스냅샷 로드 실패, 새로 생성");
            Network::with_params(params.clone())
        })
    } else {
        Network::with_params(params.clone())
    };

    // 비지도 STDP 크기 배수 (기본 1.0, 0 = 끔)
    if let Some(s) = stdp_scale {
        net.set_stdp_scale(s);
        eprintln!("  [설정] 비지도 STDP 배수 {s}");
    }

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

/// 도파민 방식 점검
fn da_test(params: &NetParams, trials: usize) {
    use network::{REWARD_TAIL, REWARD_TICKS};
    println!("{:?}", params);
    // (1) 도파민 뉴런의 발화: 평소 / 보상 입력 / 처벌 입력
    {
        let mut net = Network::with_params(params.clone());
        let da = net.da_count() as f64;
        let per100 = |spikes: u64, ticks: u64| spikes as f64 / da / ticks as f64 * 100.0;
        let s0 = net.da_spikes; net.rest(500);
        println!("\n[1] 도파민 뉴런 한 개의 발화 수 (100틱당) 와, 받는 뉴런들의 조절 수준이 기준선에서 벗어난 정도 (0 = 평소)");
        println!("  평소                : {:>5.1}   벗어남 {:+.2}", per100(net.da_spikes - s0, 500), net.mod_deviation());
        for (name, positive) in [("보상 입력 20틱", true), ("처벌 입력 20틱", false)] {
            let idx_before = net.da_spikes;
            let mut peak: f64 = 0.0;
            let d = net.deliver_reward_observed(positive, 1.0, &mut peak);
            let tail = net.da_spikes - idx_before - d;
            println!("  {:<18}: {:>5.1}   벗어남 {:+.2} (가장 클 때)   그 뒤 {}틱: {:>5.1}, 끝났을 때 벗어남 {:+.2}",
                name, per100(d, REWARD_TICKS), peak, REWARD_TAIL, per100(tail, REWARD_TAIL), net.mod_deviation());
            net.rest(500);
        }
    }
    // (2) 두 입력 A, B 를 번갈아 주고 A 뒤에만 보상(또는 처벌). B 뒤에는 같은 시간만 흘려보낸다
    println!("\n[2] 입력 A = 안녕, B = 고마워 를 번갈아 {trials}번씩. 조건에 따라 A 뒤에만 보상 또는 처벌");
    println!("    반응 = 한 번의 자극에 발화한 수 (처음 3번 평균 → 마지막 3번 평균), 가중치 = 그 입력에만 반응한 뉴런들 사이 시냅스의 평균 변화");
    println!("{:<14} {:>16} {:>16} {:>12} {:>12} {:>14} {:>14}", "조건", "A 반응", "B 반응", "A 가중치", "B 가중치", "A 출력발화", "B 출력발화");
    for (name, mode) in [("보상 없음", 0), ("A 에 보상", 1), ("A 에 처벌", 2)] {
        let mut net = Network::with_params(params.clone());
        let words = ["안녕", "고마워"];
        let w0 = net.weights_snapshot();
        let mut resp: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let mut outs: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let mut sets: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for t in 0..trials {
            for (wi, w) in words.iter().enumerate() {
                let id = net.fire(w);
                let (s, o, _) = net.last_stats();
                resp[wi].push(s); outs[wi].push(o);
                if t == 0 { sets[wi] = net.last_active(); }
                if wi == 0 && mode > 0 { net.feedback(id, mode == 1, 1.0); } else { net.rest(REWARD_TICKS + REWARD_TAIL); }
            }
        }
        let avg = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
        let k = 3.min(trials);
        let only = |a: &[u32], b: &[u32]| -> Vec<u32> { let sb: std::collections::HashSet<u32> = b.iter().cloned().collect(); a.iter().cloned().filter(|x| !sb.contains(x)).collect() };
        let (a_only, b_only) = (only(&sets[0], &sets[1]), only(&sets[1], &sets[0]));
        let (da, na) = net.weight_change_within(&a_only, &w0);
        let (db, nb) = net.weight_change_within(&b_only, &w0);
        println!("{:<14} {:>7.0} → {:>6.0} {:>7.0} → {:>6.0} {:>+11.1}% {:>+11.1}% {:>6.1} → {:>5.1} {:>6.1} → {:>5.1}   (시냅스 수 A {na}, B {nb})",
            name, avg(&resp[0][..k]), avg(&resp[0][trials - k..]), avg(&resp[1][..k]), avg(&resp[1][trials - k..]),
            da * 100.0, db * 100.0, avg(&outs[0][..k]), avg(&outs[0][trials - k..]), avg(&outs[1][..k]), avg(&outs[1][trials - k..]));
    }
}

/// 보상 예측 점검: A 뒤에 매번 보상. 보상 입력이 켜진 20틱 동안 도파민 뉴런 한 개가 발화한 수가 시행이 쌓이면서 어떻게 변하는지
fn rpe_test(params: &NetParams, trials: usize) {
    println!("{:?}", params);
    println!("\n입력 A = 안녕 뒤에 매번 보상. 값 = 보상 입력이 켜진 20틱 동안 도파민 뉴런 한 개의 발화 수 (평소 20틱이면 약 0.6)");
    println!("{:<16} {:>8} {:>10} {:>10} {:>12} {:>12} {:>12}", "조건", "시행", "도파민", "예측뉴런", "A 반응", "A 가중치", "B 가중치");
    for (name, predict) in [("예측 경로 없음", false), ("예측 경로 있음", true)] {
        let mut p = params.clone();
        p.predict = predict;
        let mut net = Network::with_params(p);
        let da = net.da_count() as f64;
        // 처음에 A, B 각각에만 반응하는 뉴런 집합을 기록 (학습 전 한 번씩, 보상 없이)
        net.fire("안녕"); let sa = net.last_active(); net.rest(120);
        net.fire("고마워"); let sb = net.last_active(); net.rest(120);
        let only = |a: &[u32], b: &[u32]| -> Vec<u32> { let sb: std::collections::HashSet<u32> = b.iter().cloned().collect(); a.iter().cloned().filter(|x| !sb.contains(x)).collect() };
        let (a_only, b_only) = (only(&sa, &sb), only(&sb, &sa));
        let w0 = net.weights_snapshot();
        let block = (trials / 8).max(1);
        let (mut d_acc, mut p_acc, mut r_acc, mut n_acc) = (0u64, 0u64, 0usize, 0usize);
        for t in 1..=trials {
            let p0 = net.pred_spikes;
            net.fire("안녕");
            r_acc += net.last_stats().0;
            p_acc += net.pred_spikes - p0;
            d_acc += net.deliver_reward(true, 1.0);
            n_acc += 1;
            if t % block == 0 || t == trials {
                let (wa, _) = net.weight_change_within(&a_only, &w0);
                let (wb, _) = net.weight_change_within(&b_only, &w0);
                println!("{:<16} {:>8} {:>10.1} {:>10.1} {:>12.0} {:>+11.1}% {:>+11.1}%", name, t, d_acc as f64 / da / n_acc as f64,
                    p_acc as f64 / n_acc as f64, r_acc as f64 / n_acc as f64, wa * 100.0, wb * 100.0);
                d_acc = 0; p_acc = 0; r_acc = 0; n_acc = 0;
            }
        }
        // 시간 경과: A 를 넣고 15틱 뒤 보상 20틱. 20틱 구간별 도파민 뉴런 한 개의 발화 수 (평소 0.6) 와 받는 뉴런들의 조절 수준이 기준선에서 벗어난 정도
        {
            net.rest(600);
            net.drive_text("안녕", 4);
            let (mut line_da, mut line_dev) = (String::new(), String::new());
            let mut d0 = net.da_spikes;
            for t in 0..240u64 {
                if t == 15 { net.drive_reward(true, 20); }
                net.step();
                if (t + 1) % 20 == 0 {
                    line_da += &format!("{:>6.1}", (net.da_spikes - d0) as f64 / da);
                    line_dev += &format!("{:>+6.2}", net.mod_deviation());
                    d0 = net.da_spikes;
                }
            }
            println!("{:<16} 시간 경과 (20틱 구간, 보상은 15~35틱)\n    도파민 발화: {}\n    기준선 대비: {}", name, line_da, line_dev);
            net.rest(600);
        }
        // 시점 점검: (1) A 뒤에 보상을 주지 않음 — 보상이 오던 20틱 동안의 도파민 (평소 0.6 보다 낮으면 정지)
        //            (2) A 뒤 100틱을 기다린 다음 보상 — 예측이 시점에 묶여 있으면 도파민이 그대로 나온다
        net.fire("안녕"); let d0 = net.da_spikes; net.rest(20); let omit = (net.da_spikes - d0) as f64 / da; net.rest(100);
        net.fire("안녕"); net.rest(100); let late = net.deliver_reward(true, 1.0) as f64 / da;
        for _ in 0..3 { net.fire("안녕"); net.deliver_reward(true, 1.0); }
        println!("{:<16} 시점 점검:  A 뒤 보상 없음 → 도파민 {:.1} (평소 0.6),  A 뒤 100틱 늦은 보상 → 도파민 {:.1}", name, omit, late);
        // 배우지 않은 입력 B 뒤의 보상, 그리고 다시 A 뒤의 보상
        // 직전 A 의 지연 흔적(최대 120틱짜리)이 남지 않도록 충분히 쉰 다음 B 를 준다
        net.rest(600);
        let p0 = net.pred_spikes; net.fire("고마워"); let pb = net.pred_spikes - p0; let db = net.deliver_reward(true, 1.0);
        net.rest(600);
        let p0 = net.pred_spikes; net.fire("안녕"); let pa = net.pred_spikes - p0; let da2 = net.deliver_reward(true, 1.0);
        // 자극 없이 보상만
        net.rest(300);
        let d0 = net.deliver_reward(true, 1.0);
        println!("{:<16} 끝난 뒤:  B 뒤 보상 → 도파민 {:.1} (예측뉴런 {}),  A 뒤 보상 → 도파민 {:.1} (예측뉴런 {}),  자극 없이 보상 → 도파민 {:.1}",
            name, db as f64 / da, pb, da2 as f64 / da, pa, d0 as f64 / da);
    }
}

/// 연속 모드 점검: 시간이 끊기지 않고 흐르는 상태에서도 보상을 준 입력만 달라지는지
fn stream_test(params: &NetParams, trials: usize) {
    use rand::prelude::*;
    const WINDOW: u64 = 30;      // 반응을 세는 구간 (입력이 시작된 뒤 30틱) — 관찰자가 정한 구간일 뿐 네트워크는 모른다
    const REWARD_AT: u64 = 30;   // 입력 시작 뒤 보상이 켜지는 시점
    const REWARD_LEN: u64 = 20;
    println!("{:?}", params);
    println!("\n연속 모드: A = 안녕, B = 고마워 를 100~400틱 간격으로 무작위 순서로 넣는다 (이전 반응·보상과 겹칠 수 있음). A 시작 30틱 뒤 보상/처벌 20틱");
    println!("반응 = 입력 시작 뒤 30틱 동안의 발화 수 (처음 5번 평균 → 마지막 5번 평균). 도파민 = A 의 보상 구간 20틱 동안 도파민 뉴런 한 개의 발화 수");
    println!("{:<12} {:>16} {:>16} {:>11} {:>11} {:>14} {:>14} {:>14}", "조건", "A 반응", "B 반응", "A 가중치", "B 가중치", "A 출력발화", "B 출력발화", "도파민");
    for (name, mode) in [("보상 없음", 0), ("A 에 보상", 1), ("A 에 처벌", 2)] {
        let mut rng = rand::rng();
        let mut net = Network::with_params(params.clone());
        let da = net.da_count() as f64;
        let words = ["안녕", "고마워"];
        let w0 = net.weights_snapshot();
        let mut resp: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let mut outs: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let mut sets: [std::collections::HashSet<u32>; 2] = [Default::default(), Default::default()];
        let mut da_win: Vec<f64> = Vec::new();
        let mut left = [trials, trials];
        let mut total_ticks = 0u64;
        while left[0] + left[1] > 0 {
            let wi = if left[0] == 0 { 1 } else if left[1] == 0 { 0 } else { rng.random_range(0..2) };
            left[wi] -= 1;
            let gap: u64 = rng.random_range(100..=400);
            net.drive_text(words[wi], 4);
            let first = resp[wi].is_empty();
            let (mut r, mut o) = (0usize, 0usize);
            let mut da0 = 0u64;
            for t in 0..gap {
                if wi == 0 && t == REWARD_AT {
                    da0 = net.da_spikes;
                    if mode > 0 { net.drive_reward(mode == 1, REWARD_LEN); }
                }
                if wi == 0 && t == REWARD_AT + REWARD_LEN { da_win.push((net.da_spikes - da0) as f64 / da); }
                net.step();
                total_ticks += 1;
                if t < WINDOW {
                    for &nid in &net.tick_ids {
                        if net.is_input(nid as usize) { continue; }
                        r += 1;
                        if net.is_output(nid as usize) { o += 1; }
                        if first { sets[wi].insert(nid); }
                    }
                }
            }
            resp[wi].push(r); outs[wi].push(o);
        }
        let avg = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
        let avgf = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let k = 5.min(trials);
        let a_only: Vec<u32> = sets[0].difference(&sets[1]).cloned().collect();
        let b_only: Vec<u32> = sets[1].difference(&sets[0]).cloned().collect();
        let (wa, na) = net.weight_change_within(&a_only, &w0);
        let (wb, nb) = net.weight_change_within(&b_only, &w0);
        let kd = k.min(da_win.len());
        println!("{:<12} {:>7.0} → {:>6.0} {:>7.0} → {:>6.0} {:>+10.1}% {:>+10.1}% {:>6.1} → {:>5.1} {:>6.1} → {:>5.1} {:>6.1} → {:>5.1}   (시냅스 수 A {na}, B {nb}, 총 {total_ticks}틱)",
            name, avg(&resp[0][..k]), avg(&resp[0][trials - k..]), avg(&resp[1][..k]), avg(&resp[1][trials - k..]),
            wa * 100.0, wb * 100.0, avg(&outs[0][..k]), avg(&outs[0][trials - k..]), avg(&outs[1][..k]), avg(&outs[1][trials - k..]),
            avgf(&da_win[..kd]), avgf(&da_win[da_win.len() - kd..]));
    }
}

/// 자발 발화 점검: 입력 없이 돌렸을 때, 스스로 발화한 양에 비해 실제 발화가 얼마나 불어나는지
fn spont_test(params: &NetParams, ticks: u64, list: &[f64]) {
    println!("{:?}", params);
    println!("\n입력 없이 {ticks}틱. 넣은 양 = 자발 발화로 기대되는 발화 수, 실제 = 은닉 뉴런의 발화 수 (1000틱당, 구간별)");
    println!("불어남 = 실제 ÷ 넣은 양 (1 = 번지지 않음). 가장 큰 틱 = 한 틱에 발화한 뉴런 수의 최댓값. 뒤의 값 = 그 뒤 자극 A 를 한 번 주고 100~300틱 뒤 200틱 동안의 발화 수");
    println!("실제·가장 큰 틱·자극 뒤 잔여는 은닉 뉴런만 센다 (입출력 뉴런 제외). 불어남의 넣은 양은 틱마다의 기대 자발 발화를 더한 값이다 (불응기를 빼지 않은 상한)");
    println!("{:>8} {:>10} {:>32} {:>8} {:>10} {:>14}   {}", "배수", "넣은 양", "실제 (처음 / 중간 / 마지막 구간)", "불어남", "가장 큰 틱", "자극 뒤 잔여", "입력 가중치 중앙값의 평균 (처음 → 끝), 경과(초)");
    for (row, &scale) in list.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let mut p = params.clone();
        p.spont = Some(scale);
        let mut net = Network::with_params(p);
        let (per_tick, w_med) = net.spont_expected();
        let injected = per_tick * 1000.0;
        if row == 0 { println!("  (은닉 뉴런 {}개, 입력 가중치 중앙값의 평균 {:.4})", net.hidden_count(), w_med); }
        let hidden_n = net.hidden_count();
        let mut exp_acc = 0.0f64;
        let blocks = (ticks / 1000).max(1) as usize;
        let mut per_block = vec![0usize; blocks];
        let mut peak = 0usize;
        // 구간(1000틱)마다 한 줄: 은닉 발화, 기대 자발, 불어남, 한 틱 최대, 도파민·예측 뉴런 발화, 조절 수준의 기준선 대비 편차 (관찰용)
        if row == 0 { println!("  (구간 줄) 도파민·예측 = 그 구간에 뉴런 한 개가 발화한 수 (도파민 {}개, 예측 {}개). 편차 = 조절 수준 ÷ 기준선 − 1 의 평균: 구간 끝 값 (구간 안 최소 ~ 최대)", net.da_count(), net.pred_count()); }
        let (da_n, pred_n) = (net.da_count().max(1) as f64, net.pred_count().max(1) as f64);
        let (mut b_exp, mut b_peak, mut b_da, mut b_pred) = (0.0f64, 0usize, net.da_spikes, net.pred_spikes);
        let (mut dev_lo, mut dev_hi) = (f64::MAX, f64::MIN);
        for t in 0..ticks {
            let e = net.spont_expected().0;
            exp_acc += e; b_exp += e;
            net.step();
            let n = net.tick_ids.iter().filter(|&&id| (id as usize) < hidden_n).count();
            peak = peak.max(n); b_peak = b_peak.max(n);
            let b = ((t / 1000) as usize).min(blocks - 1);
            per_block[b] += n;
            let dev = net.mod_deviation();
            dev_lo = dev_lo.min(dev); dev_hi = dev_hi.max(dev);
            if (t + 1) % 1000 == 0 && ((t + 1) / 1000) as usize <= blocks {
                println!("    구간 {:>3} (배수 {}) | 은닉 발화 {:>7}, 기대 자발 {:>6.0}, 불어남 {:>7.2} | 한 틱 최대 {:>4} ({:.1}%) | 도파민 {:.2}, 예측 {:.2}, 편차 {:+.3} ({:+.3} ~ {:+.3}) | {:.0}초",
                    (t + 1) / 1000, scale, per_block[b], b_exp, if b_exp > 0.0 { per_block[b] as f64 / b_exp } else { 0.0 }, b_peak, b_peak as f64 / hidden_n as f64 * 100.0,
                    (net.da_spikes - b_da) as f64 / da_n, (net.pred_spikes - b_pred) as f64 / pred_n, dev, dev_lo, dev_hi, t0.elapsed().as_secs_f64());
                { use std::io::Write; std::io::stdout().flush().ok(); }
                b_exp = 0.0; b_peak = 0; b_da = net.da_spikes; b_pred = net.pred_spikes; dev_lo = f64::MAX; dev_hi = f64::MIN;
            }
        }
        let total: usize = per_block.iter().sum();
        let w_end = net.spont_expected().1;
        let after_exp = net.spont_expected().0 * 200.0;
        net.drive_text("안녕", 4);
        let mut after = 0usize;
        for t in 0..300 { net.step(); if t >= 100 { after += net.tick_ids.iter().filter(|&&id| (id as usize) < hidden_n).count(); } }
        println!("{:>8} {:>10.0} {:>10} / {:>8} / {:>8} {:>8.2} {:>10} {:>10} (넣은 양 {:.0})   {:.6} → {:.6}, {:.0}초", scale, injected,
            per_block[0], per_block[blocks / 2], per_block[blocks - 1],
            if exp_acc > 0.0 { total as f64 / exp_acc } else { 0.0 }, peak, after, after_exp, w_med, w_end, t0.elapsed().as_secs_f64());
        { use std::io::Write; std::io::stdout().flush().ok(); }
    }
}

/// 자극 A 를 4틱 넣고 300틱 동안 입력 없이 둔다. 반환: 구간별 발화 수 (0~20, 20~50, 50~100, 100~300틱) 와 마지막으로 발화가 있었던 틱
fn after_activity(net: &mut Network) -> ([usize; 4], u64) {
    net.rest(300);
    net.drive_text("안녕", 4);
    let mut bins = [0usize; 4];
    let mut last = 0u64;
    for t in 0..300u64 {
        net.step();
        let n = net.tick_ids.iter().filter(|&&id| !net.is_input(id as usize)).count();
        if n > 0 { last = t + 1; }
        let b = if t < 20 { 0 } else if t < 50 { 1 } else if t < 100 { 2 } else { 3 };
        bins[b] += n;
    }
    (bins, last)
}

/// 학습으로 스스로 이어지는 활동이 생기는지 점검
fn persist_test(params: &NetParams, trials: usize) {
    println!("{:?}", params);
    println!("\n자극 A(4틱)가 끝난 뒤의 활동. 값 = 구간별 발화 수 (자극 시작 기준 틱), 마지막 발화 = 발화가 있었던 마지막 틱");
    println!("{:<14} {:>8} {:>9} {:>9} {:>9} {:>10} {:>12}", "조건", "시행", "0~20", "20~50", "50~100", "100~300", "마지막 발화");
    let show = |name: &str, label: &str, r: ([usize; 4], u64)| {
        println!("{:<14} {:>8} {:>9} {:>9} {:>9} {:>10} {:>12}", name, label, r.0[0], r.0[1], r.0[2], r.0[3], r.1);
    };
    let marks: Vec<usize> = vec![trials / 4, trials / 2, trials];
    for (name, reward) in [("반복만", false), ("A 에 보상", true)] {
        let mut net = Network::with_params(params.clone());
        show(name, "0", after_activity(&mut net));
        for t in 1..=trials {
            let id = net.fire("안녕");
            if reward { net.feedback(id, true, 1.0); } else { net.rest(220); }
            if marks.contains(&t) {
                show(name, &t.to_string(), after_activity(&mut net));
            }
        }
        if !reward { continue; }
        // 무엇이 활동을 끄는지: 학습이 끝난 네트워크에서 한 가지씩 꺼 보고 같은 측정을 한다
        let hidden = net.hidden_count();
        let saved: Vec<(f64, f64)> = net.neurons_mut()[..hidden].iter().map(|n| (n.adapt_inc, n.threshold_scale)).collect();
        let restore = |net: &mut Network| { for (n, s) in net.neurons_mut()[..hidden].iter_mut().zip(&saved) { n.adapt_inc = s.0; n.threshold_scale = s.1; n.adapt = 0.0; } };
        for n in net.neurons_mut()[..hidden].iter_mut() { n.adapt_inc = 0.0; n.adapt = 0.0; }
        show("  적응 끔", "", after_activity(&mut net));
        restore(&mut net);
        for n in net.neurons_mut()[..hidden].iter_mut() { if n.inhibitory { n.threshold_scale = 1e6; } }
        show("  억제 끔", "", after_activity(&mut net));
        for n in net.neurons_mut()[..hidden].iter_mut() { n.adapt_inc = 0.0; n.adapt = 0.0; }
        show("  둘 다 끔", "", after_activity(&mut net));
        restore(&mut net);
        show("  (되돌림)", "", after_activity(&mut net));
    }
}

/// 발달 과정 점검: 뉴런당 시냅스 수가 적음 → 넘침 → 안정의 곡선을 그리는지.
/// 몸의 상태(--needs)가 켜져 있으면 단순한 환경이 붙는다: 배고픔이 차면 "안녕" 뒤에 먹이, 목마름이 차면 "고마워" 뒤에 물
fn dev_test(params: &NetParams, ticks: u64, report_every: Option<u64>, no_stim: bool, save: Option<(u64, String)>) {
    use rand::prelude::*;
    use std::io::Write;
    let mut rng = rand::rng();
    println!("{:?}", params);
    let t0 = std::time::Instant::now();
    let mut net = Network::with_params(params.clone());
    let words = ["안녕", "고마워", "사랑해", "잘자", "배고파"];
    let needs = net.params_ref().needs;
    println!("[발달 시험] 보고 간격 {}, 자극 {}, 초기 배선의 합계 값 {:016x} (같은 배선 씨앗이면 같다. 씨앗 {:?})",
        match report_every { Some(n) => format!("{}틱 (--report)", n), None => "전체의 1/60 (최소 1000틱)".to_string() },
        if no_stim { "없음 (--dev-no-stim: 단어 자극과 먹이·물을 넣지 않는다)" } else { "있음" }, net.wiring_sum, params.seed);
    if no_stim {
        println!("\n뉴런당 내보내는 시냅스 수 (은닉 {}개). 자극 없음", net.hidden_count());
    } else {
    println!("\n뉴런당 내보내는 시냅스 수 (은닉 {}개). 자극은 100~400틱 간격으로 무작위 단어{}", net.hidden_count(),
        if needs { ". 배고픔이 0.8 을 넘으면 '안녕' 30틱 뒤 먹이, 목마름이 0.8 을 넘으면 '고마워' 30틱 뒤 물" } else { "" });
    }
    println!("{:>9} {:>8} {:>6} {:>7} {:>7} {:>6} {:>11} {:>11} {:>9} {:>7} {:>7} {:>8} {:>7}",
        "틱", "시냅스", "최소", "최대", "상대 수", "쌍당", "생긴 수", "사라진 수", "발화/1k", "먹이", "물", "보상도파민", "경과(초)");
    let report = report_every.unwrap_or((ticks / 60).max(1000));
    // 불러온 망이면 저장된 시험 틱 다음부터 이어 간다 (진행 중이던 자극과 난수 상태는 저장되지 않는다)
    let start = net.dev_tick;
    if params.load_from.is_some() { println!("[불러오기] {} — 시험 틱 {} 다음부터 이어서 돈다. 설정은 저장된 것: {:?}", params.load_from.as_deref().unwrap_or(""), start, net.params_ref()); }
    if let Some((n, p)) = &save { println!("[저장] {}틱마다 {}.t<틱> 으로 저장한다 (최신 둘과 이정표 200만·1000만·2500만·5000만 틱만 남긴다)", n, p); }
    let mut saved: Vec<(u64, String)> = Vec::new();
    let (mut next_stim, mut spikes) = (start + 200, 0usize);
    // 예정된 1차 보상: (틱, 종류). 보상 구간의 도파민 발화를 재기 위한 값들
    let mut pending: Option<(u64, usize)> = None;
    let (mut food_n, mut water_n) = (0usize, 0usize);
    let (mut da_acc, mut da_n, mut da_mark) = (0u64, 0usize, None::<(u64, u64)>);
    let da = net.da_count().max(1) as f64;
    // 관찰용 (네트워크 동작은 바꾸지 않는다): 자발 발화의 기대값과 실제 발화를 종류별로 구간마다 모은다
    const KINDS: [&str; 5] = ["추체", "Pv", "Sst", "Vip", "그 밖"];
    let hidden_n = net.hidden_count();
    let kind_of: Vec<usize> = net.hidden_observe().iter().map(|o| o.0).collect();
    let mut kind_n = [0usize; 5];
    for &k in &kind_of { kind_n[k] += 1; }
    let spont_by_kind = |net: &Network| { let mut e = [0.0f64; 5]; for o in net.hidden_observe() { e[o.0] += o.3; } e };
    let (mut exp_now, mut exp_kind_now) = (net.spont_expected().0, spont_by_kind(&net));
    let (mut exp_acc, mut exp_kind_acc, mut hid_kind) = (0.0f64, [0.0f64; 5], [0usize; 5]);
    let (mut last_t, mut last_sprouted, mut last_pruned) = (start, net.sprouted, net.pruned);
    let mut peak_hid = 0usize;
    let (mut last_da, mut last_pred) = (net.da_spikes, net.pred_spikes);
    // 자극 밖: 자극이 시작된 틱부터 50틱(보상이 붙으면 보상이 끝난 뒤 50틱까지)을 뺀 나머지 틱만 따로 센다
    let mut stim_until = 0u64;
    let (mut out_ticks, mut out_hid, mut out_exp, mut out_peak) = (0u64, 0usize, 0.0f64, 0usize);
    println!("  (추가 줄의 뜻) 자극 밖 = 자극이 시작된 틱부터 50틱(보상이 붙는 자극은 보상이 끝나고 50틱까지)을 뺀 틱만 센 값. 입력 자극이 직접 일으킨 발화를 빼고 본다");
    println!("  (추가 줄의 뜻) 기대 자발 = 그 구간 틱마다 은닉 뉴런의 자발 발화 확률(0.00001 × spont_scale)을 더한 값. 불응기와 문턱 발화가 겹치는 틱을 빼지 않으므로 실제 자발 발화의 상한이다");
    println!("  (추가 줄의 뜻) 은닉 발화 = 은닉 뉴런만 센 실제 발화 (윗줄의 발화/1k 는 입출력 뉴런 포함). 발화율 분위 = rate_avg × 1000 (시정수 약 5000틱)");
    println!("  (추가 줄의 뜻) 접점 = 은닉 뉴런이 내보내는 접점을 쌍이 생긴 때로 나눈 것. 처음 쌍(born = 0)에 나중에 더해진 접점은 처음 쪽에 들어간다(그 누계를 따로 찍는다). 자란 쌍은 모두 은닉→은닉");
    println!("  (추가 줄의 뜻) 받기 = 구간에 생긴 접점 수와 받기 문에 걸려 버린 시도 수를 받는 뉴런의 종류별로. 소멸의 나이 = 사라진 틱 − 그 접점이 속한 쌍이 생긴 틱 (처음부터 있던 쌍은 0틱에 생긴 것으로 센다)");
    println!("  (추가 줄의 뜻) 입력 = 은닉 뉴런마다 다른 은닉 뉴런에서 들어오는 접점 수(지연 시냅스 제외). 느린÷목표 = rate_avg ÷ (K × 자기 자발 발화 확률). 문이 닫힌 비율은 보고 시점의 값");
    println!("  (추가 줄의 뜻) 표의 '상대 수' = 서로 다른 상대의 수가 아니라 시냅스 칸의 수 (같은 상대로 가는 칸이 둘 이상일 수 있다: 초기 배선의 중복). '쌍 수'도 칸의 수다. 중복 시냅스 = 같은 (보내는 뉴런, 받는 뉴런)의 두 번째 이상인 칸");
    println!("  (추가 줄의 뜻) 조절 줄의 문 = 은닉 뉴런의 조절 수준을 10틱마다 읽어, 기준선보다 25% 이상 높은 비율(위)과 25% 이상 낮은 비율(아래)을 구간에 걸쳐 평균한 값. 도파민 수상돌기 = 도파민 뉴런 수상돌기 값의 10틱마다 표본 평균");
    let (mut gate_up, mut gate_down, mut gate_n, mut dend_acc, mut dend_n) = (0u64, 0u64, 0u64, 0.0f64, 0u64);
    let (mut last_made, mut last_drop, mut last_short) = (net.grow_made, net.gate_drop, net.pruned_short);
    const AREAS: [&str; 6] = ["감정 주", "감정 지도", "이성 주", "이성 지도", "기억", "해마"];
    for t in (start + 1)..=ticks {
        if needs && !no_stim && pending.is_none() {
            for kind in 0..2 { if pending.is_none() && net.need_level(kind) > 0.8 { net.drive_text(words[kind], 4); pending = Some((t + 30, kind)); stim_until = stim_until.max(t + 30 + 20 + 50); } }
        }
        if let Some((at, kind)) = pending {
            if t == at { net.drive_primary(kind, 20); da_mark = Some((t + 20, net.da_spikes)); if kind == 0 { food_n += 1 } else { water_n += 1 } }
            if t >= at + 200 { pending = None; }
        }
        if let Some((end, start)) = da_mark { if t == end { da_acc += net.da_spikes - start; da_n += 1; da_mark = None; } }
        if t == next_stim && !no_stim {
            if pending.is_none() { net.drive_text(words[rng.random_range(if needs { 2 } else { 0 }..words.len())], 4); stim_until = stim_until.max(t + 50); }
            next_stim = t + rng.random_range(100..=400);
        }
        // 이 틱에 쓰일 자발 발화 확률 (1만 틱마다의 갱신은 step 안에서 일어나므로, 갱신된 틱에는 한 틱 어긋난다)
        exp_acc += exp_now;
        for k in 0..5 { exp_kind_acc[k] += exp_kind_now[k]; }
        net.step();
        spikes += net.tick_ids.len();
        let mut hid_now = 0usize;
        for &id in &net.tick_ids { if (id as usize) < hidden_n { hid_kind[kind_of[id as usize]] += 1; hid_now += 1; } }
        peak_hid = peak_hid.max(hid_now);
        if t >= stim_until { out_ticks += 1; out_hid += hid_now; out_exp += exp_now; out_peak = out_peak.max(hid_now); }
        if t % 10 == 0 {
            let (u, d, n) = net.gate_open_counts();
            gate_up += u as u64; gate_down += d as u64; gate_n += n as u64;
            dend_acc += net.da_dend_mean(); dend_n += 1;
        }
        let e = net.spont_expected().0;
        if e != exp_now { exp_now = e; exp_kind_now = spont_by_kind(&net); }
        if t % report == 0 || t == ticks {
            let (avg, lo, hi, conns) = net.out_degree_stats();
            println!("{:>9} {:>8.0} {:>6} {:>7} {:>7.0} {:>6.1} {:>11} {:>11} {:>9.0} {:>7} {:>7} {:>8.1} {:>7.0}", t, avg, lo, hi, conns, avg / conns.max(1.0), net.sprouted, net.pruned,
                spikes as f64 * 1000.0 / report as f64, food_n, water_n, if da_n > 0 { da_acc as f64 / da / da_n as f64 } else { 0.0 }, t0.elapsed().as_secs_f64());
            let (em, el, eh) = net.excitability_stats();
            println!("{:>9} 감응도 평균 {:.3} (최소 {:.3}, 최대 {:.3})", "", em, el, eh);
            // 추가 관찰 줄
            let span = (t - last_t) as f64;
            let per_k = 1000.0 / span;
            let hid: usize = hid_kind.iter().sum();
            let (per_tick, w_med) = net.spont_expected();
            println!("{:>9} 자발 | 지금 기대 {:.3}/틱, 입력 가중치 중앙값의 평균 {:.6} | 구간({:.0}틱) 은닉 발화 {:.0}/1k, 기대 자발 {:.0}/1k, 비중 {:.1}% | 구간 생긴 수 {}, 사라진 수 {}",
                "", per_tick, w_med, span, hid as f64 * per_k, exp_acc * per_k, if hid > 0 { exp_acc / hid as f64 * 100.0 } else { 0.0 },
                net.sprouted - last_sprouted, net.pruned - last_pruned);
            println!("{:>9} 번짐 | 불어남 {:.2} (구간 은닉 발화 ÷ 기대 자발) | 한 틱 최대 은닉 발화 {} (은닉의 {:.1}%) | 새 접점 세기 기준의 평균 {:.6}",
                "", if exp_acc > 0.0 { hid as f64 / exp_acc } else { 0.0 }, peak_hid, peak_hid as f64 / hidden_n as f64 * 100.0, net.unit_med_mean());
            println!("{:>9} 자극 밖 | {}틱 | 불어남 {:.2} | 은닉 발화 {:.0}/1k, 기대 자발 {:.0}/1k | 한 틱 최대 은닉 발화 {} (은닉의 {:.1}%)",
                "", out_ticks, if out_exp > 0.0 { out_hid as f64 / out_exp } else { 0.0 },
                if out_ticks > 0 { out_hid as f64 * 1000.0 / out_ticks as f64 } else { 0.0 }, if out_ticks > 0 { out_exp * 1000.0 / out_ticks as f64 } else { 0.0 },
                out_peak, out_peak as f64 / hidden_n as f64 * 100.0);
            println!("{:>9} 조절 | 구간에 뉴런 한 개가 발화한 수: 도파민 {:.2}, 예측 {:.2} | 조절 수준의 기준선 대비 편차 (구간 끝) {:+.3} | 문이 열린 비율: 위 {:.1}%, 아래 {:.1}% | 예측 뉴런의 입력 가중치 합 {:.4} | 도파민 수상돌기 평균 {:+.4}",
                "", (net.da_spikes - last_da) as f64 / da, (net.pred_spikes - last_pred) as f64 / net.pred_count().max(1) as f64, net.mod_deviation(),
                if gate_n > 0 { gate_up as f64 / gate_n as f64 * 100.0 } else { 0.0 }, if gate_n > 0 { gate_down as f64 / gate_n as f64 * 100.0 } else { 0.0 },
                net.slow_input_weights().1.0, if dend_n > 0 { dend_acc / dend_n as f64 } else { 0.0 });
            gate_up = 0; gate_down = 0; gate_n = 0; dend_acc = 0.0; dend_n = 0;
            last_da = net.da_spikes; last_pred = net.pred_spikes;
            let obs = net.hidden_observe();
            for k in 0..5 {
                if kind_n[k] == 0 { continue; }
                let mut r: Vec<f64> = obs.iter().filter(|o| o.0 == k).map(|o| o.1 * 1000.0).collect();
                let mut w: Vec<f64> = obs.iter().filter(|o| o.0 == k).map(|o| o.2).collect();
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                w.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q = |v: &[f64], p: f64| v[((v.len() as f64 * p) as usize).min(v.len() - 1)];
                let n = kind_n[k] as f64;
                println!("{:>9} {} (n={}) | 발화율/1k 분위 10% {:.2}, 50% {:.2}, 90% {:.2} | 입력 가중치 중앙값의 중앙값 {:.6} | 구간 뉴런당 발화 {:.2}/1k, 기대 자발 {:.2}/1k, 비중 {:.1}%",
                    "", KINDS[k], kind_n[k], q(&r, 0.1), q(&r, 0.5), q(&r, 0.9), q(&w, 0.5),
                    hid_kind[k] as f64 * per_k / n, exp_kind_acc[k] * per_k / n, if hid_kind[k] > 0 { exp_kind_acc[k] / hid_kind[k] as f64 * 100.0 } else { 0.0 });
            }
            // 단계 C 관찰 줄
            let dv = net.dev_observe();
            let hn = hidden_n as f64;
            let med = |v: &mut Vec<f64>, p: f64| -> f64 { if v.is_empty() { return f64::NAN; } v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() as f64 * p) as usize).min(v.len() - 1)] };
            println!("{:>9} 접점 | 처음 쌍의 접점 {:.1}/뉴런 (그중 은닉→은닉 {:.1}), 자란 쌍의 접점 {:.1}/뉴런 | 쌍 수/뉴런: 처음 {:.1}, 자란 {:.1} | 처음 쌍에 더해진 접점 누계 {} | 같은 쌍의 중복 시냅스 {} (은닉→은닉 시냅스 칸 {}의 {:.1}%)",
                "", dv.init_contacts as f64 / hn, dv.init_hh as f64 / hn, dv.grown_contacts as f64 / hn, dv.init_pairs as f64 / hn, dv.grown_pairs as f64 / hn, net.added_on_initial,
                dv.dup_syn, dv.hh_syn, if dv.hh_syn > 0 { dv.dup_syn as f64 / dv.hh_syn as f64 * 100.0 } else { 0.0 });
            let by_kind = |now: &[u64; 5], last: &[u64; 5]| -> String { (0..5).filter(|&k| kind_n[k] > 0).map(|k| format!("{} {}", KINDS[k], now[k] - last[k])).collect::<Vec<_>>().join(", ") };
            let drop_sum: u64 = (0..5).map(|k| net.gate_drop[k] - last_drop[k]).sum();
            println!("{:>9} 받기 | 구간에 생긴 접점 (받는 쪽): {} | 받기 문에 걸려 버린 시도: {} (합 {})",
                "", by_kind(&net.grow_made, &last_made), by_kind(&net.gate_drop, &last_drop), drop_sum);
            let mut ages: Vec<f64> = net.prune_ages.drain(..).map(|a| a as f64).collect();
            let age_n = ages.len();
            println!("{:>9} 소멸 | 구간에 사라진 접점 {} 중 수명이 줄어서 사라진 수 {} | 사라진 접점의 나이 중앙값 {:.0}틱 (나이를 아는 {}개, 모르는 {}개)",
                "", net.pruned - last_pruned, net.pruned_short - last_short, med(&mut ages, 0.5), age_n, net.prune_age_unknown);
            net.prune_age_unknown = 0;
            let ratio = |o: &(usize, usize, f64, f64, f64, usize, usize)| if o.4 > 0.0 { o.3 / o.4 } else { f64::NAN };
            let mut inc: Vec<f64> = dv.per.iter().map(|o| (o.5 + o.6) as f64).collect();
            let zero: Vec<&(usize, usize, f64, f64, f64, usize, usize)> = dv.per.iter().filter(|o| o.5 + o.6 == 0).collect();
            let mut zero_ratio: Vec<f64> = zero.iter().map(|o| ratio(o)).filter(|r| !r.is_nan()).collect();
            let lo = dv.per.iter().min_by_key(|o| o.5 + o.6).unwrap();
            let hi = dv.per.iter().max_by_key(|o| o.5 + o.6).unwrap();
            let mut ie: Vec<f64> = dv.per.iter().filter(|o| o.5 > 0).map(|o| o.6 as f64 / o.5 as f64).collect();
            println!("{:>9} 입력 | 은닉에서 오는 접점이 0 인 뉴런 {}개 (느린÷목표 중앙값 {:.2}) | 들어오는 접점 수: 최소 {} (그 뉴런의 느린÷목표 {:.2}), 분위 10% {:.0}, 50% {:.0}, 90% {:.0}, 최대 {} ({}, {}, 느린÷목표 {:.2}) | 억제 ÷ 흥분 접점 수의 중앙값 {:.3}",
                "", zero.len(), med(&mut zero_ratio, 0.5), lo.5 + lo.6, ratio(lo), med(&mut inc, 0.1), med(&mut inc, 0.5), med(&mut inc, 0.9),
                hi.5 + hi.6, KINDS[hi.0], AREAS.get(hi.1).copied().unwrap_or("?"), ratio(hi), med(&mut ie, 0.5));
            let area_line: Vec<String> = (0..6).map(|a| {
                let mut r: Vec<f64> = dv.per.iter().filter(|o| o.0 == 0 && o.1 == a).map(|o| o.3 * 1000.0).collect();
                format!("{} {:.2} (n={})", AREAS[a], med(&mut r, 0.5), r.len())
            }).collect();
            println!("{:>9} 영역 | 추체 발화율/1k 중앙값: {}", "", area_line.join(", "));
            if net.params_ref().act_target.is_some() {
                for k in 0..5 {
                    if kind_n[k] == 0 { continue; }
                    let v: Vec<&(usize, usize, f64, f64, f64, usize, usize)> = dv.per.iter().filter(|o| o.0 == k).collect();
                    let with: Vec<&&(usize, usize, f64, f64, f64, usize, usize)> = v.iter().filter(|o| o.4 > 0.0).collect();
                    let n = v.len() as f64;
                    let (f, sl, b) = (with.iter().filter(|o| o.2 > o.4).count(), with.iter().filter(|o| o.3 > o.4).count(), with.iter().filter(|o| o.2 > o.4 || o.3 > o.4).count());
                    let mut r: Vec<f64> = with.iter().map(|o| o.3 / o.4).collect();
                    println!("{:>9} {} 문 | 발화율이 목표를 넘은 뉴런의 비율: 빠른 {:.1}%, 느린 {:.1}%, 둘 중 하나 {:.1}% | 느린÷목표 분위 10% {:.2}, 50% {:.2}, 90% {:.2} | 목표가 0 인 뉴런 {}개",
                        "", KINDS[k], f as f64 / n * 100.0, sl as f64 / n * 100.0, b as f64 / n * 100.0, med(&mut r, 0.1), med(&mut r, 0.5), med(&mut r, 0.9), v.len() - with.len());
                }
            }
            last_made = net.grow_made; last_drop = net.gate_drop; last_short = net.pruned_short;
            std::io::stdout().flush().ok();
            spikes = 0; food_n = 0; water_n = 0; da_acc = 0; da_n = 0;
            exp_acc = 0.0; exp_kind_acc = [0.0; 5]; hid_kind = [0; 5]; peak_hid = 0;
            out_ticks = 0; out_hid = 0; out_exp = 0.0; out_peak = 0;
            last_t = t; last_sprouted = net.sprouted; last_pruned = net.pruned;
        }
        if let Some((every, path)) = &save {
            if t % every == 0 {
                net.dev_tick = t;
                let file = format!("{}.t{}", path, t);
                net.save(std::path::Path::new(&file));
                println!("{:>9} [저장] {} (경과 {:.0}초)", t, file, t0.elapsed().as_secs_f64());
                saved.push((t, file));
                // 최신 둘과 이정표만 남긴다
                while saved.len() > 2 {
                    let (old_t, old_file) = saved.remove(0);
                    if ![2_000_000u64, 10_000_000, 25_000_000, 50_000_000].contains(&old_t) { std::fs::remove_file(&old_file).ok(); }
                }
                std::io::stdout().flush().ok();
            }
        }
    }
}

/// 자극을 넣고 ticks 틱 진행하며 도파민 뉴런 한 개의 발화 수(20틱 구간별)와 가치 뉴런 발화 수를 잰다. at 틱에 다른 자극이나 보상을 넣을 수 있다
fn run_window(net: &mut Network, word: &str, ticks: u64, second: Option<(u64, &str)>, reward_at: Option<u64>) -> (Vec<f64>, u64, Vec<u32>) {
    let da = net.da_count() as f64;
    net.drive_text(word, 4);
    let (mut bins, mut d0, v0) = (Vec::new(), net.da_spikes, net.val_spikes);
    let mut ids: std::collections::HashSet<u32> = Default::default();
    for t in 0..ticks {
        if let Some((at, w)) = second { if t == at { net.drive_text(w, 4); } }
        if reward_at == Some(t) { net.drive_reward(true, 20); }
        net.step();
        if t < 25 { for &id in &net.tick_ids { if !net.is_input(id as usize) { ids.insert(id); } } }
        if (t + 1) % 20 == 0 { bins.push((net.da_spikes - d0) as f64 / da); d0 = net.da_spikes; }
    }
    let mut v: Vec<u32> = ids.into_iter().collect();
    v.sort_unstable();
    (bins, net.val_spikes - v0, v)
}

/// 2차 조건화 점검
fn order2_test(params: &NetParams, n: usize) {
    println!("{:?}", params);
    println!("\nA = 안녕, B = 고마워. 1단계: A 를 넣고 30틱 뒤 보상 ({n}번).  2단계: B 를 넣고 30틱 뒤 A, 보상 없음 ({n}번).");
    println!("도파민 = 자극이 시작된 뒤 20틱 동안 도파민 뉴런 한 개의 발화 수 (평소 약 0.6). 가치뉴런 = 그 자극에 발화한 가치 뉴런의 발화 수");
    let fmt = |b: &[f64]| b.iter().take(4).map(|x| format!("{:.1}", x)).collect::<Vec<_>>().join(" ");
    for (name, phase1) in [("A 를 보상과 짝지음", true), ("짝지음 없음 (비교)", false)] {
        let mut net = Network::with_params(params.clone());
        net.rest(300);
        let (a0, va0, sa) = run_window(&mut net, "안녕", 300, None, None);
        let (b0, vb0, sb) = run_window(&mut net, "고마워", 300, None, None);
        let only = |a: &[u32], b: &[u32]| -> Vec<u32> { let sb: std::collections::HashSet<u32> = b.iter().cloned().collect(); a.iter().cloned().filter(|x| !sb.contains(x)).collect() };
        let (a_only, b_only) = (only(&sa, &sb), only(&sb, &sa));
        println!("\n[{name}]");
        println!("  학습 전      A 만: 도파민(20틱 구간) {}  가치뉴런 {}   |  B 만: 도파민 {}  가치뉴런 {}", fmt(&a0), va0, fmt(&b0), vb0);
        // 1단계
        for _ in 0..n {
            if phase1 { run_window(&mut net, "안녕", 330, None, Some(30)); } else { run_window(&mut net, "안녕", 330, None, None); }
        }
        let (a1, va1, _) = run_window(&mut net, "안녕", 300, None, None);
        let (b1, vb1, _) = run_window(&mut net, "고마워", 300, None, None);
        println!("  1단계 뒤     A 만: 도파민 {}  가치뉴런 {}   |  B 만: 도파민 {}  가치뉴런 {}", fmt(&a1), va1, fmt(&b1), vb1);
        // 2단계: B → A, 보상 없음. 진행하면서 가치 뉴런으로 가는 입력 가중치(A 쪽 / B 쪽 / 그 밖)와 구간별 반응을 찍는다
        let w0 = net.weights_snapshot();
        let step = (n / 6).max(1);
        let (mut vsum, mut d0sum, mut d2sum, mut k) = (0u64, 0.0, 0.0, 0usize);
        for t in 1..=n {
            let (bins, vs, _) = run_window(&mut net, "고마워", 330, Some((30, "안녕")), None);
            vsum += vs; d0sum += bins[0]; d2sum += bins[2]; k += 1;
            if t % step == 0 {
                let (wa, _) = net.value_input_weights(&a_only);
                let (wb, wo) = net.value_input_weights(&b_only);
                println!("    2단계 {:>5}번: 가치 뉴런 발화 {:>5.1} (B 와 A 합쳐서), 도파민 B 직후 {:.2} / A 직후(40~60틱) {:.2} | 가치 입력 가중치: A 쪽 {:.5}, B 쪽 {:.5}, 그 밖 {:.5}",
                    t, vsum as f64 / k as f64, d0sum / k as f64, d2sum / k as f64, wa, wb, wo);
                vsum = 0; d0sum = 0.0; d2sum = 0.0; k = 0;
            }
        }
        let (a2, va2, _) = run_window(&mut net, "안녕", 300, None, None);
        let (b2, vb2, _) = run_window(&mut net, "고마워", 300, None, None);
        let (wa, _) = net.weight_change_within(&a_only, &w0);
        let (wb, _) = net.weight_change_within(&b_only, &w0);
        println!("  2단계 뒤     A 만: 도파민 {}  가치뉴런 {}   |  B 만: 도파민 {}  가치뉴런 {}", fmt(&a2), va2, fmt(&b2), vb2);
        println!("  2단계 동안의 시냅스 변화:  A 쪽 {:+.1}%,  B 쪽 {:+.1}%", wa * 100.0, wb * 100.0);
    }
}

/// 한 번의 시행: 입력을 넣고 40틱 동안 발화한 출력 뉴런(글자 번호별 발화 여부)을 돌려준다
fn op_trial(net: &mut Network, word: &str) -> Vec<bool> {
    let mut fired = vec![false; net.output_count()];
    net.drive_text(word, 4);
    for _ in 0..40 {
        net.step();
        for &id in &net.tick_ids { if let Some(q) = net.output_seq(id as usize) { fired[q] = true; } }
    }
    fired
}

/// 기능 시험의 시행 안 시간대 (자극을 넣은 뒤의 틱, 끝 포함): 1~5 / 6~10 / 11~20 / 21~40 / 41~60(보상 창) / 61~160 / 161~300(쉼)
const OP_BINS: [usize; 7] = [5, 10, 20, 40, 60, 160, 300];
/// 자극 뒤 처음 몇 틱을 틱마다 따로 볼지
const OP_FINE: usize = 10;
/// 문이 열렸다 / 닫혔다고 세는 편차 (조절 수준 ÷ 기준선 − 1)
const OP_GATE: f64 = 0.25;

/// 기능 시험의 시행 안 관찰값 누적 (출력용. 네트워크 동작에는 관여하지 않는다)
#[derive(Default, Clone)]
struct OpObs {
    n: usize,
    /// 시간대별 도파민 / 가치 / 예측 뉴런 발화 수
    da: [u64; 7], val: [u64; 7], pred: [u64; 7],
    /// 시간대별 출력 뉴런 편차 표본: +OP_GATE 이상 / −OP_GATE 이하 / 표본 수 (뉴런 × 틱)
    hi: [u64; 7], lo: [u64; 7], samp: [u64; 7],
    /// 처음 OP_FINE 틱의 틱별 값: 도파민 / 가치 / 예측 발화 수, 그 틱에 처음 발화한 출력 수, 편차 +OP_GATE 이상 표본, 표본 수
    f_da: [u64; OP_FINE], f_val: [u64; OP_FINE], f_pred: [u64; OP_FINE], f_out: [u64; OP_FINE], f_hi: [u64; OP_FINE], f_samp: [u64; OP_FINE],
    /// 목표 출력이 처음 발화한 틱 (합, 발화한 시행 수), 출력이 처음 발화한 틱 (합, 출력 수), 도파민 뉴런이 처음 발화한 틱 (합, 시행 수)
    t_first: (u64, u64), o_first: (u64, u64), d_first: (u64, u64),
}
impl OpObs {
    fn add(&mut self, o: &OpObs) {
        self.n += o.n;
        for i in 0..7 { self.da[i] += o.da[i]; self.val[i] += o.val[i]; self.pred[i] += o.pred[i]; self.hi[i] += o.hi[i]; self.lo[i] += o.lo[i]; self.samp[i] += o.samp[i]; }
        for i in 0..OP_FINE { self.f_da[i] += o.f_da[i]; self.f_val[i] += o.f_val[i]; self.f_pred[i] += o.f_pred[i]; self.f_out[i] += o.f_out[i]; self.f_hi[i] += o.f_hi[i]; self.f_samp[i] += o.f_samp[i]; }
        self.t_first.0 += o.t_first.0; self.t_first.1 += o.t_first.1;
        self.o_first.0 += o.o_first.0; self.o_first.1 += o.o_first.1;
        self.d_first.0 += o.d_first.0; self.d_first.1 += o.d_first.1;
    }
    /// 누적한 값을 찍는다. 발화 값 = 그 집단의 뉴런 하나가 한 틱에 발화한 비율(%), 문 값 = 출력 뉴런 표본 중 편차가 기준을 넘은 비율(%)
    fn print(&self, label: &str, da_n: usize, val_n: usize, pred_n: usize) {
        if self.n == 0 { println!("      · {label} 0번"); return; }
        let n = self.n as f64;
        let avg = |p: (u64, u64)| if p.1 > 0 { format!("{:.1}틱", p.0 as f64 / p.1 as f64) } else { "없음".to_string() };
        println!("      · {label} {}번:  시행당 출력 수 {:.1}개,  목표 첫 발화 {} ({}번),  출력 첫 발화 평균 {},  도파민 첫 발화 {}",
            self.n, self.o_first.1 as f64 / n, avg(self.t_first), self.t_first.1, avg(self.o_first), avg(self.d_first));
        let width = |i: usize| (OP_BINS[i] - if i == 0 { 0 } else { OP_BINS[i - 1] }) as f64;
        let rate = |c: &[u64; 7], cells: usize| (0..7).map(|i| if cells == 0 { format!("{:>8}", "-") } else { format!("{:>8.2}", c[i] as f64 * 100.0 / (n * width(i) * cells as f64)) }).collect::<String>();
        let share = |c: &[u64; 7]| (0..7).map(|i| if self.samp[i] == 0 { format!("{:>8}", "-") } else { format!("{:>8.2}", c[i] as f64 * 100.0 / self.samp[i] as f64) }).collect::<String>();
        println!("          시간대(틱)    {}", ["1~5", "6~10", "11~20", "21~40", "41~60", "61~160", "161~300"].iter().map(|x| format!("{:>8}", x)).collect::<String>());
        println!("          도파민 %      {}", rate(&self.da, da_n));
        println!("          가치 %        {}", rate(&self.val, val_n));
        println!("          예측 %        {}", rate(&self.pred, pred_n));
        println!("          문 열림 %     {}", share(&self.hi));
        println!("          문 닫힘 %     {}", share(&self.lo));
        let fine = |c: &[u64; OP_FINE], cells: usize| (0..OP_FINE).map(|i| if cells == 0 { format!("{:>7}", "-") } else { format!("{:>7.2}", c[i] as f64 * 100.0 / (n * cells as f64)) }).collect::<String>();
        println!("          틱별           {}", (1..=OP_FINE).map(|t| format!("{:>7}", t)).collect::<String>());
        println!("          틱별 도파민 %  {}", fine(&self.f_da, da_n));
        println!("          틱별 가치 %    {}", fine(&self.f_val, val_n));
        println!("          틱별 예측 %    {}", fine(&self.f_pred, pred_n));
        println!("          틱별 첫 출력   {}", (0..OP_FINE).map(|i| format!("{:>7.2}", self.f_out[i] as f64 / n)).collect::<String>());
        println!("          틱별 문 열림 % {}", (0..OP_FINE).map(|i| if self.f_samp[i] == 0 { format!("{:>7}", "-") } else { format!("{:>7.2}", self.f_hi[i] as f64 * 100.0 / self.f_samp[i] as f64) }).collect::<String>());
    }
}

/// 기능 시험의 한 시행 전체(300틱)를 돌리면서 시간대별 값을 센다. 진행 순서는 op_trial + (보상) + rest(260) 과 같다:
/// 입력 4틱, 40틱 동안 출력 읽기, 목표가 나왔고 reward 이면 보상 입력 20틱, 나머지 260틱.
/// 반환: 출력별 발화 여부, 보상을 줬는지, 이 시행의 관찰값
fn op_trial_observed(net: &mut Network, word: &str, target: usize, reward: bool) -> (Vec<bool>, bool, OpObs) {
    let mut fired = vec![false; net.output_count()];
    let mut o = OpObs { n: 1, ..Default::default() };
    let mut rewarded = false;
    let mut d_seen = false;
    net.drive_text(word, 4);
    for t in 1..=300usize {
        if t == 41 && reward && fired[target] { net.drive_reward(true, 20); rewarded = true; }
        let (d0, v0, p0) = (net.da_spikes, net.val_spikes, net.pred_spikes);
        net.tick();
        let (d, v, p) = (net.da_spikes - d0, net.val_spikes - v0, net.pred_spikes - p0);
        let b = OP_BINS.iter().position(|&e| t <= e).unwrap();
        o.da[b] += d; o.val[b] += v; o.pred[b] += p;
        if t <= OP_FINE { o.f_da[t - 1] += d; o.f_val[t - 1] += v; o.f_pred[t - 1] += p; }
        if d > 0 && !d_seen { d_seen = true; o.d_first = (t as u64, 1); }
        if t <= 40 {
            for &id in &net.tick_ids {
                if let Some(q) = net.output_seq(id as usize) {
                    if !fired[q] {
                        fired[q] = true;
                        o.o_first.0 += t as u64; o.o_first.1 += 1;
                        if q == target { o.t_first = (t as u64, 1); }
                        if t <= OP_FINE { o.f_out[t - 1] += 1; }
                    }
                }
            }
        }
        // 출력 뉴런의 편차: 처음 OP_FINE 틱은 매 틱, 그 뒤로는 5틱마다 표본
        if t <= OP_FINE || t % 5 == 0 {
            let devs = net.deviations_of(1);
            let hi = devs.iter().filter(|&&x| x >= OP_GATE).count() as u64;
            let lo = devs.iter().filter(|&&x| x <= -OP_GATE).count() as u64;
            o.hi[b] += hi; o.lo[b] += lo; o.samp[b] += devs.len() as u64;
            if t <= OP_FINE { o.f_hi[t - 1] += hi; o.f_samp[t - 1] += devs.len() as u64; }
        }
    }
    (fired, rewarded, o)
}

/// 뉴런 집단별 편차(조절 수준 ÷ 기준선 − 1) 분포를 한 줄씩 찍는다: 은닉, 출력, 가치
fn op_print_deviations(net: &Network, when: &str) {
    println!("  [편차 분포, {when}] 조절 수준 ÷ 기준선 − 1");
    for (name, which) in [("은닉", 0usize), ("출력", 1), ("가치", 2)] {
        let v = net.deviations_of(which);
        if v.is_empty() { println!("      {name}: 조절 신호를 받는 뉴런 없음"); continue; }
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
        let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        println!("      {name} {:>6}개: 평균 {:+.3}, 표준편차 {:.3}, 최소 {:+.3}, 최대 {:+.3}, +{OP_GATE} 이상 {:.2}%, −{OP_GATE} 이하 {:.2}%",
            v.len(), mean, sd, lo, hi, v.iter().filter(|&&x| x >= OP_GATE).count() as f64 * 100.0 / n, v.iter().filter(|&&x| x <= -OP_GATE).count() as f64 * 100.0 / n);
    }
}

/// 기능 시험: 출력만 보고 보상을 줬을 때, 보상받는 출력이 더 자주 나오게 되는지.
/// 한 네트워크 안에서 순서대로: (1) 보상 없이 반복 — 반복만으로 비율이 변하는지, (2) 목표 출력이 나온 시행에만 보상
fn op_test(params: &NetParams, trials: usize) {
    println!("{:?}", params);
    let mut net = Network::with_params(params.clone());
    net.rest(300);
    let nout = net.output_count();
    // 먼저 60번 넣어 출력별로 나오는 비율과, 출력끼리 같이 나오는 정도를 잰다
    let probes = 60usize;
    let mut hist: Vec<Vec<bool>> = Vec::new();
    for _ in 0..probes { hist.push(op_trial(&mut net, "안녕")); net.rest(260); }
    let count: Vec<usize> = (0..nout).map(|q| hist.iter().filter(|f| f[q]).count()).collect();
    let always = count.iter().filter(|&&c| c * 20 >= probes * 19).count();
    let never = count.iter().filter(|&&c| c * 20 <= probes).count();
    let per_trial = hist.iter().map(|f| f.iter().filter(|&&x| x).count()).sum::<usize>() as f64 / probes as f64;
    println!("\n한 시행에 나오는 출력 수 평균 {:.1}개", per_trial);
    println!("입력 A = 안녕. 처음 {probes}번(보상 없음): 출력 {nout}개 중 거의 항상 나옴 {always}개, 거의 안 나옴 {never}개, 그 사이 {}개", nout - always - never);
    // 목표 = 나오는 비율이 50% 에 가장 가까운 출력. 비교 = 그다음으로 가까운 것들 중 목표와 같이 나오는 정도가 가장 낮은 출력
    let mut order: Vec<usize> = (0..nout).collect();
    order.sort_by_key(|&q| (count[q] as i64 * 2 - probes as i64).abs());
    let target = order[0];
    let both = |a: usize, b: usize| hist.iter().filter(|f| f[a] == f[b]).count() as f64 / probes as f64;
    let control = *order[1..6.min(nout)].iter().min_by(|&&a, &&b| both(target, a).partial_cmp(&both(target, b)).unwrap()).unwrap();
    println!("목표 '{}' (처음 {}/{probes}),  비교 '{}' (처음 {}/{probes}),  둘이 같은 결과였던 시행 {:.0}%",
        net.token_of(target), count[target], net.token_of(control), count[control], both(target, control) * 100.0);
    // 비교를 출력 하나에 맡기면 흔들림이 크다: 처음에 25~75% 로 나오던 다른 출력 전부의 평균도 같이 본다
    let others: Vec<usize> = (0..nout).filter(|&q| q != target && count[q] * 4 >= probes && count[q] * 4 <= probes * 3).collect();
    println!("값 = 구간별로 그 출력이 나온 시행의 비율. '다른 출력' = 처음에 25~75% 로 나오던 나머지 {}개의 평균\n", others.len());
    println!("구간마다 아래에 붙는 줄 = 시행 안 시간대별 값 (보상 시행 / 보상 없는 시행으로 나눔). 발화 % = 그 집단 뉴런 하나가 한 틱에 발화한 비율,");
    println!("문 열림·닫힘 % = 출력 뉴런 표본 중 편차가 +{OP_GATE} 이상 / −{OP_GATE} 이하인 비율 (처음 {OP_FINE}틱은 매 틱, 그 뒤 5틱마다 표본). 틱 = 자극을 넣은 뒤의 틱\n");
    let (da_n, val_n, pred_n) = (net.da_count(), net.val_count(), net.pred_count());
    op_print_deviations(&net, "준비 직후");
    let weights = |net: &Network| { let ((vw, vn), (pw, pn)) = net.slow_input_weights(); format!("가치 입력 가중치 합 {:.4} ({}개),  예측 입력 가중치 합 {:.4} ({}개)", vw, vn, pw, pn) };
    println!("  [시작] {}\n", weights(&net));
    let block = (trials / 6).max(1);
    for (name, reward) in [("보상 없이 반복", false), ("목표가 나오면 보상", true)] {
        let (mut t_hit, mut c_hit, mut n, mut rewards) = (0usize, 0usize, 0usize, 0usize);
        let mut o_hit = 0usize;
        let (mut obs_r, mut obs_n) = (OpObs::default(), OpObs::default());
        for t in 1..=trials {
            let (f, rewarded, o) = op_trial_observed(&mut net, "안녕", target, reward);
            if f[target] { t_hit += 1; }
            if f[control] { c_hit += 1; }
            o_hit += others.iter().filter(|&&q| f[q]).count();
            n += 1;
            if rewarded { rewards += 1; obs_r.add(&o); } else { obs_n.add(&o); }
            if t % block == 0 || t == trials {
                println!("  [{name}] {:>6}번까지:  목표 {:>5.1}%   비교 {:>5.1}%   다른 출력 {:>5.1}%   (보상 {}번)", t, t_hit as f64 * 100.0 / n as f64, c_hit as f64 * 100.0 / n as f64,
                    o_hit as f64 * 100.0 / (n * others.len().max(1)) as f64, rewards);
                if reward { obs_r.print("보상 시행", da_n, val_n, pred_n); }
                obs_n.print("보상 없는 시행", da_n, val_n, pred_n);
                println!("      · {}", weights(&net));
                t_hit = 0; c_hit = 0; n = 0; rewards = 0; o_hit = 0;
                obs_r = OpObs::default(); obs_n = OpObs::default();
            }
        }
    }
    println!();
    op_print_deviations(&net, "끝");
}

/// 기능 시험 (입력 여러 개): 입력마다 다른 목표 출력을 정하고, 무작위 순서로 넣으면서 그 입력의 목표가 나온 시행에만 보상.
/// 입력마다 자기 목표는 오르고, 다른 입력의 목표와 나머지 출력은 오르지 않아야 한다
fn op2_test(params: &NetParams, trials: usize) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    println!("{:?}", params);
    let mut net = Network::with_params(params.clone());
    net.rest(300);
    let words: Vec<&str> = if std::env::args().any(|a| a == "--three") { vec!["안녕", "고마워", "사랑해"] } else { vec!["안녕", "고마워"] };
    let (nw, nout, probes) = (words.len(), net.output_count(), 60usize);
    // 입력별로 출력이 나오는 비율
    let mut count = vec![vec![0usize; nout]; nw];
    for _ in 0..probes { for (w, word) in words.iter().enumerate() { let f = op_trial(&mut net, word); for q in 0..nout { if f[q] { count[w][q] += 1; } } net.rest(260); } }
    // 목표: 그 입력에서 50% 에 가깝게 나오는 출력 (서로 겹치지 않게)
    let mut targets: Vec<usize> = Vec::new();
    for w in 0..nw {
        let mut order: Vec<usize> = (0..nout).filter(|q| !targets.contains(q)).collect();
        order.sort_by_key(|&q| (count[w][q] as i64 * 2 - probes as i64).abs());
        targets.push(order[0]);
    }
    let others: Vec<Vec<usize>> = (0..nw).map(|w| (0..nout).filter(|q| !targets.contains(q) && count[w][*q] * 4 >= probes && count[w][*q] * 4 <= probes * 3).collect()).collect();
    println!("\n입력 {nw}개를 무작위 순서로. 값 = 그 입력을 넣었을 때 나온 비율: 자기 목표 / 다른 입력의 목표(평균) / 나머지 출력(처음 25~75% 이던 것들의 평균)");
    for w in 0..nw { println!("  입력 '{}' 의 목표 '{}' (처음 {}/{probes}), 나머지 {}개", words[w], net.token_of(targets[w]), count[w][targets[w]], others[w].len()); }
    let block = (trials / 6).max(1);
    for (name, reward) in [("보상 없이 반복", false), ("자기 목표가 나오면 보상", true)] {
        let mut own = vec![0usize; nw]; let mut cross = vec![0usize; nw]; let mut oth = vec![0usize; nw]; let mut n = vec![0usize; nw]; let mut rew = vec![0usize; nw];
        for t in 1..=trials * nw {
            let w = rng.random_range(0..nw);
            let f = op_trial(&mut net, words[w]);
            n[w] += 1;
            if f[targets[w]] { own[w] += 1; }
            cross[w] += (0..nw).filter(|&v| v != w && f[targets[v]]).count();
            oth[w] += others[w].iter().filter(|&&q| f[q]).count();
            if reward && f[targets[w]] { net.drive_reward(true, 20); rew[w] += 1; }
            net.rest(260);
            if t % (block * nw) == 0 || t == trials * nw {
                let mut line = format!("  [{name}] {:>6}번까지:", t / nw);
                for w in 0..nw {
                    let k = n[w].max(1) as f64;
                    line += &format!("   {}: 목표 {:>5.1}% / 남의 목표 {:>5.1}% / 나머지 {:>5.1}% (보상 {})", words[w], own[w] as f64 * 100.0 / k,
                        cross[w] as f64 * 100.0 / (k * (nw - 1) as f64), oth[w] as f64 * 100.0 / (k * others[w].len().max(1) as f64), rew[w]);
                }
                println!("{line}");
                own.iter_mut().for_each(|x| *x = 0); cross.iter_mut().for_each(|x| *x = 0); oth.iter_mut().for_each(|x| *x = 0); n.iter_mut().for_each(|x| *x = 0); rew.iter_mut().for_each(|x| *x = 0);
            }
        }
    }
}
