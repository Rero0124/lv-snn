//! 초파리 커넥톰 위에서 snn 엔진의 발화 역학을 검증한다.
//!
//!   cargo run --release -- ref   [--trials 10]          기준 모델(Shiu LIF) 실행 + 캐시
//!   cargo run --release -- run   [옵션]                  우리 엔진 1회 실행 + 기준과 비교
//!   cargo run --release -- sweep [옵션]                  w_scale 을 바꿔가며 비교
//!   cargo run --release -- mb    [옵션]                  버섯체 연합학습 테스트 (mb.rs 참고)
//!   cargo run --release -- live  [--trials N] [--no-stdp] [--w-scale F]   전 기제를 켠 채 단맛 자극 반복 — 동작이 유지되는지 (brain.rs)
//!   cargo run --release -- jamo  [--mode global|rpe|none] [--shuffle] [--w-scale F]   자모 과제를 전뇌 배선 위에서 (brain.rs)
//!   cargo run --release -- brain-da [--dan PPL101] [--da-first] [--dan-all] [--learn] [--pn]   전뇌 규모의 도파민 짝지음 (brain.rs)
//!   cargo run --release -- mb-da [--dan PPL101] [--da-first]   도파민 짝지음 실험 (조절 뉴런 검증, mb.rs)
//!   cargo run --release -- mb-jamo [--mode global|rpe|none] [--shuffle]   자모 과제를 버섯체 배선 위에서
//!
//! 옵션: --w-scale F  --noise F  --decay F  --ticks N  --trials N
//!       --adapt (self_threshold 적응 켬)  --fatigue (시냅스 피로 켬)  --full (둘 다 + noise 0.2)
//!       --no-refractory

mod brain;
mod data;
mod mb;
mod ours;
mod reference;
#[path = "../../snn/src/neuron.rs"]
mod neuron;
#[path = "../../snn/src/plasticity.rs"]
mod plasticity;
#[path = "../../snn/src/synapse.rs"]
mod synapse;
#[path = "../../snn/src/tokenizer.rs"]
#[allow(dead_code, unused_imports)]
mod tokenizer;

use data::{Connectome, MN9, SUGAR_GRN};
use rayon::prelude::*;
use std::path::Path;

/// 검증된 구성 (초파리 배선에서 단맛 500회, 두 자극 번갈아 800회 동안 반응이 유지됨):
///   가중치 = 시냅스 수 × 0.0068, 자발 발화 없음
///   흥분성 STDP + iSTDP 를 한 쌍으로, LTD 는 post 발화당 한 번,
///     억제 시냅스에는 일반 STDP 를 겹쳐 걸지 않음
///   변화량과 soft bound 는 뉴런별 지역 값: 자기 입력 가중치의 중앙값(변화량)과 상위 5% 지점(경계)
///   교과서 BCM (LTD × (자기 장기 평균 발화율 ÷ 목표)²), 목표 = 자기 시냅스들의 활성화 빈도 중앙값
///   항상성은 뉴런별 목표 방식(최대 2%), 노이즈는 자기 입력 가중치 중앙값의 1/10
/// 뉴런은 자기 상태와 자기 시냅스만 본다. 전체 분포나 다른 뉴런의 내부 값을 쓰는 곳은 없다 (가중치 배수 0.0068 은 배선 변환 값).
const VALIDATED: &str = "--w-scale 0.0068 --spont-scale 0 --no-inh-stdp --local-scale --local-pct 0.95 \
    --ltd-once --bcm-classic-local 1 --homeo-own 0.02 --noise-local 0.1";

/// 이전 검증 구성 (변화량·경계를 전체 분포에서 정한 전역 배수 방식). 비교용
const VALIDATED_GLOBAL: &str = "--w-scale 0.0068 --spont-scale 0 --no-inh-stdp --stdp-scale 0.07 --istdp-scale 0.07 \
    --bound-scale 0.22 --ltd-once --bcm-classic 0.03 --homeo-own 0.02 --noise-local 0.1";

const DATA_DIR: &str = "data";
const REF_CACHE: &str = "data/ref_sugar.bin";
const REF_MS: f64 = 1000.0;

pub(crate) fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}
pub(crate) fn flag(args: &[String], name: &str) -> bool { args.iter().any(|a| a == name) }

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    // --validated: 초파리 배선에서 반복 자극에도 반응이 유지되는 것으로 확인된 구성 (live 명령용).
    // 뒤에 붙는 값이라 같은 옵션을 앞에서 직접 주면 그 값이 우선한다.
    if args.iter().any(|a| a == "--validated") {
        args.extend(VALIDATED.split_whitespace().map(String::from));
    }
    if args.iter().any(|a| a == "--validated-global") {
        args.extend(VALIDATED_GLOBAL.split_whitespace().map(String::from));
    }
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("run");

    let c = Connectome::load(Path::new(DATA_DIR));
    let index = c.index_of();
    let stim: Vec<usize> = SUGAR_GRN.iter().filter_map(|id| index.get(id).copied()).collect();
    let mn9 = index[&MN9];
    eprintln!("[커넥톰] 뉴런 {}, 연결 {}, 억제성 뉴런 {:.1}%, 자극(단맛 GRN) {}개",
        c.n(), c.out_post.len(),
        100.0 * c.inhibitory.iter().filter(|&&b| b).count() as f64 / c.n() as f64, stim.len());

    match cmd {
        "ref" => {
            let trials: u64 = arg(&args, "--trials").unwrap_or(10);
            let rate = compute_reference(&c, &stim, trials);
            summarize_reference(&c, &rate, &stim, mn9);
            let (_, rest) = reference::run_trial_with_rest(&c, &stim, REF_MS, 500.0, 7);
            println!("  자극을 끈 뒤 500ms 동안의 발화: {rest}");
            // 자극 없이 snn 과 같은 자발 발화(뉴런당 1e-5/ms)만 줬을 때 스스로 불이 붙는지
            let spont: f64 = arg(&args, "--spont").unwrap_or(1e-5);
            let (_, _, windows) = reference::run_full(&c, &[], 0.0, 6000.0, spont, 11);
            println!("  자극 없이 자발 발화 {spont}/ms 만: 500ms 구간별 발화 {:?}", windows);
        }
        "run" | "sweep" => {
            let reference = load_reference(&c, &stim);
            let full = flag(&args, "--full");
            let decay: Option<f64> = arg(&args, "--decay");
            let ms_per_tick = -20.0 * decay.unwrap_or(0.7).ln();
            let cfg = ours::Config {
                w_scale: arg(&args, "--w-scale").unwrap_or(0.005),
                noise: arg(&args, "--noise").unwrap_or(if full { 0.2 } else { 0.0 }),
                adapt_threshold: full || flag(&args, "--adapt"),
                fatigue: full || flag(&args, "--fatigue"),
                decay,
                no_refractory: flag(&args, "--no-refractory"),
                stim_ticks: arg(&args, "--ticks").unwrap_or((REF_MS / ms_per_tick).round() as u64),
                tail_cap: 2000,
            };
            let trials: usize = arg(&args, "--trials").unwrap_or(1);
            eprintln!("[설정] {:?}, 1틱 ≈ {:.1}ms (막 시상수 20ms 기준)", cfg, ms_per_tick);
            if cmd == "run" {
                let r = run_ours(&c, &cfg, &stim, trials);
                report(&c, &reference, &r, &stim, mn9, true);
            } else {
                println!("{:>8} {:>7} {:>9} {:>6} {:>6} {:>6} {:>8} {:>8} {:>7}",
                    "w_scale", "active", "spk/trial", "prec", "recall", "jacc", "spearman", "MN9", "settle");
                for w in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05] {
                    let mut cfg = cfg.clone();
                    cfg.w_scale = w;
                    let r = run_ours(&c, &cfg, &stim, trials);
                    let m = report(&c, &reference, &r, &stim, mn9, false);
                    println!("{:>8.4} {:>7} {:>9.0} {:>6.2} {:>6.2} {:>6.2} {:>8.2} {:>8.1} {:>7}",
                        w, m.active, m.spikes, m.precision, m.recall, m.jaccard, m.spearman, m.mn9,
                        r.settle.map(|t| t.to_string()).unwrap_or("안꺼짐".into()));
                }
            }
        }
        "ref-ignite" => {
            // 우리 엔진에서 지속 활동을 하던 뉴런 집단을 기준 모델에서 100ms 동안 자극한 뒤 끄고 관찰
            let kick: Vec<usize> = std::fs::read_to_string("data/rest_active.txt").expect("data/rest_active.txt 없음")
                .lines().filter_map(|l| l.parse().ok()).collect();
            let (_, rest, windows) = reference::run_full(&c, &kick, 100.0, 3000.0, 0.0, 5);
            println!("기준 모델: 뉴런 {}개를 100ms 자극 후 끔 → 이후 3초간 총 발화 {rest}, 500ms 구간별 {:?}", kick.len(), windows);
        }
        "ref-scan" => {
            // 감각뉴런 그룹별로 기준 모델 반응 크기를 훑는다 (두 번째 자극 후보 고르기)
            let text = std::fs::read_to_string("data/io_neurons.tsv").expect("data/io_neurons.tsv 없음");
            let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
            for line in text.lines().skip(1) {
                let f: Vec<&str> = line.split('\t').collect();
                if f[1] != "in" { continue; }
                let idx: usize = f[0].parse().unwrap();
                match groups.last_mut() {
                    Some((g, v)) if g == f[2] => v.push(idx),
                    _ => groups.push((f[2].to_string(), vec![idx])),
                }
            }
            let sugar_ref = load_reference(&c, &stim);
            let rows: Vec<String> = groups.par_iter().map(|(name, members)| {
                let counts = reference::run_trial(&c, members, REF_MS, 42);
                let active: Vec<usize> = (0..c.n()).filter(|&i| counts[i] >= 1 && !members.contains(&i)).collect();
                let overlap = active.iter().filter(|&&i| sugar_ref[i] >= 1.0).count();
                let total: u64 = counts.iter().map(|&k| k as u64).sum();
                format!("{:<22} 뉴런 {:>3}  활성 {:>6}  총발화 {:>8}  단맛 활성집합과 겹침 {:>4}", name, members.len(), active.len(), total, overlap)
            }).collect();
            for r in rows { println!("{r}"); }
        }
        "mb" => mb::run(&c, &args),
        "mb-jamo" => mb::run_jamo(&c, &args),
        "mb-da" => mb::run_da(&c, &args),
        "brain-da" => brain::run_da(&c, &args),
        "jamo" => brain::run_jamo(&c, &args),
        "live" => brain::run_live(&c, &args, &stim, mn9),
        _ => eprintln!("사용법: ref | run | sweep | mb"),
    }
}

// ── 기준 모델 ──

fn compute_reference(c: &Connectome, stim: &[usize], trials: u64) -> Vec<f32> {
    let t0 = std::time::Instant::now();
    let all: Vec<Vec<u32>> = (0..trials).into_par_iter()
        .map(|t| reference::run_trial(c, stim, REF_MS, 1000 + t)).collect();
    let mut rate = vec![0f32; c.n()];
    for counts in &all {
        for (r, &k) in rate.iter_mut().zip(counts) { *r += k as f32 / trials as f32; }
    }
    let bytes: Vec<u8> = rate.iter().flat_map(|r| r.to_le_bytes()).collect();
    std::fs::write(REF_CACHE, bytes).expect("기준 캐시 저장 실패");
    eprintln!("[기준] {trials}회 시행, {:.1}초, 캐시 {REF_CACHE}", t0.elapsed().as_secs_f64());
    rate
}

/// 임의의 자극 집합에 대한 기준 모델 반응 (5회 평균, data/ref_<name>.bin 에 캐시)
pub(crate) fn reference_named(c: &Connectome, stim: &[usize], name: &str) -> Vec<f32> {
    let path = format!("data/ref_{name}.bin");
    if let Ok(raw) = std::fs::read(&path) {
        if raw.len() == c.n() * 4 {
            return raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        }
    }
    let trials = 5u64;
    let all: Vec<Vec<u32>> = (0..trials).into_par_iter()
        .map(|t| reference::run_trial(c, stim, REF_MS, 2000 + t)).collect();
    let mut rate = vec![0f32; c.n()];
    for counts in &all {
        for (r, &k) in rate.iter_mut().zip(counts) { *r += k as f32 / trials as f32; }
    }
    let bytes: Vec<u8> = rate.iter().flat_map(|r| r.to_le_bytes()).collect();
    std::fs::write(&path, bytes).ok();
    rate
}

pub(crate) fn load_reference(c: &Connectome, stim: &[usize]) -> Vec<f32> {
    match std::fs::read(REF_CACHE) {
        Ok(raw) if raw.len() == c.n() * 4 => raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        _ => compute_reference(c, stim, 10),
    }
}

fn summarize_reference(c: &Connectome, rate: &[f32], stim: &[usize], mn9: usize) {
    let active = rate.iter().filter(|&&r| r >= 1.0).count();
    let total: f32 = rate.iter().sum();
    let stim_rate: f32 = stim.iter().map(|&i| rate[i]).sum::<f32>() / stim.len() as f32;
    println!("기준 모델 (단맛 GRN 150Hz, 1초)");
    println!("  활성 뉴런(≥1Hz): {active}   총 발화/시행: {total:.0}   자극 뉴런 평균: {stim_rate:.0}Hz   MN9: {:.1}Hz", rate[mn9]);
    let mut top: Vec<usize> = (0..c.n()).filter(|i| !stim.contains(i)).collect();
    top.sort_by(|&a, &b| rate[b].partial_cmp(&rate[a]).unwrap());
    println!("  상위 10 (자극 제외): {}", top[..10].iter()
        .map(|&i| format!("{:.0}Hz", rate[i])).collect::<Vec<_>>().join(" "));
}

// ── 우리 엔진 ──

struct OursResult {
    /// 시행 평균 발화 횟수
    mean: Vec<f64>,
    per_tick: Vec<u32>,
    settle: Option<u64>,
    stim_fire_ratio: f64,
    secs: f64,
    ticks: u64,
}

fn run_ours(c: &Connectome, cfg: &ours::Config, stim: &[usize], trials: usize) -> OursResult {
    let t0 = std::time::Instant::now();
    let mut mean = vec![0f64; c.n()];
    let mut last = None;
    for _ in 0..trials {
        let mut neurons = ours::build(c, cfg, stim);
        let out = ours::run(&mut neurons, cfg, stim);
        for (m, &k) in mean.iter_mut().zip(&out.counts) { *m += k as f64 / trials as f64; }
        last = Some(out);
    }
    let out = last.unwrap();
    OursResult {
        mean, per_tick: out.per_tick, settle: out.settle_ticks,
        stim_fire_ratio: out.stim_fire_ratio, secs: t0.elapsed().as_secs_f64(), ticks: cfg.stim_ticks,
    }
}

struct Metrics { active: usize, spikes: f64, precision: f64, recall: f64, jaccard: f64, spearman: f64, mn9: f64 }

fn report(c: &Connectome, reference: &[f32], r: &OursResult, stim: &[usize], mn9: usize, verbose: bool) -> Metrics {
    let n = c.n();
    let mut is_stim = vec![false; n];
    for &i in stim { is_stim[i] = true; }
    let ref_active: Vec<bool> = (0..n).map(|i| !is_stim[i] && reference[i] >= 1.0).collect();
    let our_active: Vec<bool> = (0..n).map(|i| !is_stim[i] && r.mean[i] >= 1.0).collect();
    let both = (0..n).filter(|&i| ref_active[i] && our_active[i]).count();
    let n_ref = ref_active.iter().filter(|&&b| b).count();
    let n_our = our_active.iter().filter(|&&b| b).count();
    let union: Vec<usize> = (0..n).filter(|&i| ref_active[i] || our_active[i]).collect();
    let xs: Vec<f64> = union.iter().map(|&i| reference[i] as f64).collect();
    let ys: Vec<f64> = union.iter().map(|&i| r.mean[i]).collect();
    let m = Metrics {
        active: n_our,
        spikes: r.mean.iter().sum(),
        precision: if n_our > 0 { both as f64 / n_our as f64 } else { 0.0 },
        recall: if n_ref > 0 { both as f64 / n_ref as f64 } else { 0.0 },
        jaccard: if union.is_empty() { 0.0 } else { both as f64 / union.len() as f64 },
        spearman: spearman(&xs, &ys),
        mn9: r.mean[mn9],
    };
    if verbose {
        println!("우리 엔진 ({}틱, {:.1}초 소요)", r.ticks, r.secs);
        println!("  활성 뉴런(시행당 ≥1발화): {}  (기준 {})", n_our, n_ref);
        println!("  총 발화/시행: {:.0}  (기준 {:.0})", m.spikes, reference.iter().sum::<f32>());
        println!("  활성 집합 일치: 정밀도 {:.2}, 재현율 {:.2}, Jaccard {:.2}", m.precision, m.recall, m.jaccard);
        println!("  발화율 순위상관(Spearman, 합집합 {}개): {:.2}", union.len(), m.spearman);
        println!("  MN9: {:.1}발화/시행  (기준 {:.1}Hz)", m.mn9, reference[mn9]);
        println!("  자극 뉴런 발화 비율: {:.0}% (틱당)", 100.0 * r.stim_fire_ratio);
        println!("  자극 종료 후 소멸: {}", r.settle.map(|t| format!("{t}틱"))
            .unwrap_or("상한(2000틱)까지 안 꺼짐".into()));
        let q = r.per_tick.len() / 4;
        if q > 0 {
            let avg = |s: &[u32]| s.iter().sum::<u32>() as f64 / s.len() as f64;
            println!("  틱당 발화 수 (구간 평균): {:.0} → {:.0} → {:.0} → {:.0}",
                avg(&r.per_tick[..q]), avg(&r.per_tick[q..2 * q]), avg(&r.per_tick[2 * q..3 * q]), avg(&r.per_tick[3 * q..]));
        }
    }
    m
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let rank = (i + j) as f64 / 2.0;
        for k in i..=j { out[idx[k]] = rank; }
        i = j + 1;
    }
    out
}

pub(crate) fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 { return 0.0; }
    let (rx, ry) = (ranks(x), ranks(y));
    let mean = (x.len() - 1) as f64 / 2.0;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in rx.iter().zip(&ry) {
        sxy += (a - mean) * (b - mean);
        sxx += (a - mean) * (a - mean);
        syy += (b - mean) * (b - mean);
    }
    if sxx == 0.0 || syy == 0.0 { 0.0 } else { sxy / (sxx * syy).sqrt() }
}
