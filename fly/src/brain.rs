//! 자모 과제를 초파리 전뇌 배선 위에서.
//!
//! 전뇌 138,639 뉴런 · 1,509만 연결 전체가 은닉층이다 (snn 의 감정/이성/기억/해마 자리에 실제 뇌 배선).
//!   입력: 자모 토큰마다 감각뉴런 타입 하나 (같은 타입 뉴런 최대 20개) — data/io_neurons.tsv
//!   출력: 자모 토큰마다 하행/운동 뉴런 하나 (들어오는 연결이 많은 순)
//!   가중치: 시냅스 수 × w_scale (최대 1.0). 학습 규칙과 틱 루프는 snn 과 동일 (mb.rs 의 jamo_task)
//!   --shuffle: 뉴런별 연결 수·가중치·흥분/억제는 그대로 두고 연결 대상만 무작위로 바꾼 대조군

use crate::data::Connectome;
use crate::neuron::{Neuron, NeuronId};
use crate::synapse::Synapse;
use crate::{arg, flag, plasticity, tokenizer};
use rayon::prelude::*;

pub fn run_jamo(c: &Connectome, args: &[String]) {
    let w_scale: f64 = arg(args, "--w-scale").unwrap_or(0.007);
    let shuffle = flag(args, "--shuffle");
    let seed: u64 = arg(args, "--seed").unwrap_or(1);
    let n = c.n();

    // 입출력 후보
    let text = std::fs::read_to_string("data/io_neurons.tsv")
        .expect("data/io_neurons.tsv 없음 — scripts/prepare_io.py 를 먼저 실행");
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut out_candidates: Vec<usize> = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let idx: usize = f[0].parse().unwrap();
        if f[1] == "in" {
            match groups.last_mut() {
                Some((g, v)) if g == f[2] => v.push(idx),
                _ => groups.push((f[2].to_string(), vec![idx])),
            }
        } else {
            out_candidates.push(idx);
        }
    }

    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };

    // 뉴런 생성
    let mut neurons: Vec<Neuron> = (0..n).into_par_iter().map(|i| {
        let mut nr = if c.inhibitory[i] { Neuron::new_inhibitory(i as NeuronId, 0.0, 0.0) } else { Neuron::new(i as NeuronId, 0.0, 0.0) };
        nr.synapses = (c.out_start[i]..c.out_start[i + 1])
            .map(|k| Synapse::new(c.out_post[k], (c.out_count[k].unsigned_abs() as f64 * w_scale).min(1.0)))
            .collect();
        nr
    }).collect();
    if shuffle {
        for nr in neurons.iter_mut() {
            for syn in &mut nr.synapses { syn.target = (next() % n as u64) as NeuronId; }
        }
    }
    plasticity::rebuild_incoming(&mut neurons);

    // 토큰 배정
    let n_tokens = tokenizer::all_tokens().len();
    assert!(groups.len() >= n_tokens, "감각뉴런 그룹이 토큰 수보다 적음");
    for i in (1..groups.len()).rev() { groups.swap(i, (next() % (i as u64 + 1)) as usize); }
    let token_inputs: Vec<Vec<usize>> = groups[..n_tokens].iter().map(|g| g.1.clone()).collect();
    let is_input: std::collections::HashSet<usize> = token_inputs.iter().flatten().copied().collect();
    out_candidates.retain(|i| !is_input.contains(i));
    out_candidates.sort_by_key(|&i| std::cmp::Reverse(neurons[i].incoming.len()));
    let out_neuron: Vec<usize> = out_candidates[..n_tokens].to_vec();

    eprintln!("[전뇌] w_scale {w_scale}, 입력 그룹 예: {}, 출력 뉴런 들어오는 연결 {}~{}개",
        groups[..4].iter().map(|g| format!("{}({})", g.0, g.1.len())).collect::<Vec<_>>().join(" "),
        neurons[*out_neuron.last().unwrap()].incoming.len(), neurons[out_neuron[0]].incoming.len());
    let label = if shuffle { "전뇌 배선 — 무작위 재배선 대조군" } else { "초파리 전뇌 배선" };
    crate::mb::jamo_task(neurons, token_inputs, out_neuron, label, args);
}

// ── 정해진 배선 위에서 snn 기제를 전부 켜고 도는지 ──
//
// 초파리의 원래 입출력을 그대로 쓴다: 단맛 감각뉴런 자극 → 주둥이 운동뉴런 MN9.
// 뉴런 모델 + 시냅스 피로 + 자가 임계값 + 노이즈 + 항상성 + STDP + iSTDP 를 모두 켜고
// 같은 자극을 반복했을 때, 처음에 맞던 반응(기준 모델과의 일치)이 유지되는지 본다.
// sprout / prune / 수면은 포함하지 않는다 (2D 좌표·영역 구조에 묶여 있어 커넥톰에 그대로 적용 불가).

pub fn run_live(c: &Connectome, args: &[String], stim: &[usize], mn9: usize) {
    let w_scale: f64 = arg(args, "--w-scale").unwrap_or(0.007);
    let trials: usize = arg(args, "--trials").unwrap_or(200);
    let report: usize = arg(args, "--report").unwrap_or(20);
    let stim_ticks: u64 = arg(args, "--ticks").unwrap_or(140);
    let rest_ticks: u64 = arg(args, "--rest").unwrap_or(60);
    let noise: f64 = arg(args, "--noise").unwrap_or(0.2);
    let stdp = !flag(args, "--no-stdp");
    // 기제별 끄기 (원인 분리용)
    let fatigue = !flag(args, "--no-fatigue");
    let adapt = !flag(args, "--no-adapt");
    let homeo = !flag(args, "--no-homeo");
    let free_sensors = flag(args, "--free-sensors");
    // 지역 배수에서 "강한 입력"으로 삼을 자기 입력 분포의 지점 (기본 0.99)
    let local_pct: f64 = arg(args, "--local-pct").unwrap_or(0.99);
    // 항상성을 뉴런별 목표 방식으로 (값 = 한 번에 조정하는 최대 비율, 예: 0.02)
    let homeo_own: Option<f64> = arg(args, "--homeo-own");
    // 노이즈를 뉴런별로: 자기에게 들어오는 시냅스 가중치 중앙값 × 이 값 (예: 0.1)
    let noise_local: Option<f64> = arg(args, "--noise-local");
    // 발화 빈도 적응 시험: 발화마다 임계값 +sfa_inc, 매 틱 기본값(0.5) 쪽으로 sfa_decay 배 복귀
    let sfa_inc: f64 = arg(args, "--sfa-inc").unwrap_or(0.0);
    let sfa_decay: f64 = arg(args, "--sfa-decay").unwrap_or(0.98);
    // --snn-path: snn 본체가 쓰는 학습 경로를 켠다 (snn/src/network.rs 의 apply_dopamine_options 에서 --short-tau 를 줬을 때와 같은 설정).
    //   도파민 문: STDP·iSTDP 는 흔적만 남기고, 받는 뉴런의 조절 수준이 자기 기준선에서 ±25% 넘게 벗어난 동안에만 흔적이 가중치로 굳는다
    //             (mod_gate = --da-gain, 기본 0.05 / mod_baseline = true / gate_base = 0)
    //   일시적 변화: STDP 의 변화는 시냅스의 일시적인 부분(w_short)에 바로 반영되고 --short-tau(기본 2000)틱에 걸쳐 사라진다
    //   조절 신호는 --dan-mod 와 같이 세포 분류가 DAN 인 뉴런의 출력이다 (이 옵션이 --dan-mod 를 함께 켠다).
    //   snn 과 다른 점: 도파민 뉴런이 스스로 발화하지 않고(배선에서 들어오는 입력으로만 발화), 보상 입력이 없고,
    //   도파민 뉴런이 닿지 않는 뉴런은 문이 열리지 않아 그 입력 시냅스의 변화가 모두 일시적이다.
    // --stdp-rest: 휴지 구간에도 STDP 와 굳히기를 돌린다 (snn 은 모든 틱에 돌린다. live 의 기본은 자극 중에만)
    let snn_path = flag(args, "--snn-path");
    let da_gain: f64 = arg(args, "--da-gain").unwrap_or(0.05);
    let short_tau: f64 = arg(args, "--short-tau").unwrap_or(2000.0);
    let short_d = (-1.0 / short_tau.max(1.0)).exp();
    let stdp_rest = flag(args, "--stdp-rest");
    // ── 냄새 + 설탕 + 도파민 구동 (초파리의 냄새–설탕 짝지음. 옵션을 주지 않으면 아무것도 달라지지 않는다) ──
    // --drive-dan 이름들: 그 유형의 도파민 뉴런을 시행마다 직접 구동한다 (전위를 문턱 위로. 광유전학적 활성화에 해당).
    //     쉼표로 여러 개. 숫자 없는 이름(PAM, PPL)은 그 글자로 시작하는 유형 전부. 예: PAM01,PAM02,PAM08 / PAM
    // --drive-at 시작 길이: 자극이 시작된 뒤 몇 틱째부터 몇 틱 동안 구동하는지 (기본 20 40)
    // --drive-after N: 처음 N회는 구동하지 않는다 (기본 15). 그동안 평소 발화만으로 기준선이 잡힌다 (기준선 = 조절 신호를 받기 시작한 뒤 300~2300틱의 평균)
    // --dan-tonic 값: 구동하는 도파민 뉴런에 주는 평소 발화 (기본 0.02 = snn 본체의 도파민 뉴런과 같은 값. 뉴런마다 0.85~1.15배). 0 이면 주지 않는다.
    //     평소 발화가 없으면 기준선이 "구동 구간 + 조용한 구간"의 평균으로 잡혀서, 구동이 끝난 뒤 내내 문이 아래쪽으로 열린다
    // --dan-tonic-all: 평소 발화를 도파민 뉴런 전부에 준다 (대조용)
    // --odor 비율: 버섯체 세포(KC)의 그 비율을 무작위로 골라(실행 내내 같은 세포) 자극 구간 동안 5틱마다 발화시킨다 (brain-da 의 --kc 와 같은 방식).
    //     냄새로 켠 세포는 활성 수·정밀도·재현율을 셀 때 자극 뉴런처럼 뺀다
    let drive_dan: Option<String> = arg(args, "--drive-dan");
    let drive_at: (u64, u64) = match args.iter().position(|a| a == "--drive-at") {
        Some(p) => (args.get(p + 1).and_then(|x| x.parse().ok()).unwrap_or(20), args.get(p + 2).and_then(|x| x.parse().ok()).unwrap_or(40)),
        None => (20, 40),
    };
    let drive_after: usize = arg(args, "--drive-after").unwrap_or(15);
    let dan_tonic: f64 = arg(args, "--dan-tonic").unwrap_or(0.02);
    let dan_tonic_all = flag(args, "--dan-tonic-all");
    let odor_frac: Option<f64> = arg(args, "--odor");
    // --mod-decay 값: 받는 뉴런의 조절 수준이 틱마다 줄어드는 배수 (기본은 뉴런의 기본값 0.98 = snn 본체와 같다). 1 에 가까울수록 조절 수준이 천천히 변한다.
    //     주의: 값을 주면 snn 본체 경로의 값과 달라진다
    let mod_decay: Option<f64> = arg(args, "--mod-decay");
    // --dan-no-input: 도파민 뉴런으로 들어오는 배선의 연결을 모두 끊는다 (brain-da 의 같은 옵션을 옮긴 것). 도파민 뉴런은 평소 발화와 직접 구동으로만 발화한다.
    //     배선 그대로면 도파민 뉴런이 자극 때 들어오는 입력으로 더 발화해서, 조절 수준이 자극 때 오르고 쉴 때 내려간다 (기준선은 둘을 섞은 평균으로 잡힌다).
    //     실험자가 도파민 뉴런을 켜는 조건에 해당하고, snn 본체(도파민 뉴런이 보상 입력과 예측 뉴런에서만 받는다)와 같은 조건이 된다.
    //     단순화: 버섯체 출력 뉴런 → 도파민의 되먹임과 감각 → 도파민의 길이 빠진다
    let dan_no_input = flag(args, "--dan-no-input");
    // --extras: 냄새·구동 옵션 없이도 관찰 줄을 찍는다 (읽기만)
    let extras = drive_dan.is_some() || odor_frac.is_some() || args.iter().any(|a| a == "--extras");
    let mut opt = plasticity::PlasticityOptions {
        stdp_on_inhibitory: !flag(args, "--no-inh-stdp"),
        ltd_scale: arg(args, "--ltd-scale").unwrap_or(1.0),
        stdp_scale: arg(args, "--stdp-scale").unwrap_or(1.0),
        istdp_scale: arg(args, "--istdp-scale").unwrap_or(1.0),
        istdp_target: arg(args, "--istdp-target"),
            bound: match arg::<String>(args, "--bound").as_deref() {
                Some("hard") => plasticity::Bound::Hard,
                Some("prop") => plasticity::Bound::Proportional,
                _ => plasticity::Bound::Soft,
            },
            ltd_once_per_post: flag(args, "--ltd-once"),
            bcm: !flag(args, "--no-bcm"),
            bcm_rate_target: arg(args, "--bcm-rate"),
            bcm_rate_relative: arg(args, "--bcm-rel"),
            bcm_local: arg(args, "--bcm-local"),
            bcm_classic: arg(args, "--bcm-classic"),
            bcm_classic_local: arg(args, "--bcm-classic-local"),
            istdp_bound: if arg::<String>(args, "--istdp-bound").as_deref() == Some("hard") { plasticity::Bound::Hard } else { plasticity::Bound::Soft },
            istdp_sigma: arg(args, "--istdp-sigma").unwrap_or(0.4),
            istdp_bound_scale: arg(args, "--istdp-bound-scale").or(arg(args, "--bound-scale")).unwrap_or(1.0),
            bound_scale: arg(args, "--bound-scale").unwrap_or(1.0),
            local_scale: flag(args, "--local-scale"),
            mod_eta: arg(args, "--mod-eta").unwrap_or(0.0),
            mod_forward_sign: arg(args, "--mod-forward").unwrap_or(-1.0),
            mod_backward_sign: arg(args, "--mod-backward").unwrap_or(1.0),
            mod_stdp_gain: arg(args, "--mod-stdp").unwrap_or(0.0),
            mod_gate: arg(args, "--mod-gate"),
            gate_base: arg(args, "--gate-base").unwrap_or(0.0), mod_baseline: false, pred_eta: 0.0, short_decay: None, val_eta: 0.0,
    };
    if snn_path {
        opt.mod_gate = Some(da_gain);
        opt.gate_base = 0.0;
        opt.mod_baseline = true;
        opt.short_decay = Some(short_d);
    }
    let n = c.n();
    // 자극 목록: A = 단맛(기본). --stim2 GROUP 을 주면 그 감각뉴런 그룹을 B 로 추가해 A, B 를 번갈아 준다
    let mut stims: Vec<Vec<usize>> = vec![stim.to_vec()];
    let mut refs: Vec<Vec<f32>> = vec![crate::load_reference(c, stim)];
    let mut stim_names: Vec<String> = vec!["단맛".into()];
    if let Some(group) = arg::<String>(args, "--stim2") {
        let text = std::fs::read_to_string("data/io_neurons.tsv").expect("data/io_neurons.tsv 없음");
        let members: Vec<usize> = text.lines().skip(1).filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            if f[1] == "in" && f[2] == group { f[0].parse().ok() } else { None }
        }).collect();
        assert!(!members.is_empty(), "--stim2 그룹을 찾지 못함");
        refs.push(crate::reference_named(c, &members, &group));
        stims.push(members);
        stim_names.push(group);
    }
    let all_sensors: Vec<usize> = stims.iter().flatten().copied().collect();
    let reference = refs[0].clone();
    let decay: Option<f64> = arg(args, "--decay");

    let mut neurons: Vec<Neuron> = (0..n).into_par_iter().map(|i| {
        let mut nr = if c.inhibitory[i] { Neuron::new_inhibitory(i as NeuronId, 0.0, 0.0) } else { Neuron::new(i as NeuronId, 0.0, 0.0) };
        nr.synapses = (c.out_start[i]..c.out_start[i + 1])
            .map(|k| Synapse::new(c.out_post[k], (c.out_count[k].unsigned_abs() as f64 * w_scale).min(1.0)))
            .collect();
        nr
    }).collect();
    for &i in &all_sensors { neurons[i].skip_refractory = true; }
    if let Some(d) = decay { for nr in neurons.iter_mut() { nr.decay_rate = d; } }
    // 억제성 뉴런의 출력 이득 (억제를 흥분 대비 얼마나 세게 할지)
    let inh_gain: f64 = arg(args, "--inh-gain").unwrap_or(1.0);
    for nr in neurons.iter_mut() { if nr.inhibitory { for s in &mut nr.synapses { s.weight = (s.weight * inh_gain).min(1.0); } } }
    // --dan-mod: 도파민 뉴런(세포 분류 DAN, data/mb_neurons.tsv)을 흥분성 대신 조절 뉴런으로 취급한다
    let mut dan_idx: Vec<usize> = Vec::new();
    let mut dan_types: Vec<String> = Vec::new();
    if flag(args, "--dan-mod") || snn_path {
        let text = std::fs::read_to_string("data/mb_neurons.tsv").expect("data/mb_neurons.tsv 없음");
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f[1] == "DAN" { dan_idx.push(f[0].parse().unwrap()); dan_types.push(f[2].chars().take(3).collect()); }
        }
        for &i in &dan_idx { neurons[i].inhibitory = false; neurons[i].modulatory = true; }
        let active_ref = dan_idx.iter().filter(|&&i| refs[0][i] >= 1.0).count();
        println!("도파민 뉴런 {}개를 조절 뉴런으로 취급. 기준 모델에서 단맛에 반응하는 도파민 뉴런: {}개", dan_idx.len(), active_ref);
    }
    // 조절 신호가 닿는 범위 (관찰용): 도파민 뉴런의 출력이 닿는 뉴런 수
    let mut dan_target = vec![false; n];
    if snn_path {
        let mut by_type: std::collections::BTreeMap<String, (usize, usize, Vec<bool>)> = Default::default();
        let mut syn_total = 0usize;
        for (&i, ty) in dan_idx.iter().zip(&dan_types) {
            let e = by_type.entry(ty.clone()).or_insert((0, 0, vec![false; n]));
            e.0 += 1;
            for s in &neurons[i].synapses { dan_target[s.target as usize] = true; e.2[s.target as usize] = true; e.1 += 1; syn_total += 1; }
        }
        let cnt = |v: &[bool]| v.iter().filter(|&&b| b).count();
        let n_t = cnt(&dan_target);
        let ref_hit = (0..n).filter(|&i| dan_target[i] && refs[0][i] >= 1.0 && !stims[0].contains(&i)).count();
        let ref_all = (0..n).filter(|&i| refs[0][i] >= 1.0 && !stims[0].contains(&i)).count();
        println!("[snn 경로] 도파민 문 (세기 {da_gain}, 기준선 방식) + 일시적 변화 (시간 상수 {short_tau}틱), 휴지 구간 STDP·굳히기 {}",
            if stdp_rest { "켬" } else { "끔 (자극 중에만)" });
        println!("[snn 경로] 도파민 뉴런의 출력이 닿는 뉴런: {n_t}개 (전체 {n} 의 {:.1}%), 도파민 뉴런의 출력 연결 {syn_total}개. 그중 다른 도파민 뉴런 {}개",
            100.0 * n_t as f64 / n as f64, dan_idx.iter().filter(|&&i| dan_target[i]).count());
        println!("[snn 경로] 종류별: {}", by_type.iter().map(|(k, (a, b, v))| format!("{k} {a}개 → 표적 {}개 (연결 {b})", cnt(v))).collect::<Vec<_>>().join(", "));
        println!("[snn 경로] 기준 모델에서 단맛에 활성인 뉴런 {ref_all}개 중 도파민이 닿는 것: {ref_hit}개. MN9 에 닿는가: {}. 단맛 감각뉴런 {}개 중 닿는 것: {}개",
            if dan_target[mn9] { "예" } else { "아니오" }, stims[0].len(), stims[0].iter().filter(|&&i| dan_target[i]).count());
        // 일시적인 부분의 감쇠 배수: 조절 뉴런이 아닌 모든 뉴런의 출력 시냅스에 (snn 은 도파민·보상·예측 뉴런을 뺀 모든 시냅스에 넣는다)
        for nr in neurons.iter_mut() { if !nr.modulatory { for s in &mut nr.synapses { s.short_decay = short_d; } } }
    }
    // 자발 발화 배수 (기준 모델은 자발 발화가 없다)
    let spont: f64 = arg(args, "--spont-scale").unwrap_or(1.0);
    for nr in neurons.iter_mut() { nr.spont_scale = spont; }
    if dan_no_input {
        let text = std::fs::read_to_string("data/mb_neurons.tsv").expect("data/mb_neurons.tsv 없음");
        let mut is_dan = vec![false; n];
        for line in text.lines().skip(1) { let f: Vec<&str> = line.split('\t').collect(); if f[1] == "DAN" { is_dan[f[0].parse::<usize>().unwrap()] = true; } }
        let mut cut = 0usize;
        for nr in neurons.iter_mut() { let before = nr.synapses.len(); nr.synapses.retain(|s| !is_dan[s.target as usize]); cut += before - nr.synapses.len(); }
        println!("[도파민 입력 끊음] 도파민 뉴런 {}개로 들어오는 연결 {cut}개를 끊었다 (--dan-no-input)", is_dan.iter().filter(|&&b| b).count());
    }
    // 냄새 + 도파민 구동의 준비 (옵션을 줄 때만)
    let mut drive_idx: Vec<usize> = Vec::new();   // 시행마다 구동하는 도파민 뉴런
    let mut tonic_idx: Vec<usize> = Vec::new();   // 평소 발화를 주는 도파민 뉴런
    let mut odor_idx: Vec<usize> = Vec::new();    // 냄새로 켜는 버섯체 세포
    let mut mbon_idx: Vec<usize> = Vec::new();
    let (mut is_kc, mut is_mbon) = (Vec::new(), Vec::new());
    let mut is_apl: Vec<bool> = Vec::new();
    // --rest-dump 앞머리: 쉬는 구간 발화가 뛴 보고 시행에서, 쉬는 구간에 발화한 뉴런의 번호를 "앞머리.t<시행>.txt" 로 남긴다 (관찰용)
    let rest_dump: Option<String> = arg(args, "--rest-dump");
    // --watch 파일: 그 파일에 적힌 번호의 뉴런 무리(예: 다른 실행에서 번진 뉴런)의 감응도·항상성 목표·느린 발화율 평균을 시행마다 찍는다 (관찰용).
    // 이 실행에서 처음 번진 보고 시행의 무리도 그때부터 함께 찍는다.
    let watch_given: Vec<usize> = arg::<String>(args, "--watch").map(|f| std::fs::read_to_string(&f).expect("--watch 파일 없음")
        .split_whitespace().filter_map(|x| x.parse::<usize>().ok()).filter(|&i| i < n).collect()).unwrap_or_default();
    let mut watch_own: Vec<usize> = Vec::new();
    if !watch_given.is_empty() { println!("[지켜봄] --watch 무리 {}개", watch_given.len()); }
    if extras {
        let text = std::fs::read_to_string("data/mb_neurons.tsv").expect("data/mb_neurons.tsv 없음");
        let names: Vec<String> = drive_dan.as_deref().unwrap_or("").split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
        let wanted = |ty: &str| names.iter().any(|nm| ty == nm || (!nm.chars().any(|ch| ch.is_ascii_digit()) && ty.starts_with(nm.as_str())));
        let (mut kc_all, mut all_dan) = (Vec::new(), Vec::new());
        let mut driven_types: std::collections::BTreeMap<String, usize> = Default::default();
        is_kc = vec![false; n]; is_mbon = vec![false; n]; is_apl = vec![false; n];
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let idx: usize = f[0].parse().unwrap();
            match f[1] {
                "KC" => { kc_all.push(idx); is_kc[idx] = true; }
                "MBON" => { mbon_idx.push(idx); is_mbon[idx] = true; }
                "APL" => { is_apl[idx] = true; }
                "DAN" => { all_dan.push(idx); if wanted(f[2]) { drive_idx.push(idx); *driven_types.entry(f[2].to_string()).or_default() += 1; } }
                _ => {}
            }
        }
        if drive_dan.is_some() { assert!(!drive_idx.is_empty(), "--drive-dan 에 맞는 도파민 뉴런 유형이 없음"); }
        let seed: u64 = arg(args, "--seed").unwrap_or(1);
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        if let Some(frac) = odor_frac {
            let want = ((kc_all.len() as f64 * frac) as usize).min(kc_all.len());
            let mut picked = vec![false; n];
            while odor_idx.len() < want {
                let k = kc_all[(next() % kc_all.len() as u64) as usize];
                if !picked[k] { picked[k] = true; odor_idx.push(k); }
            }
        }
        if drive_dan.is_some() && dan_tonic > 0.0 {
            tonic_idx = if dan_tonic_all { all_dan.clone() } else { drive_idx.clone() };
            // snn 본체의 도파민 뉴런과 같은 설정 (snn/src/network.rs): 느린 감쇠(0.98) + 지속 입력으로 스스로 발화, 불응기 3/3, 자발 발화 없음.
            // 뉴런마다 주기와 시작 위상이 달라 같은 틱에 몰려 발화하지 않는다
            for &i in &tonic_idx {
                let nr = &mut neurons[i];
                nr.decay_rate = 0.98;
                nr.tonic = dan_tonic * (0.85 + 0.3 * (next() % 10_000) as f64 / 10_000.0);
                nr.potential = 0.5 * (next() % 10_000) as f64 / 10_000.0;
                nr.abs_refractory = 3;
                nr.rel_refractory = 3;
                nr.slow_adapt = false;
                nr.spont_scale = 0.0;
            }
        }
        if let Some(d) = mod_decay { for nr in neurons.iter_mut() { nr.mod_decay = d; } }
        {
            // 뉴런 하나가 듣는 도파민 뉴런 수 (배선에서 센다. 관찰용): 평소 발화를 주는 것만 / 도파민 뉴런 전부
            let dist = |src: &[usize]| -> String {
                let mut cnt = vec![0u32; n];
                for &d in src { let mut seen: Vec<u32> = neurons[d].synapses.iter().map(|s| s.target).collect(); seen.sort_unstable(); seen.dedup(); for t in seen { cnt[t as usize] += 1; } }
                let mut v: Vec<u32> = cnt.into_iter().filter(|&x| x > 0).collect();
                if v.is_empty() { return "없음".into(); }
                v.sort_unstable();
                let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
                let bins = [(1u32, 1u32), (2, 2), (3, 5), (6, 10), (11, 20), (21, u32::MAX)];
                format!("뉴런 {}개, 최소 {} / 10% {} / 중앙 {} / 90% {} / 최대 {} | 1개 {}, 2개 {}, 3~5개 {}, 6~10개 {}, 11~20개 {}, 21개 이상 {}", v.len(), v[0], q(0.1), q(0.5), q(0.9), v[v.len() - 1],
                    v.iter().filter(|&&x| x >= bins[0].0 && x <= bins[0].1).count(), v.iter().filter(|&&x| x == 2).count(), v.iter().filter(|&&x| x >= 3 && x <= 5).count(),
                    v.iter().filter(|&&x| x >= 6 && x <= 10).count(), v.iter().filter(|&&x| x >= 11 && x <= 20).count(), v.iter().filter(|&&x| x >= 21).count())
            };
            println!("[냄새·도파민 구동] 뉴런 하나가 듣는 도파민 뉴런 수 — 평소 발화를 주는 것만: {}", dist(&tonic_idx));
            println!("[냄새·도파민 구동] 뉴런 하나가 듣는 도파민 뉴런 수 — 도파민 뉴런 전부: {}", dist(&all_dan));
            println!("[냄새·도파민 구동] 조절 수준의 감쇠 배수 {}", match mod_decay { Some(d) => format!("{d} (옵션으로 바꿈. snn 본체는 0.98)"), None => "0.98 (기본)".into() });
        }
        println!("[냄새·도파민 구동] 구동하는 도파민 뉴런 {}개 ({}), 자극 시작 뒤 {}틱째부터 {}틱 동안, {}회째 시행 뒤부터 | 평소 발화 {} (뉴런 {}개{}) | 냄새: 버섯체 세포 {}개 중 {}개 (5틱마다) | 버섯체 출력 뉴런 {}개{}",
            drive_idx.len(), driven_types.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "), drive_at.0, drive_at.1, drive_after,
            if tonic_idx.is_empty() { "없음".to_string() } else { format!("{dan_tonic}") }, tonic_idx.len(), if dan_tonic_all { ", 도파민 뉴런 전부" } else { "" },
            kc_all.len(), odor_idx.len(), mbon_idx.len(),
            if drive_dan.is_some() && !(snn_path || flag(args, "--dan-mod")) { " | 주의: --snn-path 가 없어 도파민 뉴런이 조절 뉴런이 아니라 흥분성으로 돈다" } else { "" });
    }
    plasticity::rebuild_incoming(&mut neurons);
    // 교과서 BCM 은 장기 평균을 목표율에서 시작시킨다 (0 에서 시작하면 초반에 LTD 가 전혀 안 걸림)
    if let Some(rho0) = opt.bcm_classic { for nr in neurons.iter_mut() { nr.rate_avg = rho0; } }
    let local_noise = |neurons: &[Neuron], k: f64| -> Vec<f64> {
        neurons.par_iter().map(|nr| {
            let mut w: Vec<f64> = nr.incoming.iter().map(|&(p, s)| neurons[p].synapses[s].weight).collect();
            if w.is_empty() { return 0.0; }
            let mid = w.len() / 2;
            let (_, med, _) = w.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            *med * k
        }).collect()
    };
    if opt.local_scale { plasticity::refresh_local_scales(&mut neurons, local_pct); }
    if flag(args, "--scales-report") {
        // 뉴런별 지역 배수의 분포: 전체 / 각 자극의 기준 활성 뉴런. 전역 방식의 값은 변화량 0.07, 경계 0.22
        let pct = |v: &mut Vec<f64>, q: f64| -> f64 { if v.is_empty() { return 0.0; } v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * q) as usize] };
        let show = |name: &str, idx: Vec<usize>| {
            let mut amp: Vec<f64> = idx.iter().map(|&i| neurons[i].w_ref_med / 0.2).filter(|&x| x > 0.0).collect();
            let mut bnd: Vec<f64> = idx.iter().map(|&i| neurons[i].w_ref_max).filter(|&x| x > 0.0).collect();
            let mut deg: Vec<f64> = idx.iter().map(|&i| neurons[i].incoming.len() as f64).collect();
            println!("{name:<22} 뉴런 {:>6} | 변화량 배수 10%/50%/90%: {:.3} / {:.3} / {:.3} | 경계 배수 10%/50%/90%: {:.3} / {:.3} / {:.3} | 입력 수 중앙값 {:.0}",
                idx.len(), pct(&mut amp, 0.1), pct(&mut amp, 0.5), pct(&mut amp, 0.9), pct(&mut bnd, 0.1), pct(&mut bnd, 0.5), pct(&mut bnd, 0.9), pct(&mut deg, 0.5));
        };
        show("전체", (0..n).collect());
        for (k, name) in stim_names.iter().enumerate() {
            show(&format!("{name} 기준 활성"), (0..n).filter(|&i| refs[k][i] >= 1.0).collect());
        }
        return;
    }
    let mut noise_amp: Vec<f64> = match noise_local { Some(k) => local_noise(&neurons, k), None => vec![noise; n] };
    let initial: Vec<Vec<f32>> = neurons.iter().map(|nr| nr.synapses.iter().map(|s| s.weight as f32).collect()).collect();
    let mut is_stim = vec![false; n];
    for &i in &all_sensors { is_stim[i] = true; }
    for &i in &odor_idx { is_stim[i] = true; }
    // 문의 상태를 지켜볼 뉴런: 도파민 뉴런의 출력이 닿는 뉴런 (관찰용)
    let gate_watch: Vec<usize> = if extras && snn_path { (0..n).filter(|&i| dan_target[i]).collect() } else { Vec::new() };
    // 관찰용 누적 (지난 보고 뒤): [구동 중, 자극 중 구동 밖, 휴지, 구동 직전 20틱, 휴지의 마지막 100틱] × (틱 수, 위쪽으로 열린 뉴런 수의 합, 아래쪽)
    //   뒤의 둘은 앞의 둘째·셋째에 이미 든 틱을 따로 한 번 더 센 것이다 (구동이 올린 조절 수준의 꼬리가 다음 구동 전에 닫히는지 보는 용도)
    let mut gate_acc = [[0.0f64; 3]; 5];
    // 구동하는 유형을 듣는 뉴런만 따로: 구동 중 (틱 수, 위쪽, 아래쪽)
    let mut gate_drv = [0.0f64; 3];
    let mut hears_driven = vec![false; if extras { n } else { 0 }];
    for &d in &drive_idx { for s in &neurons[d].synapses { hears_driven[s.target as usize] = true; } }
    let n_hears = gate_watch.iter().filter(|&&i| hears_driven[i]).count();
    // 구동 유형을 듣는 뉴런을 받는 쪽으로 가른다: 버섯체 세포 / 버섯체 출력 뉴런 / APL / 그 밖(허브)
    let hub_idx: Vec<usize> = if extras { (0..n).filter(|&i| hears_driven[i] && !is_kc[i] && !is_mbon[i] && !is_apl[i] && !neurons[i].modulatory).collect() } else { Vec::new() };
    if extras && !drive_idx.is_empty() {
        println!("[냄새·도파민 구동] 구동 유형을 듣는 뉴런 {}개 = 버섯체 세포 {} + 버섯체 출력 뉴런 {} + APL {} + 도파민 뉴런 {} + 그 밖 {}",
            (0..n).filter(|&i| hears_driven[i]).count(), (0..n).filter(|&i| hears_driven[i] && is_kc[i]).count(), (0..n).filter(|&i| hears_driven[i] && is_mbon[i]).count(),
            (0..n).filter(|&i| hears_driven[i] && is_apl[i]).count(), (0..n).filter(|&i| hears_driven[i] && neurons[i].modulatory).count(), hub_idx.len());
    }
    // 굳은 양의 처음부터의 부호 있는 누계: [버섯체 세포→출력, 단맛 사이, 그 밖] 과 듣는 뉴런을 받는 쪽으로 가른 것 [보내는 쪽 버섯체 세포 / 아님] × [받는 쪽 버섯체 세포 / APL / 그 밖]
    let mut cum_grp = [0.0f64; 3];
    let mut cum_h = [[0.0f64; 3]; 2];
    // (시행 수, 발화한 도파민 뉴런, 그중 구동 대상, 구동 중 한 번이라도 문이 열린 뉴런, 그중 그 시행에 발화한 뉴런, 버섯체 출력 뉴런 발화, 발화한 냄새 세포)
    let mut ex_acc = [0.0f64; 7];
    // 평소 발화를 주는 도파민 뉴런의 발화 수: (자극 구간 틱 수, 발화 수, 휴지 구간 틱 수, 발화 수) — 구동 중인 틱은 빼고 센다
    let mut dan_rate = [0.0f64; 4];
    let mut is_tonic = vec![false; if extras { n } else { 0 }];
    for &i in &tonic_idx { is_tonic[i] = true; }
    let mut opened_at = vec![0u32; if extras { n } else { 0 }];
    let mut prev_w: Vec<Vec<f32>> = if extras { initial.clone() } else { Vec::new() };
    let all_dan_idx: Vec<usize> = if extras { (0..n).filter(|&i| neurons[i].modulatory).collect() } else { Vec::new() };
    let ref_actives: Vec<Vec<bool>> = refs.iter()
        .map(|r| (0..n).map(|i| !is_stim[i] && r[i] >= 1.0).collect()).collect();
    let n_ref = ref_actives[0].iter().filter(|&&b| b).count();
    for (k, name) in stim_names.iter().enumerate().skip(1) {
        println!("자극 {name}: 뉴런 {}, 기준 모델 활성 {}", stims[k].len(), ref_actives[k].iter().filter(|&&b| b).count());
    }

    println!("[전 기제 가동] w_scale {w_scale}, noise {noise}, 피로 {fatigue}, 임계값적응 {adapt}, 항상성 {homeo}, STDP+iSTDP {}, 자극 {stim_ticks}틱 + 휴지 {rest_ticks}틱 × {trials}회",
        if stdp { "켬" } else { "끔" });
    println!("기준 모델: 활성 뉴런 {n_ref}, MN9 {:.0}Hz", reference[mn9]);
    println!("{:>5} {:>6} {:>7} {:>6} {:>6} {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>5}",
        "시행", "MN9", "활성", "정밀도", "재현율", "순위상관", "총발화", "휴지발화", "평균w", "강화%", "약화%", "초");

    // --spont-off-after N: N번째 시행 뒤 자발 발화를 끈다 (지속 활동이 스스로 유지되는지 확인)
    let spont_off_after: usize = arg(args, "--spont-off-after").unwrap_or(usize::MAX);
    let classes: Vec<String> = std::fs::read_to_string("data/superclass.tsv")
        .map(|t| t.lines().map(|l| l.to_string()).collect()).unwrap_or_default();
    let mut rest_active = vec![false; n];
    let mut first_counts: Vec<Option<Vec<u32>>> = vec![None; stims.len()];
    // 창 평균 (관찰용): 보고 간격 사이의 모든 시행에서 모은 값. 자극별로 따로 모은다.
    //   [시행 수, MN9, 활성 수, 총발화, 휴지발화, 정밀도, 재현율, 시행 1 의 활성 집합과의 겹침]
    let mut win: Vec<[f64; 8]> = vec![[0.0; 8]; stims.len()];
    let mut first_active: Vec<Option<Vec<bool>>> = vec![None; stims.len()];
    let t0 = std::time::Instant::now();
    let mut tick = 0u64;
    for trial in 1..=trials {
        if trial == spont_off_after + 1 {
            for nr in neurons.iter_mut() { nr.spont_scale = 0.0; }
            println!("  -- 자발 발화 끔 --");
        }
        rest_active.iter_mut().for_each(|b| *b = false);
        let cur = (trial - 1) % stims.len();
        let stim: &[usize] = &stims[cur];
        let reference: &[f32] = &refs[cur];
        let ref_active: &[bool] = &ref_actives[cur];
        let n_ref = ref_active.iter().filter(|&&b| b).count();
        let mut counts = vec![0u32; n];
        let (mut total, mut rest_total) = (0u64, 0u64);
        let mut tick_counts: Vec<u32> = if extras { vec![0; (stim_ticks + rest_ticks) as usize] } else { Vec::new() };
        for t in 0..stim_ticks + rest_ticks {
            tick += 1;
            let stimulating = t < stim_ticks;
            let driving = stimulating && !drive_idx.is_empty() && trial > drive_after && t >= drive_at.0 && t < drive_at.0 + drive_at.1;
            // 평소 발화를 주는 도파민 뉴런은 임계값 적응과 항상성의 대상이 아니다 (평소 발화율이 흘러가면 고정된 기준선에서 벗어나 문이 저절로 열린다)
            for &i in &tonic_idx { neurons[i].self_threshold = 0.5; neurons[i].excitability = 1.0; }
            if stimulating && !odor_idx.is_empty() && t % 5 == 0 {
                for &i in &odor_idx { neurons[i].potential = 1.0; neurons[i].self_threshold = 0.5; }
            }
            if driving { for &d in &drive_idx { neurons[d].potential = 1.0; neurons[d].self_threshold = 0.5; } }
            if stimulating {
                // 입력 뉴런은 감각기: 임계값 적응과 항상성(감응도)의 대상이 아니다.
                // 감응도를 그대로 두면 자주 발화하는 입력일수록 1만 틱마다 ×0.95 씩 깎여서
                // 7번째(7만 틱)부터 자극(1.0 → decay 후 0.7)이 임계값을 못 넘는다.
                // 지금 주는 자극이 아닌 감각뉴런은 조용히 둔다
                for &i in &all_sensors { neurons[i].potential = 0.0; }
                for &i in stim {
                    neurons[i].potential = 1.0;
                    neurons[i].self_threshold = 0.5;
                    neurons[i].excitability = 1.0;
                }
            } else if !free_sensors {
                // snn 의 입력 뉴런은 들어오는 시냅스가 없어 자극이 없으면 발화하지 않는다.
                // 초파리 감각뉴런은 서로 시냅스가 있어서, 그대로 두면 불응기 없는 자극 뉴런끼리
                // 고리를 이뤄 휴지 구간에도 계속 발화한다 → 자극이 없을 때는 전위를 0 으로 고정
                for &i in &all_sensors { neurons[i].potential = 0.0; }
            }
            let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = neurons.par_iter_mut().with_min_len(2048).filter_map(|nr| {
                nr.decay();
                if !adapt && sfa_inc == 0.0 { nr.self_threshold = 0.5; }
                if sfa_inc > 0.0 { nr.self_threshold = 0.5 + (nr.self_threshold - 0.5) * sfa_decay; }
                let before = nr.self_threshold;
                let out = nr.try_fire(tick, 0.5, noise_amp[nr.id as usize], stimulating);
                if sfa_inc > 0.0 {
                    // try_fire 내부의 ±0.001 조정은 무시하고 발화 시에만 sfa_inc 를 더한다 (상한 없음)
                    nr.self_threshold = if out.is_some() { before + sfa_inc } else { before };
                }
                if out.is_some() && !fatigue {
                    for s in &mut nr.synapses { s.fatigue = 0.0; }
                }
                out.map(|d| (nr.id, d))
            }).collect();
            let mut spikes = Vec::with_capacity(fired.len());
            for (nid, deliveries) in &fired {
                spikes.push(*nid);
                if stimulating { counts[*nid as usize] += 1; } else { rest_active[*nid as usize] = true; }
                let pre = *nid as usize;
                let is_mod = neurons[pre].modulatory;
                for (k, &(target, w)) in deliveries.iter().enumerate() {
                    let t = target as usize;
                    if is_mod {
                        // 조절 뉴런: 전위는 올리지 않고 조절 수준만 올린다
                        neurons[t].receive_mod(w.abs());
                        plasticity::on_modulator(&mut neurons, t, w.abs(), tick, &opt);
                    } else {
                        neurons[t].receive(w);
                        if neurons[t].mod_level > 0.001 { plasticity::on_pre_under_modulator(&mut neurons, pre, k, &opt); }
                    }
                }
            }
            if stimulating { total += spikes.len() as u64; } else { rest_total += spikes.len() as u64; }
            if extras { tick_counts[t as usize] = spikes.len() as u32; }
            if !tonic_idx.is_empty() && !driving {
                let k = if stimulating { 0 } else { 2 };
                dan_rate[k] += 1.0; dan_rate[k + 1] += spikes.iter().filter(|&&id| is_tonic[id as usize]).count() as f64;
            }
            if !gate_watch.is_empty() {
                // 문의 상태 (읽기만 한다): 이 틱의 굳히기가 보는 값과 같다
                let period = if driving { 0 } else if stimulating { 1 } else { 2 };
                let (mut up, mut down, mut dup, mut ddown) = (0usize, 0usize, 0usize, 0usize);
                for &i in &gate_watch {
                    let x = &neurons[i];
                    if x.mod_base > 1e-6 {
                        let r = x.mod_level / x.mod_base - 1.0;
                        if r >= plasticity::MOD_DEADBAND { up += 1; if driving { opened_at[i] = trial as u32; if hears_driven[i] { dup += 1; } } }
                        else if r <= -plasticity::MOD_DEADBAND { down += 1; if driving { opened_at[i] = trial as u32; if hears_driven[i] { ddown += 1; } } }
                    }
                }
                gate_acc[period][0] += 1.0; gate_acc[period][1] += up as f64; gate_acc[period][2] += down as f64;
                if driving { gate_drv[0] += 1.0; gate_drv[1] += dup as f64; gate_drv[2] += ddown as f64; }
                let extra = if stimulating && !drive_idx.is_empty() && trial > drive_after && t < drive_at.0 && t + 20 >= drive_at.0 { Some(3) }
                    else if !stimulating && t + 100 >= stim_ticks + rest_ticks { Some(4) } else { None };
                if let Some(e) = extra { gate_acc[e][0] += 1.0; gate_acc[e][1] += up as f64; gate_acc[e][2] += down as f64; }
            }
            // snn 과 같이 자극 중에만 STDP (idle 틱에는 없음)
            if stdp && (stimulating || stdp_rest) { plasticity::stdp_step(&mut neurons, &spikes, tick, &opt); }
            if homeo && tick % 10_000 == 0 {
                match homeo_own {
                    Some(eta) => neurons.par_iter_mut().for_each(|nr| nr.homeostasis_own(10_000.0, eta, 0.1)),
                    None => neurons.par_iter_mut().for_each(|nr| nr.homeostasis()),
                }
                // 뉴런별 노이즈는 학습으로 바뀐 가중치를 따라가도록 같은 주기로 다시 계산
                if let Some(k) = noise_local { noise_amp = local_noise(&neurons, k); }
                if opt.local_scale { plasticity::refresh_local_scales(&mut neurons, local_pct); }
            }
            if total > 20_000_000 { break; }
        }
        let cycle = (trial - 1) / stims.len() + 1;
        if extras {
            // 시행마다 한 줄: 쉬는 구간 발화가 어느 시행에 켜지고 꺼지는지 (관찰용)
            {
                let (mut hi, mut mid, mut lo, mut mx, mut sum) = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
                for nr in neurons.iter() { let e = nr.excitability; sum += e; if e > mx { mx = e; } if e >= 1.4999 { hi += 1; } else if e > 1.2 { mid += 1; } if e < 0.8 { lo += 1; } }
                for (nm, g) in [("준 무리", &watch_given), ("이 실행에서 처음 번진 무리", &watch_own)] {
                    if g.is_empty() { continue; }
                    let m = g.len() as f64;
                    let (mut e, mut sp, mut ra, mut rf, mut th, mut hi2, mut fired) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
                    for &i in g.iter() { let nr = &neurons[i]; e += nr.excitability; sp += nr.set_point; ra += nr.rate_avg; rf += nr.rate_fast; th += nr.self_threshold; if nr.excitability >= 1.4999 { hi2 += 1; } if rest_active[i] { fired += 1; } }
                    let (ea, spa, raa) = (neurons.iter().map(|x| x.excitability).sum::<f64>(), neurons.iter().map(|x| x.set_point).sum::<f64>(), neurons.iter().map(|x| x.rate_avg).sum::<f64>());
                    let o = (n as f64 - m).max(1.0);
                    println!("  [지켜봄] {trial} {nm} {}개: 감응도 {:.4} 항상성 목표 {:.6} 느린 발화율 {:.6} 빠른 발화율 {:.6} 자가 임계 {:.4} 1.5에 닿은 수 {hi2} 쉬는 구간에 발화한 수 {fired} | 나머지: 감응도 {:.4} 항상성 목표 {:.6} 느린 발화율 {:.6}",
                        g.len(), e / m, sp / m, ra / m, rf / m, th / m, (ea - e) / o, (spa - sp) / o, (raa - ra) / o);
                }
                println!("  [시행] {trial} 틱 {tick} 자극발화 {total} 휴지발화 {rest_total} MN9 {} | 감응도: 평균 {:.4} 최대 {:.3} 1.5에 닿은 뉴런 {hi} 1.2~1.5 {mid} 0.8 아래 {lo}", counts[mn9], sum / n as f64, mx);
            }
            ex_acc[0] += 1.0;
            ex_acc[1] += all_dan_idx.iter().filter(|&&i| counts[i] > 0).count() as f64;
            ex_acc[2] += drive_idx.iter().filter(|&&i| counts[i] > 0).count() as f64;
            let opened: Vec<usize> = gate_watch.iter().copied().filter(|&i| opened_at[i] == trial as u32).collect();
            ex_acc[3] += opened.len() as f64;
            ex_acc[4] += opened.iter().filter(|&&i| counts[i] > 0).count() as f64;
            ex_acc[5] += mbon_idx.iter().map(|&i| counts[i] as f64).sum::<f64>();
            ex_acc[6] += odor_idx.iter().filter(|&&i| counts[i] > 0).count() as f64;
        }
        {
            // 창 평균에 이번 시행을 더한다 (출력만을 위한 계산)
            let act: Vec<bool> = (0..n).map(|i| !is_stim[i] && counts[i] >= 1).collect();
            let n_act = act.iter().filter(|&&b| b).count();
            let both = (0..n).filter(|&i| ref_active[i] && act[i]).count();
            let first = first_active[cur].get_or_insert_with(|| act.clone());
            let inter = (0..n).filter(|&i| first[i] && act[i]).count();
            let uni = (0..n).filter(|&i| first[i] || act[i]).count();
            let w = &mut win[cur];
            w[0] += 1.0; w[1] += counts[mn9] as f64; w[2] += n_act as f64; w[3] += total as f64; w[4] += rest_total as f64;
            w[5] += if n_act > 0 { both as f64 / n_act as f64 } else { 0.0 };
            w[6] += both as f64 / n_ref as f64;
            w[7] += if uni > 0 { inter as f64 / uni as f64 } else { 0.0 };
        }
        if cycle == 1 || cycle % report == 0 {
            let our_active: Vec<bool> = (0..n).map(|i| !is_stim[i] && counts[i] >= 1).collect();
            let both = (0..n).filter(|&i| ref_active[i] && our_active[i]).count();
            let n_our = our_active.iter().filter(|&&b| b).count();
            let union: Vec<usize> = (0..n).filter(|&i| ref_active[i] || our_active[i]).collect();
            let xs: Vec<f64> = union.iter().map(|&i| reference[i] as f64).collect();
            let ys: Vec<f64> = union.iter().map(|&i| counts[i] as f64).collect();
            let (mut sum, mut cnt, mut up, mut down) = (0.0f64, 0usize, 0usize, 0usize);
            for (nr, init) in neurons.iter().zip(&initial) {
                for (s, &w0) in nr.synapses.iter().zip(init) {
                    sum += s.weight; cnt += 1;
                    let d = s.weight - w0 as f64;
                    if d > 0.01 { up += 1; } else if d < -0.01 { down += 1; }
                }
            }
            if flag(args, "--trace") {
                // 경로 추적: 자극 뉴런이 내보내는 시냅스 / MN9 로 들어오는 시냅스의 가중치 합이 처음 대비 몇 배인가
                let ratio = |pairs: &mut dyn Iterator<Item = (f64, f64)>| -> (f64, f64) {
                    let (mut now, mut init) = (0.0, 0.0);
                    for (a, b) in pairs { now += a; init += b; }
                    (now, if init > 0.0 { now / init } else { 0.0 })
                };
                let (so, sr) = ratio(&mut stim.iter().flat_map(|&i|
                    neurons[i].synapses.iter().zip(&initial[i]).map(|(s, &w0)| (s.weight, w0 as f64))));
                // 자극 시냅스 중 강한 것(초기 0.3 이상)만
                let (_, ssr) = ratio(&mut stim.iter().flat_map(|&i|
                    neurons[i].synapses.iter().zip(&initial[i]).filter(|(_, w0)| **w0 >= 0.3).map(|(s, &w0)| (s.weight, w0 as f64))));
                let inc = &neurons[mn9].incoming;
                let (_, me) = ratio(&mut inc.iter().filter(|(p, _)| !neurons[*p].inhibitory)
                    .map(|&(p, k)| (neurons[p].synapses[k].weight, initial[p][k] as f64)));
                let (_, mi) = ratio(&mut inc.iter().filter(|(p, _)| neurons[*p].inhibitory)
                    .map(|&(p, k)| (neurons[p].synapses[k].weight, initial[p][k] as f64)));
                let mn9_pre_active = inc.iter().filter(|(p, _)| counts[*p] > 0).count();
                // 자극 뉴런의 직접 하류(초기 가중치 0.1 이상) 중 발화한 수
                let direct: std::collections::HashSet<usize> = stim.iter().flat_map(|&i|
                    neurons[i].synapses.iter().zip(&initial[i]).filter(|(_, w0)| **w0 >= 0.1).map(|(s, _)| s.target as usize)).collect();
                let direct_active = direct.iter().filter(|&&t| counts[t] > 0).count();
                println!("      [추적] 자극출력 합 {so:.1} (처음의 {sr:.2}배, 강한 것만 {ssr:.2}배) | 직접 하류 {direct_active}/{} 발화 | MN9 입력: 흥분 {me:.2}배, 억제 {mi:.2}배, 발화한 상류 {mn9_pre_active}/{} | MN9 임계값 {:.3}",
                    direct.len(), inc.len(), neurons[mn9].self_threshold);
            }
            println!("{:>5} {:>6} {:>7} {:>6.2} {:>6.2} {:>8.2} {:>8} {:>9} {:>8.4} {:>8.3} {:>8.3} {:>5.0}",
                trial, counts[mn9], n_our,
                if n_our > 0 { both as f64 / n_our as f64 } else { 0.0 },
                both as f64 / n_ref as f64, crate::spearman(&xs, &ys), total, rest_total,
                sum / cnt as f64, 100.0 * up as f64 / cnt as f64, 100.0 * down as f64 / cnt as f64,
                t0.elapsed().as_secs_f64());
            if !dan_idx.is_empty() {
                println!("        도파민 뉴런 중 이번 시행에 발화한 수: {}", dan_idx.iter().filter(|&&i| counts[i] > 0).count());
            }
            {
                // 자기 첫 반응과 얼마나 같은가 (기준 모델과 별개로, 학습이 반응을 얼마나 바꿨는지)
                let first = first_counts[cur].get_or_insert_with(|| counts.clone());
                let a: Vec<bool> = (0..n).map(|i| !is_stim[i] && first[i] >= 1).collect();
                let inter = (0..n).filter(|&i| a[i] && our_active[i]).count();
                let uni: Vec<usize> = (0..n).filter(|&i| a[i] || our_active[i]).collect();
                let fx: Vec<f64> = uni.iter().map(|&i| first[i] as f64).collect();
                let fy: Vec<f64> = uni.iter().map(|&i| counts[i] as f64).collect();
                println!("        ↑ 자극 {} (주기 {cycle}) | 첫 반응 대비: 활성집합 겹침 {:.2}, 순위상관 {:.2}",
                    stim_names[cur], if uni.is_empty() { 0.0 } else { inter as f64 / uni.len() as f64 }, crate::spearman(&fx, &fy));
            }
            if snn_path {
                // snn 경로의 상태 (관찰용, 읽기만 한다). 일시적인 부분은 마지막으로 맞춘 틱에서 지금까지의 감쇠를 계산해서 본다
                let (mut cnt, mut sh_n, mut el_n, mut eff_up, mut eff_down, mut sh_abs) = (0usize, 0usize, 0usize, 0usize, 0usize, 0.0f64);
                let (mut cnt_t, mut up_t, mut down_t) = (0usize, 0usize, 0usize);
                for (nr, init) in neurons.iter().zip(&initial) {
                    for (s, &w0) in nr.synapses.iter().zip(init) {
                        cnt += 1;
                        let ws = if s.w_short != 0.0 { s.w_short * s.short_decay.powi(tick.saturating_sub(s.t).min(1_000_000) as i32) } else { 0.0 };
                        if ws.abs() > 0.01 { sh_n += 1; }
                        sh_abs += ws.abs();
                        if s.elig != 0.0 { el_n += 1; }
                        let d = s.weight + ws - w0 as f64;
                        if d > 0.01 { eff_up += 1; } else if d < -0.01 { eff_down += 1; }
                        if dan_target[s.target as usize] {
                            cnt_t += 1;
                            let dp = s.weight - w0 as f64;
                            if dp > 0.01 { up_t += 1; } else if dp < -0.01 { down_t += 1; }
                        }
                    }
                }
                let based = neurons.iter().filter(|x| x.mod_base > 1e-6).count();
                let (mut open_up, mut open_down) = (0usize, 0usize);
                for x in neurons.iter() {
                    if x.mod_base > 1e-6 { let r = x.mod_level / x.mod_base - 1.0; if r >= plasticity::MOD_DEADBAND { open_up += 1; } else if r <= -plasticity::MOD_DEADBAND { open_down += 1; } }
                }
                let pc = |a: usize, b: usize| if b > 0 { 100.0 * a as f64 / b as f64 } else { 0.0 };
                println!("        snn 경로 | 굳은 변화(위 강화%·약화%)와 별개로, 일시 성분 포함 강화 {:.3}% 약화 {:.3}% | 일시 성분이 0.01 넘는 시냅스 {:.3}%, 평균 크기 {:.6} | 흔적이 남은 시냅스 {:.3}%",
                    pc(eff_up, cnt), pc(eff_down, cnt), pc(sh_n, cnt), sh_abs / cnt.max(1) as f64, pc(el_n, cnt));
                println!("        snn 경로 | 도파민이 닿는 뉴런으로 들어가는 시냅스 {cnt_t}개 중 굳은 강화 {:.3}% 약화 {:.3}% | 기준선이 잡힌 뉴런 {based}개, 시행 끝(휴지 뒤)에 조절 수준이 기준선보다 25% 넘게 높은 뉴런 {open_up}, 낮은 뉴런 {open_down}",
                    pc(up_t, cnt_t), pc(down_t, cnt_t));
            }
            {
                // 지난 보고 뒤의 모든 시행 평균 (윗줄들은 이번 한 시행의 값)
                let w = win[cur];
                let k = w[0].max(1.0);
                println!("        창 평균 (최근 {:.0}회) | MN9 {:.2} | 활성 {:.1} | 총발화 {:.0} | 휴지발화 {:.1} | 정밀도 {:.3} | 재현율 {:.3} | 시행 1 의 활성집합과 겹침 {:.3}",
                    w[0], w[1] / k, w[2] / k, w[3] / k, w[4] / k, w[5] / k, w[6] / k, w[7] / k);
                win[cur] = [0.0; 8];
            }
            if extras {
                // 냄새·도파민 구동의 관찰 줄 (읽기만 한다)
                let k = ex_acc[0].max(1.0);
                let g = |p: usize, j: usize| if gate_acc[p][0] > 0.0 { gate_acc[p][j] / gate_acc[p][0] } else { 0.0 };
                let (mut based, mut bsum, mut bmin, mut bmax) = (0usize, 0.0f64, f64::MAX, 0.0f64);
                let mut bases: Vec<f64> = Vec::new();
                for &i in &gate_watch { let b = neurons[i].mod_base; if b > 1e-6 { based += 1; bsum += b; bmin = bmin.min(b); bmax = bmax.max(b); bases.push(b); } }
                bases.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let bmed = if bases.is_empty() { 0.0 } else { bases[bases.len() / 2] };
                let rate = |k: usize| if dan_rate[k] > 0.0 && !tonic_idx.is_empty() { dan_rate[k + 1] / dan_rate[k] / tonic_idx.len() as f64 * 1000.0 } else { 0.0 };
                println!("        구동·냄새 | 평소 발화를 주는 도파민 뉴런 {}개의 발화율 (뉴런당, 1000틱당, 구동 중인 틱 제외): 자극 구간 {:.1}, 휴지 구간 {:.1} | 기준선 값의 중앙 {:.5}",
                    tonic_idx.len(), rate(0), rate(2), bmed);
                dan_rate = [0.0; 4];
                println!("        구동·냄새 | 창 평균 (최근 {:.0}회): 발화한 도파민 뉴런 {:.1} (구동 대상 {}개 중 {:.1}) | 문이 열린 뉴런 수의 틱 평균, 위/아래: 구동 중 {:.1}/{:.1} ({:.0}틱), 자극 중 구동 밖 {:.1}/{:.1} ({:.0}틱), 휴지 {:.1}/{:.1} ({:.0}틱) | 구동 중 한 번이라도 열린 뉴런 {:.1}, 그중 그 시행에 발화한 뉴런 {:.1}",
                    ex_acc[0], ex_acc[1] / k, drive_idx.len(), ex_acc[2] / k, g(0, 1), g(0, 2), gate_acc[0][0], g(1, 1), g(1, 2), gate_acc[1][0], g(2, 1), g(2, 2), gate_acc[2][0], ex_acc[3] / k, ex_acc[4] / k);
                println!("        구동·냄새 | 도파민이 닿는 뉴런 {}개 중 기준선이 잡힌 뉴런 {based}개, 기준선 값 평균 {:.5} (최소 {:.5} ~ 최대 {:.5}) | 버섯체 출력 뉴런의 발화 수: 이번 시행 {}, 창 평균 {:.1} | 냄새 세포 {}개 중 발화: 이번 시행 {}, 창 평균 {:.1}",
                    gate_watch.len(), if based > 0 { bsum / based as f64 } else { 0.0 }, if based > 0 { bmin } else { 0.0 }, bmax,
                    mbon_idx.iter().map(|&i| counts[i] as u64).sum::<u64>(), ex_acc[5] / k, odor_idx.len(), odor_idx.iter().filter(|&&i| counts[i] > 0).count(), ex_acc[6] / k);
                println!("        구동·냄새 | 꼬리: 문이 열린 뉴런 수의 틱 평균, 위/아래 — 구동 직전 20틱 {:.1}/{:.1} ({:.0}틱), 휴지의 마지막 100틱 {:.1}/{:.1} ({:.0}틱) | 구동하는 유형을 듣는 뉴런 {}개 중 구동 중에 열린 수의 틱 평균, 위/아래: {:.1}/{:.1}",
                    g(3, 1), g(3, 2), gate_acc[3][0], g(4, 1), g(4, 2), gate_acc[4][0], n_hears,
                    if gate_drv[0] > 0.0 { gate_drv[1] / gate_drv[0] } else { 0.0 }, if gate_drv[0] > 0.0 { gate_drv[2] / gate_drv[0] } else { 0.0 });
                gate_drv = [0.0; 3];
                // 굳은 양: 지난 보고 뒤 가중치(굳은 부분)가 바뀐 양. 세 무리로 — 버섯체 세포 → 버섯체 출력 뉴런, 단맛에 활성인 뉴런 사이(기준 모델 기준, 감각뉴런 포함), 그 밖
                let sugar = |i: usize| refs[0][i] >= 1.0;
                let mut grp = [[0.0f64; 3]; 3]; // (절댓값 합, 부호 있는 합, 바뀐 시냅스 수)
                // "그 밖"을 다시 나눈다: [버섯체 세포 → 구동 유형을 듣는 다른 뉴런, 버섯체 세포가 아닌 것 → 버섯체 출력 뉴런, 버섯체 세포가 아닌 것 → 구동 유형을 듣는 다른 뉴런, 구동 유형을 듣지 않는 뉴런으로]
                let mut oth = [[0.0f64; 3]; 4];
                let mut hsp = [[[0.0f64; 2]; 3]; 2]; // [보내는 쪽 버섯체 세포 / 아님][받는 쪽 버섯체 세포 / APL / 그 밖] (절댓값 합, 부호 있는 합)
                for (pre, (nr, pw)) in neurons.iter().zip(prev_w.iter_mut()).enumerate() {
                    for (s, w0) in nr.synapses.iter().zip(pw.iter_mut()) {
                        let d = s.weight - *w0 as f64;
                        if d != 0.0 {
                            let t = s.target as usize;
                            let gi = if is_kc[pre] && is_mbon[t] { 0 } else if sugar(pre) && sugar(t) { 1 } else { 2 };
                            grp[gi][0] += d.abs(); grp[gi][1] += d; if d.abs() > 1e-6 { grp[gi][2] += 1.0; }
                            if hears_driven[t] && !is_mbon[t] {
                                let (a, b) = (if is_kc[pre] { 0 } else { 1 }, if is_kc[t] { 0 } else if is_apl[t] { 1 } else { 2 });
                                hsp[a][b][0] += d.abs(); hsp[a][b][1] += d;
                            }
                            if gi == 2 {
                                let oi = if !hears_driven[t] { 3 } else if is_kc[pre] { 0 } else if is_mbon[t] { 1 } else { 2 };
                                oth[oi][0] += d.abs(); oth[oi][1] += d; if d.abs() > 1e-6 { oth[oi][2] += 1.0; }
                            }
                            *w0 = s.weight as f32;
                        }
                    }
                }
                println!("        구동·냄새 | 굳은 양 (지난 보고 뒤 가중치 변화의 절댓값 합 / 부호 있는 합 / 바뀐 시냅스 수): 버섯체 세포→출력 뉴런 {:.5} / {:+.5} / {:.0}, 단맛 활성 뉴런 사이 {:.5} / {:+.5} / {:.0}, 그 밖 {:.5} / {:+.5} / {:.0}",
                    grp[0][0], grp[0][1], grp[0][2], grp[1][0], grp[1][1], grp[1][2], grp[2][0], grp[2][1], grp[2][2]);
                println!("        구동·냄새 | '그 밖'의 굳은 양을 다시 나눔 (절댓값 합 / 부호 있는 합 / 바뀐 시냅스 수): 버섯체 세포→구동 유형을 듣는 다른 뉴런 {:.5} / {:+.5} / {:.0}, 버섯체 세포 아닌 것→버섯체 출력 뉴런 {:.5} / {:+.5} / {:.0}, 버섯체 세포 아닌 것→구동 유형을 듣는 다른 뉴런 {:.5} / {:+.5} / {:.0}, 구동 유형을 듣지 않는 뉴런으로 {:.5} / {:+.5} / {:.0}",
                    oth[0][0], oth[0][1], oth[0][2], oth[1][0], oth[1][1], oth[1][2], oth[2][0], oth[2][1], oth[2][2], oth[3][0], oth[3][1], oth[3][2]);
                for gi in 0..3 { cum_grp[gi] += grp[gi][1]; }
                for a in 0..2 { for b in 0..3 { cum_h[a][b] += hsp[a][b][1]; } }
                println!("        구동·냄새 | 구동 유형을 듣는 뉴런(출력 뉴런 제외)으로 들어가는 굳은 양을 받는 쪽으로 가름 (절댓값 합 / 부호 있는 합): 버섯체 세포→버섯체 세포 {:.5} / {:+.5}, 버섯체 세포→APL {:.5} / {:+.5}, 버섯체 세포→그 밖 {:.5} / {:+.5}, 버섯체 세포 아닌 것→버섯체 세포 {:.5} / {:+.5}, 아닌 것→APL {:.5} / {:+.5}, 아닌 것→그 밖 {:.5} / {:+.5}",
                    hsp[0][0][0], hsp[0][0][1], hsp[0][1][0], hsp[0][1][1], hsp[0][2][0], hsp[0][2][1], hsp[1][0][0], hsp[1][0][1], hsp[1][1][0], hsp[1][1][1], hsp[1][2][0], hsp[1][2][1]);
                println!("        구동·냄새 | 처음부터의 부호 있는 누계: 버섯체 세포→출력 뉴런 {:+.5}, 단맛 활성 뉴런 사이 {:+.5}, 그 밖 {:+.5} | 버섯체 세포→버섯체 세포 {:+.5}, 버섯체 세포→APL {:+.5}, 버섯체 세포→그 밖 {:+.5}, 아닌 것→버섯체 세포 {:+.5}, 아닌 것→APL {:+.5}, 아닌 것→그 밖 {:+.5}",
                    cum_grp[0], cum_grp[1], cum_grp[2], cum_h[0][0], cum_h[0][1], cum_h[0][2], cum_h[1][0], cum_h[1][1], cum_h[1][2]);
                {
                    // 그 밖(허브)으로 들어오는 가중치 합: 굳은 것과 일시 성분을 따로 (관찰용)
                    let (mut w, mut ws, mut cnt) = (0.0f64, 0.0f64, 0usize);
                    for &h in &hub_idx { for &(pre, k) in &neurons[h].incoming { let sy = &neurons[pre].synapses[k];
                        w += sy.weight; cnt += 1;
                        if sy.w_short != 0.0 { ws += sy.w_short * sy.short_decay.powi(tick.saturating_sub(sy.t).min(1_000_000) as i32); } } }
                    let ex = |v: &[usize]| if v.is_empty() { 0.0 } else { v.iter().map(|&i| neurons[i].excitability).sum::<f64>() / v.len() as f64 };
                    println!("        구동·냄새 | 그 밖(허브) {}개로 들어오는 시냅스 {cnt}개: 굳은 가중치 합 {:.4}, 일시 성분 합 {:+.4} | 허브의 감응도 평균 {:.4}, 이번 시행에 발화한 허브 {}개",
                        hub_idx.len(), w, ws, ex(&hub_idx), hub_idx.iter().filter(|&&i| counts[i] > 0).count());
                    // 자극 구간의 틱별 발화 수와 쉬는 구간의 50틱 묶음 (이번 시행)
                    let st: Vec<String> = tick_counts[..stim_ticks as usize].iter().map(|x| x.to_string()).collect();
                    let rs: Vec<String> = tick_counts[stim_ticks as usize..].chunks(50).map(|c| c.iter().map(|&x| x as u64).sum::<u64>().to_string()).collect();
                    println!("        구동·냄새 | 이번 시행의 자극 구간 틱별 발화 수: {} | 쉬는 구간 50틱 묶음: {}", st.join(" "), rs.join(" "));
                }
                {
                    // 이번 시행의 쉬는 구간에 발화한 뉴런: 종류별, 그리고 무리별 (관찰용)
                    let ra: Vec<usize> = (0..n).filter(|&i| rest_active[i]).collect();
                    {
                        let exm = |v: &Vec<usize>| if v.is_empty() { 0.0 } else { v.iter().map(|&i| neurons[i].excitability).sum::<f64>() / v.len() as f64 };
                        let act: Vec<usize> = ra.iter().copied().filter(|&i| !is_tonic[i]).collect();
                        let quiet: Vec<usize> = (0..n).filter(|&i| !rest_active[i]).collect();
                        println!("        구동·냄새 | 쉬는 구간에 발화한 뉴런(평소 발화 도파민 제외) {}개의 감응도 평균 {:.4}, 발화하지 않은 뉴런의 감응도 평균 {:.4}", act.len(), exm(&act), exm(&quiet));
                        // 감응도가 1.2 를 넘는 뉴런: 번호, 분류, 감응도, 느린 발화율, 항상성 목표, 이번 시행의 발화 수 (관찰용)
                        let high: Vec<usize> = (0..n).filter(|&i| neurons[i].excitability > 1.2).collect();
                        println!("        구동·냄새 | 감응도 1.2 넘는 뉴런 {}개 (번호:분류:감응도:느린 발화율:항상성 목표:이번 시행 발화 수): {}", high.len(),
                            high.iter().map(|&i| format!("{i}:{}:{:.3}:{:.6}:{:.6}:{}", if classes.is_empty() { "?" } else { classes[i].as_str() }, neurons[i].excitability, neurons[i].rate_avg, neurons[i].set_point, counts[i])).collect::<Vec<_>>().join(" "));
                        if act.len() > 200 && watch_own.is_empty() { watch_own = act.clone(); }
                        if act.len() > 200 { if let Some(pre) = &rest_dump {
                            std::fs::write(format!("{pre}.t{trial}.txt"), act.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("\n")).ok();
                        } }
                    }
                    let mut by: std::collections::BTreeMap<&str, usize> = Default::default();
                    if !classes.is_empty() { for &i in &ra { *by.entry(classes[i].as_str()).or_default() += 1; } }
                    let mut rows: Vec<(&str, usize)> = by.into_iter().collect();
                    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
                    println!("        구동·냄새 | 이번 시행의 쉬는 구간에 발화한 뉴런 {}개 (평소 발화하는 도파민 뉴런 {}개 포함) | 종류별: {} | 버섯체 세포 {}, 버섯체 출력 뉴런 {}, 구동 유형을 듣는 뉴런 {}, 단맛에 활성인 뉴런(기준 모델) {}, 도파민 뉴런 {}",
                        ra.len(), ra.iter().filter(|&&i| is_tonic[i]).count(),
                        rows.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "),
                        ra.iter().filter(|&&i| is_kc[i]).count(), ra.iter().filter(|&&i| is_mbon[i]).count(), ra.iter().filter(|&&i| hears_driven[i]).count(),
                        ra.iter().filter(|&&i| refs[0][i] >= 1.0).count(), ra.iter().filter(|&&i| neurons[i].modulatory).count());
                }
                gate_acc = [[0.0; 3]; 5]; ex_acc = [0.0; 7];
            }
        }
    }
    // 마지막 휴지 구간에 발화한 뉴런 목록 저장 (기준 모델에서 같은 집단을 점화해 보기 위함)
    let list: Vec<String> = (0..n).filter(|&i| rest_active[i]).map(|i| i.to_string()).collect();
    std::fs::write("data/rest_active.txt", list.join("\n")).ok();
    if !classes.is_empty() {
        let mut hist: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
        for i in 0..n {
            let e = hist.entry(classes[i].as_str()).or_default();
            e.1 += 1;
            if rest_active[i] { e.0 += 1; }
        }
        let mut rows: Vec<_> = hist.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        println!("마지막 시행의 휴지 구간에 발화한 뉴런 (종류별): {}", rows.iter()
            .filter(|r| r.1.0 > 0).map(|(k, (a, t))| format!("{k} {a}/{t}")).collect::<Vec<_>>().join(", "));
    }
}

// ── 전뇌 규모의 냄새 + 도파민 짝지음 ──
//
// 버섯체 서브회로(mb.rs 의 run_da)에서 본 결과가 전뇌 배선 전체에서도 나오는지 본다.
//   냄새: 후각 감각뉴런(ORN) 그룹 몇 개를 자극 (--pn 이면 투사뉴런을 직접 자극)
//   도파민: 특정 유형의 도파민 뉴런을 직접 구동 (실험의 광유전학적 활성화에 해당)
//   측정: 짝지음 전후의 MBON 반응 (냄새 A = 짝지은 냄새, 냄새 B = 대조)
// --dan-all 이면 도파민 뉴런의 모든 출력을 조절 신호로 쓰고, 아니면 MBON 으로 가는 것만 쓴다.
// --learn 이면 검증된 학습 구성(STDP·iSTDP·BCM 등)도 함께 켠다.

pub fn run_da(c: &Connectome, args: &[String]) {
    let w_scale: f64 = arg(args, "--w-scale").unwrap_or(0.0068);
    let dan_name = arg::<String>(args, "--dan").unwrap_or("PPL101".into());
    let backward = flag(args, "--da-first");
    let pairings: usize = arg(args, "--pairings").unwrap_or(20);
    let n_groups: usize = arg(args, "--glom").unwrap_or(6);
    let seed: u64 = arg(args, "--seed").unwrap_or(1);
    let use_pn = flag(args, "--pn");
    // --kc: 냄새를 KC 층에서 직접 준다 (무작위로 고른 KC 의 일부가 5틱마다 발화).
    // 후각 감각뉴런이나 투사뉴런을 자극하면 이 배선 모델은 기준 모델에서도 뇌 전체가 폭주하므로, 그 단계를 건너뛴다
    let kc_frac: Option<f64> = arg(args, "--kc");
    let dan_all = flag(args, "--dan-all");
    let learn = flag(args, "--learn");
    let n = c.n();

    // 버섯체 관련 뉴런 목록 (전뇌 인덱스)
    let text = std::fs::read_to_string("data/mb_neurons.tsv").expect("data/mb_neurons.tsv 없음");
    let (mut mbon, mut dans, mut all_dan, mut kc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut pn_groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let idx: usize = f[0].parse().unwrap();
        match f[1] {
            "MBON" => mbon.push((idx, format!("{}({})", f[2], &f[3][..1]))),
            "DAN" => { all_dan.push(idx); if f[2] == dan_name { dans.push(idx); } }
            "KC" => kc.push(idx),
            "PN" => pn_groups.entry(f[2].to_string()).or_default().push(idx),
            _ => {}
        }
    }
    assert!(!dans.is_empty(), "도파민 뉴런 유형을 찾지 못함: {dan_name}");
    // 냄새 입력 그룹: ORN 유형별 (또는 --pn 이면 사구체별 투사뉴런)
    let mut groups: Vec<Vec<usize>> = if use_pn {
        pn_groups.into_values().collect()
    } else {
        let io = std::fs::read_to_string("data/io_neurons.tsv").expect("data/io_neurons.tsv 없음");
        let mut g: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        for line in io.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f[1] == "in" && f[2].starts_with("ORN_") { g.entry(f[2].to_string()).or_default().push(f[0].parse().unwrap()); }
        }
        g.into_values().collect()
    };
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    for i in (1..groups.len()).rev() { groups.swap(i, (next() % (i as u64 + 1)) as usize); }
    let odors: Vec<Vec<usize>> = match kc_frac {
        Some(frac) => (0..2).map(|_| {
            let want = (kc.len() as f64 * frac) as usize;
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < want {
                let k = kc[(next() % kc.len() as u64) as usize];
                if !chosen.contains(&k) { chosen.push(k); }
            }
            chosen
        }).collect(),
        None => (0..2).map(|k| groups[k * n_groups..(k + 1) * n_groups].iter().flatten().copied().collect()).collect(),
    };
    let pulse: u64 = if kc_frac.is_some() { 5 } else { 1 };
    let sensors: Vec<usize> = odors.iter().flatten().copied().collect();

    let is_mbon: std::collections::HashSet<usize> = mbon.iter().map(|m| m.0).collect();
    let mut neurons: Vec<Neuron> = (0..n).into_par_iter().map(|i| {
        let mut nr = if c.inhibitory[i] { Neuron::new_inhibitory(i as NeuronId, 0.0, 0.0) } else { Neuron::new(i as NeuronId, 0.0, 0.0) };
        nr.synapses = (c.out_start[i]..c.out_start[i + 1])
            .map(|k| Synapse::new(c.out_post[k], (c.out_count[k].unsigned_abs() as f64 * w_scale).min(1.0)))
            .collect();
        nr.spont_scale = 0.0;
        nr
    }).collect();
    for &i in &all_dan {
        neurons[i].inhibitory = false;
        neurons[i].modulatory = true;
        neurons[i].skip_refractory = true;
        if !dan_all { neurons[i].synapses.retain(|s| is_mbon.contains(&(s.target as usize))); }
    }
    for &i in &sensors { neurons[i].skip_refractory = true; }
    // --dan-no-input: 도파민 뉴런으로 들어오는 연결을 끊는다 (직접 구동할 때만 발화).
    // 이 배선 모델에서는 냄새만 줘도 KC 입력으로 도파민 뉴런이 스스로 발화하기 때문에, 그 영향을 빼고 볼 때 쓴다
    if flag(args, "--dan-no-input") {
        let is_dan: std::collections::HashSet<usize> = all_dan.iter().copied().collect();
        for nr in neurons.iter_mut() { nr.synapses.retain(|s| !is_dan.contains(&(s.target as usize))); }
    }
    plasticity::rebuild_incoming(&mut neurons);

    let mut dan_to: std::collections::HashMap<usize, f64> = Default::default();
    for &d in &dans { for s in &neurons[d].synapses { *dan_to.entry(s.target as usize).or_default() += s.weight; } }

    let mut opt = plasticity::PlasticityOptions {
        mod_eta: arg(args, "--mod-eta").unwrap_or(0.5),
        mod_forward_sign: arg(args, "--mod-forward").unwrap_or(-1.0),
        mod_backward_sign: arg(args, "--mod-backward").unwrap_or(1.0),
        stdp_on_inhibitory: false, local_scale: true, ltd_once_per_post: true, bcm_classic_local: Some(1.0),
        mod_gate: arg(args, "--mod-gate"),
        gate_base: arg(args, "--gate-base").unwrap_or(0.0),
        ..plasticity::PlasticityOptions::default()
    };
    if learn { plasticity::refresh_local_scales(&mut neurons, 0.95); }

    // 시행 사이 휴지 틱 수. 시냅스의 활성화 흔적(틱당 ×0.99)이 다음 시행까지 남으면
    // "도파민 → 냄새" 순서에서도 직전 시행의 흔적 때문에 약화가 걸리므로 충분히 길게 둔다
    let rest: u64 = arg(args, "--rest").unwrap_or(700);
    // --no-da: 도파민 없이 냄새만 같은 횟수 반복 (대조)
    let no_da = flag(args, "--no-da");
    let mut tick = 0u64;
    // 휴지 구간 마지막 100틱의 발화 수 (자극이 끝난 뒤에도 활동이 남는지 확인용)
    let late = std::cell::RefCell::new(0u64);
    // 한 시행: 냄새(odor_on 구간)와 도파민 뉴런(da_on 구간)을 구동. 반환: 뉴런별 발화 수
    let mut trial = |neurons: &mut Vec<Neuron>, opt: &plasticity::PlasticityOptions, odor: &[usize], odor_on: (u64, u64), da_on: Option<(u64, u64)>, plastic: bool| -> Vec<u32> {
        let mut counts = vec![0u32; n];
        for t in 0..80u64 + rest {
            tick += 1;
            for &i in &sensors { neurons[i].potential = 0.0; }
            if t >= odor_on.0 && t < odor_on.1 && (t - odor_on.0) % pulse == 0 {
                for &i in odor { neurons[i].potential = 1.0; neurons[i].self_threshold = 0.5; }
            }
            if let Some((a, b)) = da_on {
                if t >= a && t < b { for &d in &dans { neurons[d].potential = 1.0; neurons[d].self_threshold = 0.5; } }
            }
            let now = tick;
            let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = neurons.par_iter_mut().with_min_len(2048).filter_map(|nr| {
                nr.decay();
                nr.try_fire(now, 0.5, 0.0, true).map(|d| (nr.id, d))
            }).collect();
            let mut spikes = Vec::with_capacity(fired.len());
            for (nid, deliveries) in &fired {
                spikes.push(*nid);
                if t < 80 { counts[*nid as usize] += 1; }
                if t >= rest - 20 { *late.borrow_mut() += 1; }
                let pre = *nid as usize;
                let is_mod = neurons[pre].modulatory;
                for (k, &(target, w)) in deliveries.iter().enumerate() {
                    let tg = target as usize;
                    if is_mod {
                        neurons[tg].receive_mod(w.abs());
                        if plastic { plasticity::on_modulator(neurons, tg, w.abs(), now, opt); }
                    } else {
                        neurons[tg].receive(w);
                        if plastic && neurons[tg].mod_level > 0.001 { plasticity::on_pre_under_modulator(neurons, pre, k, opt); }
                    }
                }
            }
            if plastic && learn { plasticity::stdp_step(neurons, &spikes, now, opt); }
        }
        counts
    };

    let before: Vec<Vec<u32>> = odors.iter().map(|o| trial(&mut neurons, &opt, o, (0, 30), None, false)).collect();
    println!("측정(전) 뒤 휴지 구간 마지막 100틱의 발화 수 합: {}", late.replace(0));
    // 냄새만 줬을 때 뇌 안의 도파민 뉴런이 스스로 얼마나 발화하는가 (구동하지 않은 상태)
    for (k, name) in ["A", "B"].iter().enumerate() {
        let firing: Vec<usize> = all_dan.iter().copied().filter(|&d| before[k][d] > 0).collect();
        let spikes: u32 = firing.iter().map(|&d| before[k][d]).sum();
        println!("냄새 {name} 만 줬을 때 스스로 발화한 도파민 뉴런: {}개 / {} (총 {}회)", firing.len(), all_dan.len(), spikes);
    }
    // --only-driven: 직접 구동한 도파민 뉴런 유형의 신호만 학습에 쓴다 (다른 도파민 뉴런의 자발적 발화는 무시)
    if flag(args, "--only-driven") {
        let driven: std::collections::HashSet<usize> = dans.iter().copied().collect();
        for &d in &all_dan { if !driven.contains(&d) { neurons[d].synapses.clear(); } }
        plasticity::rebuild_incoming(&mut neurons);
    }
    let (odor_on, da_on) = if backward { ((25, 55), (0, 20)) } else { ((0, 30), (10, 40)) };
    for _ in 0..pairings { trial(&mut neurons, &opt, &odors[0], odor_on, if no_da { None } else { Some(da_on) }, true); }
    opt.mod_eta = 0.0;
    println!("짝지음 {pairings}회 동안 휴지 구간 마지막 100틱의 발화 수 합: {}", late.replace(0));
    let after: Vec<Vec<u32>> = odors.iter().map(|o| trial(&mut neurons, &opt, o, (0, 30), None, false)).collect();
    println!("측정(후) 뒤 휴지 구간 마지막 100틱의 발화 수 합: {}", late.replace(0));

    let kc_active = |c: &Vec<u32>| kc.iter().filter(|&&k| c[k] > 0).count();
    let mbon_total = |c: &Vec<u32>| mbon.iter().map(|m| c[m.0]).sum::<u32>();
    println!("[전뇌 도파민 짝지음] 도파민 뉴런 {dan_name} ({}개), 순서: {}, 짝지음 {pairings}회, 냄새 입력: {} 그룹 {n_groups}개 ({}개 뉴런), 도파민 출력: {}, 일반 학습 규칙: {}",
        dans.len(), if backward { "도파민 → 냄새" } else { "냄새 → 도파민" }, if kc_frac.is_some() { "KC 직접" } else if use_pn { "투사뉴런" } else { "감각뉴런" }, odors[0].len(),
        if dan_all { "전부" } else { "MBON 으로 가는 것만" }, if learn { "켬" } else { "끔" });
    println!("냄새 A: 발화한 KC {} → {} / 전체 {} , MBON 총 발화 {} → {}", kc_active(&before[0]), kc_active(&after[0]), kc.len(), mbon_total(&before[0]), mbon_total(&after[0]));
    println!("냄새 B: 발화한 KC {} → {} , MBON 총 발화 {} → {}", kc_active(&before[1]), kc_active(&after[1]), mbon_total(&before[1]), mbon_total(&after[1]));
    println!("{:<14} {:>10} {:>8} {:>8} {:>8} {:>8}", "MBON", "도파민 입력", "A 전", "A 후", "B 전", "B 후");
    let mut rows: Vec<(usize, &String, f64)> = mbon.iter().map(|m| (m.0, &m.1, dan_to.get(&m.0).copied().unwrap_or(0.0))).collect();
    rows.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    let (mut ta, mut tb, mut oa, mut ob) = ([0u32; 2], [0u32; 2], [0u32; 2], [0u32; 2]);
    let (mut za, mut zb, mut nz) = ([0u32; 2], [0u32; 2], 0usize);
    for (m, name, w) in rows {
        let (a0, a1, b0, b1) = (before[0][m], after[0][m], before[1][m], after[1][m]);
        if w >= 0.05 { ta[0] += a0; ta[1] += a1; tb[0] += b0; tb[1] += b1; }
        else if w > 0.0 { oa[0] += a0; oa[1] += a1; ob[0] += b0; ob[1] += b1; }
        else { za[0] += a0; za[1] += a1; zb[0] += b0; zb[1] += b1; nz += 1; }
        if w >= 0.05 && a0 + a1 + b0 + b1 > 0 { println!("{:<14} {:>10.2} {:>8} {:>8} {:>8} {:>8}", name, w, a0, a1, b0, b1); }
    }
    println!("합계 — 이 도파민 뉴런이 닿는 MBON (입력 0.05 이상): 냄새 A {} → {}, 냄새 B {} → {}", ta[0], ta[1], tb[0], tb[1]);
    println!("합계 — 약하게 닿는 MBON (입력 0 초과 0.05 미만):     냄새 A {} → {}, 냄새 B {} → {}", oa[0], oa[1], ob[0], ob[1]);
    println!("합계 — 전혀 닿지 않는 MBON ({nz}개):               냄새 A {} → {}, 냄새 B {} → {}", za[0], za[1], zb[0], zb[1]);
}
