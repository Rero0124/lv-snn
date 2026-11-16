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
    let opt = plasticity::PlasticityOptions {
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
            gate_base: arg(args, "--gate-base").unwrap_or(0.0), mod_baseline: false, pred_eta: 0.0, short_decay: None,
    };
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
    if flag(args, "--dan-mod") {
        let text = std::fs::read_to_string("data/mb_neurons.tsv").expect("data/mb_neurons.tsv 없음");
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f[1] == "DAN" { dan_idx.push(f[0].parse().unwrap()); }
        }
        for &i in &dan_idx { neurons[i].inhibitory = false; neurons[i].modulatory = true; }
        let active_ref = dan_idx.iter().filter(|&&i| refs[0][i] >= 1.0).count();
        println!("도파민 뉴런 {}개를 조절 뉴런으로 취급. 기준 모델에서 단맛에 반응하는 도파민 뉴런: {}개", dan_idx.len(), active_ref);
    }
    // 자발 발화 배수 (기준 모델은 자발 발화가 없다)
    let spont: f64 = arg(args, "--spont-scale").unwrap_or(1.0);
    for nr in neurons.iter_mut() { nr.spont_scale = spont; }
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
        for t in 0..stim_ticks + rest_ticks {
            tick += 1;
            let stimulating = t < stim_ticks;
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
            // snn 과 같이 자극 중에만 STDP (idle 틱에는 없음)
            if stdp && stimulating { plasticity::stdp_step(&mut neurons, &spikes, tick, &opt); }
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
