//! 버섯체(mushroom body) 연합학습 테스트.
//!
//! 실제 배선(PN → KC → MBON, APL 되먹임 억제) 위에서 snn 의 학습 규칙
//! (plasticity.rs: STDP + iSTDP + R-STDP)이 "냄새 → 특정 MBON" 연합을 배우는지 본다.
//! 자모 과제(입력 패턴 → 특정 출력 뉴런)와 같은 형태이고, 틱 루프도 Network::fire 와 같다.
//!
//! 규칙:
//!   none    — 가소성 없음 (기준선)
//!   stdp    — 비지도 STDP 만 (피드백 없음)
//!   ours    — STDP + R-STDP 피드백 (scripts/test_words.py 와 같은 보상 방식)
//!   rstdp   — R-STDP 피드백만 (비지도 STDP 끔)
//!   control — 대조군: KC→MBON 시냅스에만 델타 규칙 (이 배선에서 과제가 풀리는지 확인용)
//!
//! teach: --teach (snn 과 동일: 목표 출력 전위 +0.6 한 번), --teach-sustain (sustain 동안 매 틱 1.0 유지)
//!
//! 보상 변형 (ours / rstdp):
//!   --strict      오답 출력이 섞이면 음의 보상 (정답+오답 -0.5, 오답만 -1.0)
//!   --per-output  전역 보상 대신 출력 뉴런별 보상: 정답 출력으로 가는 시냅스 +1,
//!                 발화한 오답 출력으로 가는 시냅스 -1, 그 외 시냅스는 변경 없음
//!   --rpe         이미 정답(목표만 발화)인 시행은 갱신하지 않음 (보상 예측 오차 방식)

use crate::data::Connectome;
use crate::neuron::{Neuron, NeuronId};
use crate::plasticity::{self, PlasticityOptions};
use crate::synapse::Synapse;
use crate::{arg, flag};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Kind { Pn, Kc, Mbon, Apl, Dan }

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rule { None, Stdp, Rstdp, Ours, Control }

struct Circuit {
    neurons: Vec<Neuron>,
    kind: Vec<Kind>,
    /// 사구체별 PN (로컬 인덱스)
    glomeruli: Vec<(String, Vec<usize>)>,
    kc: Vec<usize>,
    mbon: Vec<usize>,
    mbon_type: HashMap<usize, String>,
    /// 도파민 뉴런 (로컬 인덱스 → 세포 유형, 예: PPL101)
    dan_type: HashMap<usize, String>,
}

struct Scales { pn_kc: f64, kc_mbon: f64, kc_apl: f64, apl_kc: f64, other: f64, dan_mbon: f64 }

fn load_circuit(c: &Connectome, scales: &Scales, print_stats: bool, shuffle: Option<u64>) -> Circuit {
    let text = std::fs::read_to_string("data/mb_neurons.tsv")
        .expect("data/mb_neurons.tsv 없음 — scripts/prepare_mb.py 를 먼저 실행");
    let mut local: HashMap<usize, usize> = HashMap::new();
    let mut kind = Vec::new();
    let mut full_idx = Vec::new();
    let mut glom_map: HashMap<String, Vec<usize>> = HashMap::new();
    let mut mbon_type = HashMap::new();
    let mut dan_type: HashMap<usize, String> = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let idx: usize = f[0].parse().unwrap();
        let k = match f[1] { "PN" => Kind::Pn, "KC" => Kind::Kc, "MBON" => Kind::Mbon, "DAN" => Kind::Dan, _ => Kind::Apl };
        if k == Kind::Dan { dan_type.insert(kind.len(), f[2].to_string()); }
        let li = kind.len();
        local.insert(idx, li);
        kind.push(k);
        full_idx.push(idx);
        if k == Kind::Pn { glom_map.entry(f[2].to_string()).or_default().push(li); }
        if k == Kind::Mbon { mbon_type.insert(li, format!("{}({})", f[2], &f[3][..1])); }
    }
    let mut glomeruli: Vec<(String, Vec<usize>)> = glom_map.into_iter().collect();
    glomeruli.sort();

    let mut stats: HashMap<(Kind, Kind), (usize, u64)> = HashMap::new();
    let mut neurons: Vec<Neuron> = Vec::with_capacity(kind.len());
    for (li, &fi) in full_idx.iter().enumerate() {
        let mut n = if c.inhibitory[fi] {
            Neuron::new_inhibitory(li as NeuronId, 0.0, 0.0)
        } else {
            Neuron::new(li as NeuronId, 0.0, 0.0)
        };
        for k in c.out_start[fi]..c.out_start[fi + 1] {
            let Some(&post) = local.get(&(c.out_post[k] as usize)) else { continue };
            let count = c.out_count[k].unsigned_abs() as f64;
            let pair = (kind[li], kind[post]);
            let s = match pair {
                (Kind::Pn, Kind::Kc) => scales.pn_kc,
                (Kind::Kc, Kind::Mbon) => scales.kc_mbon,
                (Kind::Kc, Kind::Apl) => scales.kc_apl,
                (Kind::Apl, Kind::Kc) => scales.apl_kc,
                // 도파민 뉴런: MBON 으로 가는 조절 신호만 쓴다. KC 축삭의 구획별 작용은 점 뉴런으로 표현할 수 없어 제외
                (Kind::Dan, Kind::Mbon) => scales.dan_mbon,
                (Kind::Dan, _) | (_, Kind::Dan) => 0.0,
                _ => scales.other,
            };
            let e = stats.entry(pair).or_default();
            e.0 += 1;
            e.1 += count as u64;
            if s > 0.0 {
                n.synapses.push(Synapse::new(post as NeuronId, (count * s).min(1.0)));
            }
        }
        // 입력(PN)은 snn 의 입력 뉴런과 같게 불응기 없음
        if kind[li] == Kind::Pn { n.skip_refractory = true; }
        if kind[li] == Kind::Dan { n.modulatory = true; n.skip_refractory = true; }
        neurons.push(n);
    }
    // 대조군: 연결 수와 가중치는 그대로 두고 PN→KC, KC→MBON 의 대상만 무작위로 바꾼다
    if let Some(seed) = shuffle {
        let mut rng = SplitMix(seed);
        let of = |k: Kind| -> Vec<usize> { (0..kind.len()).filter(|&i| kind[i] == k).collect() };
        let (kcs, mbons) = (of(Kind::Kc), of(Kind::Mbon));
        for (li, n) in neurons.iter_mut().enumerate() {
            for syn in &mut n.synapses {
                let pool = match (kind[li], kind[syn.target as usize]) {
                    (Kind::Pn, Kind::Kc) => &kcs,
                    (Kind::Kc, Kind::Mbon) => &mbons,
                    _ => continue,
                };
                syn.target = pool[(rng.next() % pool.len() as u64) as usize] as NeuronId;
            }
        }
    }
    plasticity::rebuild_incoming(&mut neurons);

    let pick = |k: Kind| -> Vec<usize> { (0..kind.len()).filter(|&i| kind[i] == k).collect() };
    let (kc, mbon, n_pn, n_apl) = (pick(Kind::Kc), pick(Kind::Mbon), pick(Kind::Pn).len(), pick(Kind::Apl).len());
    let circuit = Circuit { kc, mbon, neurons, kind, glomeruli, mbon_type, dan_type };
    if print_stats {
        let inh = |k: Kind| circuit.kind.iter().zip(&circuit.neurons)
            .filter(|(kk, n)| **kk == k && n.inhibitory).count();
        println!("[버섯체] PN {} (사구체 {}), KC {} (억제 표시 {}), MBON {} (억제 {}), APL {}",
            n_pn, circuit.glomeruli.len(), circuit.kc.len(), inh(Kind::Kc),
            circuit.mbon.len(), inh(Kind::Mbon), n_apl);
        let mut rows: Vec<_> = stats.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        for ((a, b), (n, total)) in rows {
            println!("  {:?}→{:?}: 연결 {n}, 연결당 평균 시냅스 {:.1}", a, b, total as f64 / n as f64);
        }
    }
    circuit
}

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

pub(crate) struct Sim {
    pub(crate) neurons: Vec<Neuron>,
    tick: u64,
    noise: f64,
    opt: PlasticityOptions,
    idle_ticks: u64,
    /// teach 를 sustain 동안 유지 (목표 출력을 입력처럼 매 틱 1.0 으로 고정)
    teach_sustain: bool,
    /// 한 자극에서 기록할 최대 발화 수 (넘으면 폭주로 보고 자극 중단)
    max_spikes: usize,
    /// 폭주로 중단된 자극 수
    runaway: usize,
}

impl Sim {
    /// Network::tick 과 같은 순서: decay → try_fire → 전달 → (STDP)
    fn step(&mut self, stimulated: bool, learn: bool) -> Vec<NeuronId> {
        self.tick += 1;
        let tick = self.tick;
        let noise = self.noise;
        let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = self.neurons.par_iter_mut().with_min_len(512)
            .filter_map(|n| {
                n.decay();
                n.try_fire(tick, 0.5, noise, stimulated).map(|d| (n.id, d))
            })
            .collect();
        let mut spikes = Vec::with_capacity(fired.len());
        for (nid, deliveries) in &fired {
            spikes.push(*nid);
            let pre = *nid as usize;
            let is_mod = self.neurons[pre].modulatory;
            for (k, &(target, w)) in deliveries.iter().enumerate() {
                let t = target as usize;
                if is_mod {
                    // 조절 뉴런: 전위는 올리지 않고 조절 수준만 올린다 + 조절 신호에 의한 가소성
                    self.neurons[t].receive_mod(w.abs());
                    plasticity::on_modulator(&mut self.neurons, t, w.abs(), tick, &self.opt);
                } else {
                    self.neurons[t].receive(w);
                    if self.neurons[t].mod_level > 0.001 {
                        plasticity::on_pre_under_modulator(&mut self.neurons, pre, k, &self.opt);
                    }
                }
            }
        }
        if learn {
            plasticity::stdp_step(&mut self.neurons, &spikes, tick, &self.opt);
        }
        if tick % 10_000 == 0 {
            for n in &mut self.neurons { n.homeostasis(); }
        }
        spikes
    }

    /// Network::fire 와 같은 한 번의 자극: sustain 4틱, 10틱 무발화 또는 2000틱에 종료
    fn trial(&mut self, inputs: &[usize], teach: Option<usize>, learn: bool) -> Vec<(NeuronId, u64)> {
        if let Some(t) = teach { self.neurons[t].potential += 0.6; }
        for &i in inputs { self.neurons[i].potential = 1.0; }
        let (mut sustain, mut silent, mut elapsed) = (4u32, 0u32, 0u64);
        let mut all = Vec::new();
        while silent < 10 && elapsed < 2000 {
            if all.len() > self.max_spikes { self.runaway += 1; break; }
            elapsed += 1;
            if sustain > 0 {
                for &i in inputs {
                    self.neurons[i].potential = 1.0;
                    self.neurons[i].self_threshold = 0.5;
                }
                if let (Some(t), true) = (teach, self.teach_sustain) { self.neurons[t].potential = 1.0; }
                sustain -= 1;
            }
            let spikes = self.step(true, learn);
            if spikes.is_empty() { if sustain == 0 { silent += 1; } } else { silent = 0; }
            let tick = self.tick;
            all.extend(spikes.into_iter().map(|n| (n, tick)));
        }
        for _ in 0..self.idle_ticks { self.step(false, false); }
        all
    }
}

pub fn run(c: &Connectome, args: &[String]) {
    let scales = Scales {
        pn_kc: arg(args, "--pn-kc").unwrap_or(0.02),
        kc_mbon: arg(args, "--kc-mbon").unwrap_or(0.002),
        kc_apl: arg(args, "--kc-apl").unwrap_or(0.002),
        apl_kc: arg(args, "--apl-kc").unwrap_or(0.02),
        other: arg(args, "--other").unwrap_or(0.0),
        dan_mbon: arg(args, "--dan-mbon").unwrap_or(0.01),
    };
    let rule = match arg::<String>(args, "--rule").as_deref() {
        Some("stdp") => Rule::Stdp, Some("ours") => Rule::Ours, Some("rstdp") => Rule::Rstdp,
        Some("control") => Rule::Control, _ => Rule::None,
    };
    let n_odors: usize = arg(args, "--odors").unwrap_or(4);
    let glom_per_odor: usize = arg(args, "--glom").unwrap_or(8);
    let epochs: usize = arg(args, "--epochs").unwrap_or(500);
    let probe_every: usize = arg(args, "--probe-every").unwrap_or(50);
    let teach = flag(args, "--teach");
    let strict = flag(args, "--strict");
    let per_output = flag(args, "--per-output");
    let rpe = flag(args, "--rpe");
    let seed: u64 = arg(args, "--seed").unwrap_or(1);

    let circuit = load_circuit(c, &scales, true, None);
    let Circuit { neurons, kind, glomeruli, kc, mbon, mbon_type, .. } = circuit;

    // 판독 뉴런: KC 입력이 가장 많은 MBON n_odors 개 (snn 의 출력 뉴런처럼 불응기 없음)
    let mut kc_inputs: HashMap<usize, usize> = HashMap::new();
    for &k in &kc {
        for s in &neurons[k].synapses {
            if kind[s.target as usize] == Kind::Mbon { *kc_inputs.entry(s.target as usize).or_default() += 1; }
        }
    }
    let mut ranked: Vec<usize> = mbon.clone();
    ranked.sort_by_key(|m| std::cmp::Reverse(kc_inputs.get(m).copied().unwrap_or(0)));
    let readout: Vec<usize> = ranked[..n_odors].to_vec();

    // 냄새: 사구체 무작위 부분집합
    let mut rng = SplitMix(seed);
    let odors: Vec<Vec<usize>> = (0..n_odors).map(|_| {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < glom_per_odor {
            let g = (rng.next() % glomeruli.len() as u64) as usize;
            if !chosen.contains(&g) { chosen.push(g); }
        }
        chosen.iter().flat_map(|&g| glomeruli[g].1.iter().copied()).collect()
    }).collect();

    let mut sim = Sim {
        neurons, tick: 0,
        noise: arg(args, "--noise").unwrap_or(0.2),
        opt: PlasticityOptions {
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
        },
        idle_ticks: arg(args, "--idle").unwrap_or(20),
        teach_sustain: flag(args, "--teach-sustain"),
        max_spikes: usize::MAX, runaway: 0,
    };
    for &r in &readout { sim.neurons[r].skip_refractory = true; }

    println!("[과제] 냄새 {n_odors}개(사구체 {glom_per_odor}개씩) → 판독 MBON {}",
        readout.iter().map(|r| format!("{} KC입력 {}", mbon_type[r], kc_inputs[r])).collect::<Vec<_>>().join(", "));
    println!("[설정] 규칙 {:?}, strict {strict}, per_output {per_output}, rpe {rpe}, teach {}, noise {}, {:?}, 가중치 배수 pn_kc {} kc_mbon {} kc_apl {} apl_kc {} other {}",
        rule, teach, sim.noise, sim.opt, scales.pn_kc, scales.kc_mbon, scales.kc_apl, scales.apl_kc, scales.other);
    println!("{:>6} {:>5} {:>6} {:>7} {:>8} {:>6} {:>7} {:>8} {:>9}  판독별 발화(냄새×MBON)",
        "epoch", "정답", "단독", "목표spk", "비목표spk", "KC%", "총spk", "w(PN→KC)", "w(KC→판독)");

    let mean_w = |sim: &Sim, pre: Kind, post: &dyn Fn(usize) -> bool| -> f64 {
        let (mut sum, mut n) = (0.0, 0usize);
        for (i, nr) in sim.neurons.iter().enumerate() {
            if kind[i] != pre { continue; }
            for s in &nr.synapses {
                if post(s.target as usize) { sum += s.weight; n += 1; }
            }
        }
        if n == 0 { 0.0 } else { sum / n as f64 }
    };

    let count_readout = |spikes: &[(NeuronId, u64)]| -> Vec<u32> {
        readout.iter().map(|&r| spikes.iter().filter(|s| s.0 as usize == r).count() as u32).collect()
    };

    for epoch in 0..=epochs {
        // ── 평가 (가소성 끔, teach 없음) ──
        if epoch % probe_every == 0 {
            let (mut correct, mut exclusive) = (0, 0);
            let (mut tgt, mut non, mut kc_frac, mut total) = (0.0, 0.0, 0.0, 0.0);
            let mut matrix = Vec::new();
            for (o, inputs) in odors.iter().enumerate() {
                let spikes = sim.trial(inputs, None, false);
                let counts = count_readout(&spikes);
                let best = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
                let unique_best = counts.iter().filter(|&&v| v == *best.1).count() == 1;
                if *best.1 > 0 && unique_best && best.0 == o { correct += 1; }
                if counts[o] > 0 && counts.iter().enumerate().all(|(i, &v)| i == o || v == 0) { exclusive += 1; }
                tgt += counts[o] as f64;
                non += counts.iter().enumerate().filter(|(i, _)| *i != o).map(|(_, &v)| v as f64).sum::<f64>()
                    / (n_odors - 1).max(1) as f64;
                let mut active = vec![false; sim.neurons.len()];
                for &(n, _) in &spikes { active[n as usize] = true; }
                kc_frac += kc.iter().filter(|&&k| active[k]).count() as f64 / kc.len() as f64;
                total += spikes.len() as f64;
                matrix.push(counts.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            }
            let k = n_odors as f64;
            println!("{:>6} {:>3}/{} {:>4}/{} {:>7.1} {:>8.1} {:>6.1} {:>7.0} {:>8.3} {:>9.3}  {}",
                epoch, correct, n_odors, exclusive, n_odors, tgt / k, non / k, 100.0 * kc_frac / k, total / k,
                mean_w(&sim, Kind::Pn, &|t| kind[t] == Kind::Kc),
                mean_w(&sim, Kind::Kc, &|t| readout.contains(&t)),
                matrix.join(" | "));
        }
        if epoch == epochs || rule == Rule::None { if rule == Rule::None && epoch >= probe_every { break; } else if epoch == epochs { break; } }

        // ── 학습 ──
        for (o, inputs) in odors.iter().enumerate() {
            let learn_stdp = matches!(rule, Rule::Stdp | Rule::Ours);
            let spikes = sim.trial(inputs, if teach { Some(readout[o]) } else { None }, learn_stdp);
            let counts = count_readout(&spikes);
            let target_fired = counts[o] > 0;
            let wrong_fired = counts.iter().enumerate().any(|(i, &v)| i != o && v > 0);
            match rule {
                Rule::Ours | Rule::Rstdp => {
                    let elig = plasticity::compute_eligibility(&sim.neurons, &spikes, &sim.opt);
                    if rpe && target_fired && !wrong_fired {
                        // 보상 예측 오차 0: 이미 정답이면 더 강화하지 않음
                    } else if per_output {
                        let target = readout[o];
                        let readout = &readout;
                        // --rpe 와 함께면 목표 출력은 오답 출력보다 약할 때만 강화 (오차 기반)
                        let max_wrong = counts.iter().enumerate().filter(|(i, _)| *i != o).map(|(_, &v)| v).max().unwrap_or(0);
                        let target_gain = if rpe && counts[o] > max_wrong { 0.0 } else { 1.0 };
                        plasticity::apply_reward(&mut sim.neurons, &elig, 1.0, |_, post| {
                            if post == target { target_gain } else if readout.contains(&post) { -1.0 } else { 0.0 }
                        });
                    } else {
                        // 기본: test_words.py 와 같은 방식 — 정답 +1.0, 부분 일치 양의 보상, 무출력/오답 -0.05
                        let reward = match (target_fired, wrong_fired, strict) {
                            (true, false, _) => 1.0,
                            (true, true, false) => 0.5,
                            (true, true, true) => -0.5,
                            (false, true, true) => -1.0,
                            _ => -0.05,
                        };
                        plasticity::apply_reward(&mut sim.neurons, &elig, reward, |_, _| 1.0);
                    }
                }
                Rule::Control => {
                    // 델타 규칙: 발화한 KC → 판독 MBON 시냅스를 (목표 - 실제) 방향으로
                    let mut fired = vec![false; sim.neurons.len()];
                    for &(n, _) in &spikes { fired[n as usize] = true; }
                    for (i, &r) in readout.iter().enumerate() {
                        let err = (if i == o { 1.0 } else { 0.0 }) - (if counts[i] > 0 { 1.0 } else { 0.0 });
                        if err == 0.0 { continue; }
                        let pairs = sim.neurons[r].incoming.clone();
                        for (pre, syn) in pairs {
                            if kind[pre] == Kind::Kc && fired[pre] {
                                let w = &mut sim.neurons[pre].synapses[syn].weight;
                                *w = (*w + 0.01 * err).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ── 자모 과제를 버섯체 배선 위에서 ──
//
// snn 의 자모 과제(fire + feedback, teach 없음)를 그대로 옮긴다.
//   입력: 자모 토큰마다 사구체 g개 조합 → 단어 = 그 자모들의 PN 전부
//   출력: 자모 토큰마다 MBON 하나 → 발화 순서대로 자모를 이어 붙여 문자열
//   은닉: 실제 버섯체 배선 (--shuffle 이면 연결 대상만 무작위로 섞은 대조군)
//   피드백: global (test_words.py 의 유사도 점수) / rpe (평소 점수 대비) — 둘 다 경로 전체에 적용

use crate::tokenizer;
use std::collections::HashSet;

fn jamo_set(s: &str) -> HashSet<String> {
    tokenizer::decompose_to_jamo(s).into_iter()
        .filter(|j| j.chars().next().is_some_and(|c| ('\u{3131}'..='\u{3163}').contains(&c)))
        .collect()
}

fn syllable_len(s: &str) -> f64 {
    s.chars().map(|c| if ('가'..='힣').contains(&c) { 1.0 } else { 0.3 }).sum()
}

/// test_words.py 와 같은 점수 → (positive, strength)
fn global_reward(out: &str, expected: &str) -> (bool, f64) {
    if out == expected { return (true, 1.0); }
    if out.is_empty() { return (false, 0.05); }
    let (o, e) = (jamo_set(out), jamo_set(expected));
    let mut score = 0.0;
    if out.contains(expected) { score += 0.5; }
    if out.chars().next() == expected.chars().next() { score += 0.2; }
    score += 0.3 * o.intersection(&e).count() as f64 / e.len().max(1) as f64;
    let diff = (syllable_len(out) - syllable_len(expected)).abs();
    if diff <= 1.5 { score += 0.2 / (1.0 + diff); }
    if score > 0.0 { (true, score.min(1.0)) } else { (false, 0.05) }
}

fn f1(out: &str, expected: &str) -> f64 {
    let (o, e) = (jamo_set(out), jamo_set(expected));
    let hit = o.intersection(&e).count() as f64;
    if hit == 0.0 { 0.0 } else { 2.0 * hit / (o.len() + e.len()) as f64 }
}

pub fn run_jamo(c: &Connectome, args: &[String]) {
    let scales = Scales {
        pn_kc: arg(args, "--pn-kc").unwrap_or(0.014),
        kc_mbon: arg(args, "--kc-mbon").unwrap_or(0.002),
        kc_apl: arg(args, "--kc-apl").unwrap_or(0.002),
        apl_kc: arg(args, "--apl-kc").unwrap_or(0.02),
        other: arg(args, "--other").unwrap_or(0.0),
        dan_mbon: arg(args, "--dan-mbon").unwrap_or(0.01),
    };
    let shuffle = if flag(args, "--shuffle") { Some(arg(args, "--shuffle-seed").unwrap_or(99)) } else { None };
    let glom_per_token: usize = arg(args, "--glom").unwrap_or(2);
    let Circuit { neurons, kind, glomeruli, kc, mbon, .. } = load_circuit(c, &scales, false, shuffle);
    let n_tokens = tokenizer::all_tokens().len();

    let mut kc_inputs: HashMap<usize, usize> = HashMap::new();
    for &k in &kc {
        for s in &neurons[k].synapses {
            if kind[s.target as usize] == Kind::Mbon { *kc_inputs.entry(s.target as usize).or_default() += 1; }
        }
    }
    let mut ranked = mbon.clone();
    ranked.sort_by_key(|m| std::cmp::Reverse(kc_inputs.get(m).copied().unwrap_or(0)));
    let out_neuron: Vec<usize> = ranked[..n_tokens].to_vec();

    let mut rng = SplitMix(arg(args, "--seed").unwrap_or(1));
    let token_inputs: Vec<Vec<usize>> = (0..n_tokens).map(|_| {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < glom_per_token {
            let g = (rng.next() % glomeruli.len() as u64) as usize;
            if !chosen.contains(&g) { chosen.push(g); }
        }
        chosen.iter().flat_map(|&g| glomeruli[g].1.iter().copied()).collect()
    }).collect();
    let label = if shuffle.is_some() { "버섯체 배선 — 무작위 재배선 대조군" } else { "버섯체 배선" };
    jamo_task(neurons, token_inputs, out_neuron, label, args);
}

/// 자모 과제 공용 루프 (fire + feedback, teach 없음). 어떤 배선이든 입력/출력 뉴런만 정해주면 된다.
pub(crate) fn jamo_task(neurons: Vec<Neuron>, token_inputs: Vec<Vec<usize>>, out_neuron: Vec<usize>, label: &str, args: &[String]) {
    let mode = arg::<String>(args, "--mode").unwrap_or("global".into());
    let rounds: usize = arg(args, "--rounds").unwrap_or(3000);
    let report: usize = arg(args, "--report").unwrap_or(500);
    let tokens = tokenizer::all_tokens();
    let token_idx: HashMap<&str, usize> = tokens.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    let out_token: HashMap<usize, usize> = out_neuron.iter().enumerate().map(|(t, &n)| (n, t)).collect();

    let inputs_for = |word: &str| -> Vec<usize> {
        let mut v: Vec<usize> = tokenizer::decompose_to_jamo(word).iter()
            .filter_map(|j| token_idx.get(j.as_str())).flat_map(|&t| token_inputs[t].iter().copied()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let decode = |spikes: &[(NeuronId, u64)]| -> String {
        let mut outs: Vec<(u64, usize)> = spikes.iter()
            .filter_map(|&(n, t)| out_token.get(&(n as usize)).map(|&tok| (t, tok))).collect();
        outs.sort_by_key(|x| x.0);
        let toks: Vec<String> = outs.iter().map(|&(_, tok)| tokens[tok].clone()).collect();
        tokenizer::recompose_tokens(&toks)
    };

    let mut sim = Sim {
        neurons, tick: 0,
        noise: arg(args, "--noise").unwrap_or(0.2),
        opt: PlasticityOptions {
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
        },
        idle_ticks: arg(args, "--idle").unwrap_or(20),
        teach_sustain: false,
        max_spikes: arg(args, "--max-spikes").unwrap_or(300_000), runaway: 0,
    };
    for &o in &out_neuron { sim.neurons[o].skip_refractory = true; }
    for inputs in &token_inputs { for &i in inputs { sim.neurons[i].skip_refractory = true; } }

    let tests = [("안녕", "반가워"), ("고마워", "천만에"), ("사랑해", "나도"), ("잘자", "좋은꿈"), ("배고파", "밥먹자")];
    println!("[자모 과제 / {label}] 뉴런 {}, 피드백 {mode}, 비지도 STDP 배수 {}, noise {}",
        sim.neurons.len(), sim.opt.stdp_scale, sim.noise);
    println!("{:>6} {:>5} {:>6} {:>6} {:>7} {:>8} {:>7} {:>5} {:>6}  예시",
        "round", "정답", "재현율", "정밀도", "출력길이", "활성뉴런", "발화수", "폭주", "초");

    let t0 = std::time::Instant::now();
    let mut baseline: HashMap<&str, f64> = HashMap::new();
    for round in 0..=rounds {
        if round % report == 0 {
            let (mut exact, mut rec, mut prec, mut len, mut active_n, mut total) = (0, 0.0, 0.0, 0usize, 0usize, 0usize);
            let mut samples = Vec::new();
            for (inp, expected) in tests {
                let spikes = sim.trial(&inputs_for(inp), None, false);
                let out = decode(&spikes);
                if out == expected { exact += 1; }
                let (o, e) = (jamo_set(&out), jamo_set(expected));
                let hit = o.intersection(&e).count() as f64;
                rec += hit / e.len() as f64;
                prec += if o.is_empty() { 0.0 } else { hit / o.len() as f64 };
                len += out.chars().count();
                let distinct: HashSet<NeuronId> = spikes.iter().map(|s| s.0).collect();
                active_n += distinct.len();
                total += spikes.len();
                samples.push(out.chars().take(8).collect::<String>());
            }
            let n = tests.len();
            println!("{:>6} {:>3}/{} {:>6.2} {:>6.2} {:>7.1} {:>8} {:>7} {:>5} {:>6.0}  {}",
                round, exact, n, rec / n as f64, prec / n as f64, len as f64 / n as f64,
                active_n / n, total / n, sim.runaway, t0.elapsed().as_secs_f64(), samples.join(" | "));
            sim.runaway = 0;
        }
        if round == rounds { break; }
        for (inp, expected) in tests {
            let before = sim.runaway;
            let spikes = sim.trial(&inputs_for(inp), None, true);
            if sim.runaway > before { continue; } // 폭주한 자극에는 피드백을 주지 않음
            let out = decode(&spikes);
            let reward = match mode.as_str() {
                "none" => 0.0,
                "rpe" => {
                    let score = f1(&out, expected);
                    let base = *baseline.get(inp).unwrap_or(&score);
                    baseline.insert(inp, 0.9 * base + 0.1 * score);
                    ((score - base) * 5.0).clamp(-1.0, 1.0)
                }
                _ => { let (pos, s) = global_reward(&out, expected); if pos { s } else { -s } }
            };
            if reward != 0.0 {
                let elig = plasticity::compute_eligibility(&sim.neurons, &spikes, &sim.opt);
                plasticity::apply_reward(&mut sim.neurons, &elig, reward, |_, _| 1.0);
            }
        }
    }
}

// ── 도파민 짝지음 실험 (조절 뉴런 검증) ──
//
// 알려진 실험 결과 (Hige 2015, Handler 2019 등):
//   냄새 → 도파민 순서로 짝지으면 그 도파민 뉴런이 닿는 구획의 KC→MBON 시냅스가 약해지고,
//   도파민 → 냄새 순서면 강해진다. 효과는 짝지은 냄새와 그 구획에 국한된다.
// 여기서는 부호(약화/강화)는 규칙의 설정값으로 주고, 다음이 배선에서 저절로 나오는지를 본다.
//   1) 어느 MBON 이 변하는가 — 그 도파민 뉴런이 시냅스를 만드는 MBON 인가
//   2) 냄새 특이성 — 짝지은 냄새 A 의 반응만 변하고 냄새 B 는 유지되는가
// 한계: 도파민 뉴런 → KC 축삭의 구획별 작용은 점 뉴런으로 표현할 수 없어, 도파민 뉴런 → MBON 연결만 쓴다.

pub fn run_da(c: &Connectome, args: &[String]) {
    let scales = Scales {
        pn_kc: arg(args, "--pn-kc").unwrap_or(0.014),
        kc_mbon: arg(args, "--kc-mbon").unwrap_or(0.002),
        kc_apl: arg(args, "--kc-apl").unwrap_or(0.002),
        apl_kc: arg(args, "--apl-kc").unwrap_or(0.02),
        other: 0.0,
        dan_mbon: arg(args, "--dan-mbon").unwrap_or(0.01),
    };
    let dan_name = arg::<String>(args, "--dan").unwrap_or("PPL101".into());
    let backward = flag(args, "--da-first");
    let pairings: usize = arg(args, "--pairings").unwrap_or(20);
    let glom_per_odor: usize = arg(args, "--glom").unwrap_or(8);
    let seed: u64 = arg(args, "--seed").unwrap_or(1);

    let Circuit { neurons, kind, glomeruli, mbon, mbon_type, dan_type, .. } = load_circuit(c, &scales, false, None);
    let dans: Vec<usize> = dan_type.iter().filter(|(_, t)| **t == dan_name).map(|(&i, _)| i).collect();
    assert!(!dans.is_empty(), "도파민 뉴런 유형을 찾지 못함: {dan_name}");

    // 이 도파민 뉴런이 각 MBON 으로 보내는 조절 가중치 합
    let mut dan_to: HashMap<usize, f64> = HashMap::new();
    for &d in &dans {
        for s in &neurons[d].synapses { *dan_to.entry(s.target as usize).or_default() += s.weight; }
    }

    let mut rng = SplitMix(seed);
    let odors: Vec<Vec<usize>> = (0..2).map(|_| {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < glom_per_odor {
            let g = (rng.next() % glomeruli.len() as u64) as usize;
            if !chosen.contains(&g) { chosen.push(g); }
        }
        chosen.iter().flat_map(|&g| glomeruli[g].1.iter().copied()).collect()
    }).collect();

    let mut sim = Sim {
        neurons, tick: 0,
        noise: arg(args, "--noise").unwrap_or(0.0),
        opt: PlasticityOptions {
            mod_eta: arg(args, "--mod-eta").unwrap_or(0.5),
            mod_forward_sign: arg(args, "--mod-forward").unwrap_or(-1.0),
            mod_backward_sign: arg(args, "--mod-backward").unwrap_or(1.0),
            mod_stdp_gain: arg(args, "--mod-stdp").unwrap_or(0.0),
            mod_gate: arg(args, "--mod-gate"),
            gate_base: arg(args, "--gate-base").unwrap_or(0.0),
            ..PlasticityOptions::default()
        },
        idle_ticks: 0, teach_sustain: false, max_spikes: usize::MAX, runaway: 0,
    };
    // 자발 발화 없이 결정적으로 (짝지음 전후 비교가 흔들리지 않게)
    for n in sim.neurons.iter_mut() { n.spont_scale = 0.0; }

    // 한 시행: 냄새(PN)를 odor_on 구간에, 도파민 뉴런을 da_on 구간에 구동. 일반 STDP 는 쓰지 않는다
    let trial = |sim: &mut Sim, odor: &[usize], odor_on: (u64, u64), da_on: Option<(u64, u64)>| -> Vec<u32> {
        let mut counts = vec![0u32; sim.neurons.len()];
        for t in 0..80u64 {
            if t >= odor_on.0 && t < odor_on.1 {
                for &i in odor { sim.neurons[i].potential = 1.0; sim.neurons[i].self_threshold = 0.5; }
            }
            if let Some((a, b)) = da_on {
                if t >= a && t < b { for &d in &dans { sim.neurons[d].potential = 1.0; sim.neurons[d].self_threshold = 0.5; } }
            }
            for n in sim.step(true, false) { counts[n as usize] += 1; }
        }
        for _ in 0..400 { sim.step(true, false); } // 흔적이 사라질 때까지 휴지
        counts
    };
    let probe = |sim: &mut Sim, odor: &[usize]| -> Vec<u32> {
        // 측정 중에는 가소성이 일어나지 않게 한다
        let eta = sim.opt.mod_eta;
        sim.opt.mod_eta = 0.0;
        let c = trial(sim, odor, (0, 30), None);
        sim.opt.mod_eta = eta;
        c
    };

    let before: Vec<Vec<u32>> = odors.iter().map(|o| probe(&mut sim, o)).collect();
    let (odor_on, da_on) = if backward { ((25, 55), (0, 20)) } else { ((0, 30), (10, 40)) };
    for _ in 0..pairings { trial(&mut sim, &odors[0], odor_on, Some(da_on)); }
    let after: Vec<Vec<u32>> = odors.iter().map(|o| probe(&mut sim, o)).collect();

    println!("[도파민 짝지음] 도파민 뉴런 {dan_name} ({}개), 순서: {}, 짝지음 {pairings}회, 냄새 A 와 짝지음",
        dans.len(), if backward { "도파민 → 냄새" } else { "냄새 → 도파민" });
    println!("{:<14} {:>10} {:>12} {:>12} {:>12} {:>12}", "MBON", "도파민 입력", "A 전", "A 후", "B 전", "B 후");
    let mut rows: Vec<(usize, f64)> = mbon.iter().map(|&m| (m, dan_to.get(&m).copied().unwrap_or(0.0))).collect();
    rows.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let (mut tgt_a, mut tgt_b, mut oth_a, mut oth_b) = ([0u32; 2], [0u32; 2], [0u32; 2], [0u32; 2]);
    for (m, w) in rows {
        let (a0, a1, b0, b1) = (before[0][m], after[0][m], before[1][m], after[1][m]);
        if w > 0.0 { tgt_a[0] += a0; tgt_a[1] += a1; tgt_b[0] += b0; tgt_b[1] += b1; }
        else { oth_a[0] += a0; oth_a[1] += a1; oth_b[0] += b0; oth_b[1] += b1; }
        if a0 + a1 + b0 + b1 > 0 && (w > 0.0 || a0 != a1 || b0 != b1) {
            println!("{:<14} {:>10.2} {:>12} {:>12} {:>12} {:>12}", mbon_type[&m], w, a0, a1, b0, b1);
        }
    }
    println!("합계 — 이 도파민 뉴런이 닿는 MBON: 냄새 A {} → {}, 냄새 B {} → {}", tgt_a[0], tgt_a[1], tgt_b[0], tgt_b[1]);
    println!("합계 — 닿지 않는 MBON:          냄새 A {} → {}, 냄새 B {} → {}", oth_a[0], oth_a[1], oth_b[0], oth_b[1]);
    let _ = kind;
}
