//! 우리 엔진: snn 크레이트의 Neuron/Synapse 를 그대로 쓰고,
//! Network::tick 과 같은 순서(자극 → decay → try_fire → 전달)로 돌린다.
//! 가소성(STDP/sprout/수면)은 없음 — 배선이 고정된 커넥톰에서 발화 역학만 본다.

use crate::data::Connectome;
use crate::neuron::Neuron;
use crate::synapse::Synapse;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Config {
    /// 시냅스 1개당 가중치 (weight = 시냅스수 × w_scale)
    pub w_scale: f64,
    /// 막전위 노이즈 범위 (snn 기본 0.2)
    pub noise: f64,
    /// self_threshold 자가 적응 사용 (snn 기본 동작)
    pub adapt_threshold: bool,
    /// 시냅스 피로 사용 (snn 기본 동작)
    pub fatigue: bool,
    /// 막전위 감쇠율 (None = snn 기본 0.7)
    pub decay: Option<f64>,
    /// 불응기 없음 (모든 뉴런 skip_refractory)
    pub no_refractory: bool,
    /// 자극 틱 수
    pub stim_ticks: u64,
    /// 자극 종료 후 관찰 상한
    pub tail_cap: u64,
}

pub struct Outcome {
    pub counts: Vec<u32>,
    /// 자극 중 틱별 발화 수
    pub per_tick: Vec<u32>,
    /// 자극 종료 후 10틱 연속 무발화까지 걸린 틱 (None = 상한까지 안 꺼짐)
    pub settle_ticks: Option<u64>,
    pub stim_fire_ratio: f64,
}

pub fn build(c: &Connectome, cfg: &Config, stim: &[usize]) -> Vec<Neuron> {
    let mut neurons: Vec<Neuron> = (0..c.n()).into_par_iter().map(|i| {
        let mut n = if c.inhibitory[i] {
            Neuron::new_inhibitory(i as u32, 0.0, 0.0)
        } else {
            Neuron::new(i as u32, 0.0, 0.0)
        };
        n.synapses = (c.out_start[i]..c.out_start[i + 1])
            .map(|k| Synapse::new(c.out_post[k], c.out_count[k].unsigned_abs() as f64 * cfg.w_scale))
            .collect();
        if let Some(d) = cfg.decay { n.decay_rate = d; }
        if cfg.no_refractory { n.skip_refractory = true; }
        n
    }).collect();
    // 자극 뉴런은 snn 의 입력 뉴런과 같게: 불응기 없음
    for &i in stim { neurons[i].skip_refractory = true; }
    neurons
}

pub fn run(neurons: &mut [Neuron], cfg: &Config, stim: &[usize]) -> Outcome {
    let n = neurons.len();
    let mut counts = vec![0u32; n];
    let mut per_tick = Vec::new();
    let mut silent = 0u64;
    let mut settle_ticks = None;
    let mut stim_fires = 0u64;
    let mut is_stim = vec![false; n];
    for &i in stim { is_stim[i] = true; }

    let total = cfg.stim_ticks + cfg.tail_cap;
    for tick in 1..=total {
        let stimulating = tick <= cfg.stim_ticks;
        if stimulating {
            // Network::tick 의 sustain 과 동일: 전위 1.0 + 입력 뉴런 임계값 고정
            for &i in stim {
                neurons[i].potential = 1.0;
                neurons[i].self_threshold = 0.5;
            }
        }
        let fired: Vec<(u32, Vec<(u32, f64)>)> = neurons.par_iter_mut().with_min_len(2048).filter_map(|nr| {
            nr.decay();
            if !cfg.adapt_threshold { nr.self_threshold = 0.5; }
            let out = nr.try_fire(tick, 0.5, cfg.noise, true);
            if out.is_some() && !cfg.fatigue {
                for s in &mut nr.synapses { s.fatigue = 0.0; }
            }
            out.map(|d| (nr.id, d))
        }).collect();

        for (nid, deliveries) in &fired {
            if stimulating {
                counts[*nid as usize] += 1;
                if is_stim[*nid as usize] { stim_fires += 1; }
            }
            for &(target, w) in deliveries {
                neurons[target as usize].receive(w);
            }
        }
        if stimulating {
            per_tick.push(fired.len() as u32);
        } else {
            if fired.is_empty() { silent += 1; } else { silent = 0; }
            if silent >= 10 {
                settle_ticks = Some(tick - cfg.stim_ticks);
                break;
            }
        }
    }
    let stim_fire_ratio = stim_fires as f64 / (cfg.stim_ticks.max(1) * stim.len().max(1) as u64) as f64;
    Outcome { counts, per_tick, settle_ticks, stim_fire_ratio }
}
