use crate::neuron::{Neuron, NeuronId};
use crate::region::RegionType;
use crate::tokenizer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── 상수 ──

const INITIAL_WEIGHT: f64 = 0.2;
const MIN_WEIGHT: f64 = 0.1;
const PRUNE_INTERVAL: u64 = 5_000;
const EMOTION_COUNT: usize = 5000;
const REASON_COUNT: usize = 5000;
const MEMORY_COUNT: usize = 2000;
const HIPPOCAMPUS_COUNT: usize = 500;

// ── 수면 파라미터 ──
const SLEEP_FIRE_INTERVAL: u64 = 30_000;
const SLEEP_TICK_INTERVAL: u64 = 500_000;
const SLEEP_DURATION_TICKS: u64 = 1_000;
const SLEEP_WEIGHT_SCALE: f64 = 0.92;

// ── 발화 기록 ──

pub struct DeliveryTrace {
    pub from: NeuronId,
    pub to: NeuronId,
    pub from_potential: f64,
    pub to_before: f64,
    pub weight: f64,
    pub delivered: f64,
    pub tick: u64,
}

pub struct EligibilityTrace {
    pub neuron_idx: usize,
    pub synapse_idx: usize,
    pub dw: f64,
}

pub struct FireRecord {
    pub id: u64,
    pub output: String,
    pub output_tokens: Vec<(String, f64)>,
    pub spiked_neurons: Vec<(NeuronId, u64)>,
    pub traces: Vec<DeliveryTrace>,
    pub eligibility: Vec<EligibilityTrace>,
}

pub struct ActiveStimulus {
    pub fire_id: u64,
    pub input_indices: Vec<usize>,
    pub remaining_sustain: u64,
    pub elapsed_ticks: u64,
    pub silent_ticks: u64,
    pub all_spikes: Vec<(NeuronId, u64)>,
    pub output_spikes: Vec<(NeuronId, u64)>,
    pub traces: Vec<DeliveryTrace>,
    pub collect_traces: bool,
}

// ── 네트워크 ──

pub struct Network {
    neurons: Vec<Neuron>,
    vocab: HashMap<String, u32>,
    reverse_vocab: Vec<String>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    emotion_range: (usize, usize),
    emotion_main_range: (usize, usize),
    emotion_map_range: (usize, usize),
    reason_range: (usize, usize),
    reason_main_range: (usize, usize),
    reason_map_range: (usize, usize),
    memory_range: (usize, usize),
    hippocampus_range: (usize, usize),

    pub fire_records: Vec<FireRecord>,
    next_fire_id: u64,
    pub debug: bool,
    pub active_stimuli: Vec<ActiveStimulus>,
    recent_spikes: Vec<(NeuronId, u64)>,
    pub threshold: f64,
    pub noise_range: f64,
    sprout_cooldown: HashMap<NeuronId, u64>,
    global_tick: u64,
    fires_since_sleep: u64,
    last_sleep_tick: u64,
}

impl Network {
    pub fn new() -> Self {
        let mut neurons: Vec<Neuron> = Vec::new();
        let cols = 50usize;

        // 감정 뉴런: Main 80% + Map 20% (30% 억제성)
        let emotion_start = neurons.len();
        let emotion_main_count = EMOTION_COUNT * 80 / 100;
        for i in 0..EMOTION_COUNT {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32;
            let id = neurons.len() as NeuronId;
            if i % 10 >= 7 {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let emotion_end = neurons.len();
        let emotion_main_range = (emotion_start, emotion_start + emotion_main_count);
        let emotion_map_range = (emotion_start + emotion_main_count, emotion_end);
        // Map 영역 threshold_scale 설정 (0.85 / 0.50 = 1.7)
        for idx in emotion_map_range.0..emotion_map_range.1 {
            neurons[idx].threshold_scale = 1.7;
        }

        // 이성 뉴런: Main 80% + Map 20% (30% 억제성)
        let reason_start = neurons.len();
        let reason_main_count = REASON_COUNT * 80 / 100;
        for i in 0..REASON_COUNT {
            let x = (i % cols) as f32;
            let y = (i / cols) as f32 + 100.0;
            let id = neurons.len() as NeuronId;
            if i % 10 >= 7 {
                neurons.push(Neuron::new_inhibitory(id, x, y));
            } else {
                neurons.push(Neuron::new(id, x, y));
            }
        }
        let reason_end = neurons.len();
        let reason_main_range = (reason_start, reason_start + reason_main_count);
        let reason_map_range = (reason_start + reason_main_count, reason_end);
        for idx in reason_map_range.0..reason_map_range.1 {
            neurons[idx].threshold_scale = 1.7;
        }

        // 기억 피질 뉴런 (30% 억제성, 느린 감쇠 0.85)
        let memory_start = neurons.len();
        for i in 0..MEMORY_COUNT {
            let x = (i % cols) as f32 + 50.0;
            let y = (i / cols) as f32 + 50.0;
            let id = neurons.len() as NeuronId;
            let inhibitory = i % 10 >= 7;
            neurons.push(Neuron::new_slow_decay(id, x, y, inhibitory));
        }
        let memory_end = neurons.len();

        // 해마 뉴런 (20% 억제성, 느린 감쇠 0.85)
        let hippo_start = neurons.len();
        for i in 0..HIPPOCAMPUS_COUNT {
            let x = (i % 25) as f32 + 60.0;
            let y = (i / 25) as f32 + 75.0;
            let id = neurons.len() as NeuronId;
            let inhibitory = i % 10 >= 8;
            neurons.push(Neuron::new_slow_decay(id, x, y, inhibitory));
        }
        let hippo_end = neurons.len();

        let mut net = Network {
            neurons,
            vocab: HashMap::new(),
            reverse_vocab: Vec::new(),
            input_idx: Vec::new(),
            output_idx: Vec::new(),
            emotion_range: (emotion_start, emotion_end),
            emotion_main_range,
            emotion_map_range,
            reason_range: (reason_start, reason_end),
            reason_main_range,
            reason_map_range,
            memory_range: (memory_start, memory_end),
            hippocampus_range: (hippo_start, hippo_end),
            fire_records: Vec::new(),
            next_fire_id: 1,
            debug: false,
            active_stimuli: Vec::new(),
            recent_spikes: Vec::new(),
            threshold: 0.50,
            noise_range: 0.2,
            sprout_cooldown: HashMap::new(),
            global_tick: 0,
            fires_since_sleep: 0,
            last_sleep_tick: 0,
        };

        // 초기 자모 등록
        let initial_tokens = tokenizer::all_tokens();
        for tok in &initial_tokens {
            net.register_token(tok);
        }

        // 초기 랜덤 시냅스
        net.seed_random_synapses(10);

        let total_synapses: usize = net.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [초기화] 어휘 {}개, 뉴런 {}개 (감정 {}, 이성 {}, 기억 {}, 해마 {}, 입출력 {}), 시냅스 {}개",
            net.vocab.len(), net.neurons.len(),
            EMOTION_COUNT, REASON_COUNT, MEMORY_COUNT, HIPPOCAMPUS_COUNT,
            net.input_idx.len() * 2, total_synapses);
        net
    }

    fn register_token(&mut self, token: &str) -> u32 {
        if let Some(&seq) = self.vocab.get(token) {
            return seq;
        }
        let seq = self.reverse_vocab.len() as u32;
        self.vocab.insert(token.to_string(), seq);
        self.reverse_vocab.push(token.to_string());

        let x = (seq % 20) as f32;
        let y_in = -10.0 + (seq / 20) as f32;
        let input_id = self.neurons.len() as NeuronId;
        self.neurons.push(Neuron::new_io(input_id, x, y_in));
        self.input_idx.push(input_id as usize);

        let y_out = 200.0 + (seq / 20) as f32;
        let output_id = self.neurons.len() as NeuronId;
        let mut out_neuron = Neuron::new_io(output_id, x, y_out);
        // 출력 뉴런 excitability 기본값(1.0) 유지
        self.neurons.push(out_neuron);
        self.output_idx.push(output_id as usize);

        seq
    }

    fn get_seq(&mut self, token: &str) -> u32 {
        self.register_token(token)
    }

    #[inline]
    fn input_neuron(&self, seq: u32) -> usize { self.input_idx[seq as usize] }
    #[inline]
    fn output_neuron(&self, seq: u32) -> usize { self.output_idx[seq as usize] }

    fn neuron_region(&self, idx: usize) -> RegionType {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;
        if idx >= es && idx < ee { RegionType::Emotion }
        else if idx >= rs && idx < re { RegionType::Reason }
        else if idx >= ms && idx < me { RegionType::Memory }
        else if idx >= hs && idx < he { RegionType::Hippocampus }
        else {
            if self.output_idx.contains(&idx) { RegionType::Output }
            else { RegionType::Input }
        }
    }

    fn is_output_neuron(&self, idx: usize) -> bool {
        self.output_idx.contains(&idx)
    }

    /// 구조 기반 랜덤 시냅스 생성
    fn seed_random_synapses(&mut self, per_neuron: usize) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        let (em_s, em_e) = self.emotion_main_range;
        let (emap_s, emap_e) = self.emotion_map_range;
        let (rm_s, rm_e) = self.reason_main_range;
        let (rmap_s, rmap_e) = self.reason_map_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;

        fn to_info(neurons: &[Neuron], range: std::ops::Range<usize>) -> Vec<(usize, f32, f32)> {
            range.map(|i| (i, neurons[i].x, neurons[i].y)).collect()
        }

        let emo_main_info = to_info(&self.neurons, em_s..em_e);
        let emo_map_info = to_info(&self.neurons, emap_s..emap_e);
        let rea_main_info = to_info(&self.neurons, rm_s..rm_e);
        let rea_map_info = to_info(&self.neurons, rmap_s..rmap_e);
        let mem_info = to_info(&self.neurons, ms..me);
        let hippo_info = to_info(&self.neurons, hs..he);
        let output_info: Vec<(usize, f32, f32)> = self.output_idx.iter()
            .map(|&i| (i, self.neurons[i].x, self.neurons[i].y)).collect();

        // Main 영역 (입력이 연결되는 곳): 감정_Main + 이성_Main + 기억
        let main_areas: Vec<(usize, f32, f32)> = emo_main_info.iter()
            .chain(rea_main_info.iter())
            .chain(mem_info.iter())
            .cloned().collect();

        // Map 영역: 감정_Map + 이성_Map
        let map_areas: Vec<(usize, f32, f32)> = emo_map_info.iter()
            .chain(rea_map_info.iter())
            .cloned().collect();

        fn pick_random(rng: &mut rand::rngs::ThreadRng, candidates: &[(usize, f32, f32)], exclude: usize) -> Option<usize> {
            if candidates.len() <= 1 { return None; }
            for _ in 0..10 {
                let idx = rng.random_range(0..candidates.len());
                if candidates[idx].0 != exclude {
                    return Some(candidates[idx].0);
                }
            }
            None
        }

        let w = INITIAL_WEIGHT;

        // 1. 입력 → 감정_Main, 이성_Main, 기억 (병렬)
        let input_indices: Vec<usize> = self.input_idx.clone();
        for &idx in &input_indices {
            for _ in 0..per_neuron {
                if let Some(target) = pick_random(&mut rng, &main_areas, idx) {
                    let tid = self.neurons[target].id;
                    self.neurons[idx].add_seed_synapse(tid, 0.35);
                }
            }
        }

        // 2. 감정_Main: 내부 + → 감정_Map + → 출력
        for &(idx, _, _) in &emo_main_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.5 {
                    // 내부 연결
                    if let Some(t) = pick_random(&mut rng, &emo_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else if r < 0.7 {
                    // → 감정_Map
                    if let Some(t) = pick_random(&mut rng, &emo_map_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else {
                    // → 출력
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, 1.0);
                    }
                }
            }
        }

        // 3. 이성_Main: 내부 + → 이성_Map + → 출력
        for &(idx, _, _) in &rea_main_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.5 {
                    if let Some(t) = pick_random(&mut rng, &rea_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else if r < 0.7 {
                    if let Some(t) = pick_random(&mut rng, &rea_map_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else {
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, 1.0);
                    }
                }
            }
        }

        // 4. 기억: → 출력 + → 해마
        for &(idx, _, _) in &mem_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.5 {
                    // 내부
                    if let Some(t) = pick_random(&mut rng, &mem_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else if r < 0.7 {
                    // → 출력
                    if let Some(t) = pick_random(&mut rng, &output_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, 1.0);
                    }
                } else {
                    // → 해마
                    if let Some(t) = pick_random(&mut rng, &hippo_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                }
            }
        }

        // 5. 감정_Map ↔ 기억, 해마
        let mem_hippo: Vec<(usize, f32, f32)> = mem_info.iter()
            .chain(hippo_info.iter()).cloned().collect();
        for &(idx, _, _) in &emo_map_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.4 {
                    // → 감정_Main (역방향)
                    if let Some(t) = pick_random(&mut rng, &emo_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else {
                    // → 기억 or 해마
                    if let Some(t) = pick_random(&mut rng, &mem_hippo, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                }
            }
        }

        // 6. 이성_Map ↔ 기억, 해마
        for &(idx, _, _) in &rea_map_info {
            for _ in 0..per_neuron {
                let r: f64 = rng.random();
                if r < 0.4 {
                    if let Some(t) = pick_random(&mut rng, &rea_main_info, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                } else {
                    if let Some(t) = pick_random(&mut rng, &mem_hippo, idx) {
                        self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                    }
                }
            }
        }

        // 7. 해마 → Map 영역 + 기억 (역방향)
        let map_and_mem: Vec<(usize, f32, f32)> = map_areas.iter()
            .chain(mem_info.iter()).cloned().collect();
        for &(idx, _, _) in &hippo_info {
            for _ in 0..per_neuron {
                if let Some(t) = pick_random(&mut rng, &map_and_mem, idx) {
                    self.neurons[idx].add_seed_synapse(t as NeuronId, w);
                }
            }
        }
    }

    /// 자극 주입
    pub fn inject_stimulus(&mut self, text: &str) -> u64 {
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;
        let jamo_list = tokenizer::decompose_to_jamo(text);
        let seqs: Vec<u32> = jamo_list.iter().map(|j| self.get_seq(j)).collect();
        let input_indices: Vec<usize> = seqs.iter().map(|&s| self.input_neuron(s)).collect();
        for &idx in &input_indices {
            self.neurons[idx].potential = 1.0;
        }
        self.active_stimuli.push(ActiveStimulus {
            fire_id,
            input_indices,
            remaining_sustain: 4,
            elapsed_ticks: 0,
            silent_ticks: 0,
            all_spikes: Vec::new(),
            output_spikes: Vec::new(),
            traces: Vec::new(),
            collect_traces: self.debug,
        });
        fire_id
    }

    /// idle 틱
    pub fn idle_tick(&mut self) {
        self.global_tick += 1;
        let global_tick = self.global_tick;

        for n in &mut self.neurons {
            n.decay();
        }

        let threshold = self.threshold;
        let noise_range = self.noise_range;
        let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = self.neurons
            .par_iter_mut()
            .filter_map(|n| {
                n.try_fire(global_tick, threshold, noise_range, false)
                    .map(|deliveries| (n.id, deliveries))
            })
            .collect();

        let mut tick_spikes: Vec<NeuronId> = Vec::new();
        for (nid, deliveries) in &fired {
            tick_spikes.push(*nid);
            for (target, weight) in deliveries {
                let tidx = *target as usize;
                if tidx < self.neurons.len() {
                    self.neurons[tidx].receive(*weight);
                }
            }
        }

        for &nid in &tick_spikes {
            self.recent_spikes.push((nid, global_tick));
        }

        if global_tick % 10_000 == 0 {
            for n in &mut self.neurons {
                n.homeostasis();
            }
        }

        if global_tick % 500 == 0 && !self.recent_spikes.is_empty() {
            let spikes: Vec<(NeuronId, u64)> = self.recent_spikes.drain(..).collect();
            self.sprout(&spikes);
        }
    }

    pub fn tick(&mut self) -> Vec<u64> {
        self.global_tick += 1;
        let global_tick = self.global_tick;

        for stim in &mut self.active_stimuli {
            stim.elapsed_ticks += 1;
            if stim.remaining_sustain > 0 {
                for &idx in &stim.input_indices {
                    self.neurons[idx].potential = 1.0;
                }
                stim.remaining_sustain -= 1;
            }
        }

        for n in &mut self.neurons {
            n.decay();
        }

        let threshold = self.threshold;
        let noise_range = self.noise_range;

        let fired: Vec<(NeuronId, f64, Vec<(NeuronId, f64)>)> = self.neurons
            .par_iter_mut()
            .filter_map(|n| {
                let pot = n.potential;
                n.try_fire(global_tick, threshold, noise_range, true)
                    .map(|deliveries| (n.id, pot, deliveries))
            })
            .collect();

        let mut tick_spikes: Vec<NeuronId> = Vec::new();
        let output_idx_snapshot = self.output_idx.clone();

        for (nid, from_potential, deliveries) in &fired {
            let nid = *nid;
            let from_potential = *from_potential;
            tick_spikes.push(nid);

            let is_output = output_idx_snapshot.contains(&(nid as usize));

            for stim in &mut self.active_stimuli {
                if stim.elapsed_ticks <= 14 {
                    stim.all_spikes.push((nid, global_tick));
                }
                if is_output {
                    stim.output_spikes.push((nid, global_tick));
                }
            }

            for (target, weight) in deliveries {
                let target = *target;
                let weight = *weight;
                let tidx = target as usize;
                if tidx >= self.neurons.len() { continue; }

                let to_before = self.neurons[tidx].potential;
                self.neurons[tidx].receive(weight);

                for stim in &mut self.active_stimuli {
                    if stim.collect_traces {
                        stim.traces.push(DeliveryTrace {
                            from: nid, to: target, from_potential,
                            to_before, weight, delivered: weight,
                            tick: global_tick,
                        });
                    }
                }
            }
        }

        // STDP: 발화 타이밍 기반 강화/약화 (LTD = LTP × 1.2)
        const STDP_A_PLUS_BASE: f64 = 0.004;
        const STDP_A_MINUS_BASE: f64 = 0.0048; // LTD:LTP = 1.2:1
        const STDP_TAU: f64 = 20.0;
        const BCM_TARGET_RATE: f64 = 25.0;
        let spike_ticks: Vec<Option<u64>> = self.neurons.iter()
            .map(|n| n.last_spike_tick).collect();
        let tick_spike_set: std::collections::HashSet<NeuronId> =
            tick_spikes.iter().cloned().collect();
        let fire_counts: Vec<u32> = self.neurons.iter()
            .map(|n| n.fire_count_window).collect();

        for &nid in &tick_spikes {
            let pidx = nid as usize;
            if pidx >= self.neurons.len() { continue; }
            let bcm_scale = BCM_TARGET_RATE / (fire_counts[pidx] as f64 + BCM_TARGET_RATE);
            let stdp_a_plus = STDP_A_PLUS_BASE * bcm_scale;
            let stdp_a_minus = STDP_A_MINUS_BASE * (2.0 - bcm_scale);
            for syn in &mut self.neurons[pidx].synapses {
                let tidx = syn.target as usize;
                if tidx >= spike_ticks.len() { continue; }

                if let Some(post_tick) = spike_ticks[tidx] {
                    let dt = post_tick as f64 - global_tick as f64;
                    if dt == 0.0 {
                        // 동시 발화: LTP
                        let ltp_bonus = syn.accumulate_ltp();
                        syn.weight = (syn.weight + stdp_a_plus + ltp_bonus).min(1.0);
                    } else if dt < 0.0 && dt > -50.0 {
                        // LTD
                        let dw = stdp_a_minus * (dt / STDP_TAU).exp();
                        syn.weight = (syn.weight - dw).max(0.0);
                    }
                }
            }
        }

        // iSTDP: 억제성 뉴런 발화 시, 타깃 활동 기반 억제 시냅스 조정
        const ISTDP_RATE: f64 = 0.04;
        const ISTDP_TARGET_RATE: f64 = 10.0;
        for &nid in &tick_spikes {
            let pidx = nid as usize;
            if pidx >= self.neurons.len() || !self.neurons[pidx].inhibitory { continue; }
            for syn in &mut self.neurons[pidx].synapses {
                let tidx = syn.target as usize;
                if tidx >= fire_counts.len() { continue; }
                let target_rate = fire_counts[tidx] as f64;
                if target_rate > ISTDP_TARGET_RATE {
                    syn.weight = (syn.weight + ISTDP_RATE).min(1.0);
                } else {
                    syn.weight = (syn.weight - ISTDP_RATE * 0.5).max(0.0);
                }
            }
        }

        for &nid in &tick_spikes {
            self.recent_spikes.push((nid, global_tick));
        }

        let has_spikes = !tick_spikes.is_empty();
        for stim in &mut self.active_stimuli {
            if has_spikes {
                stim.silent_ticks = 0;
            } else if stim.remaining_sustain == 0 {
                stim.silent_ticks += 1;
            }
        }

        let mut completed_ids: Vec<u64> = Vec::new();
        let mut completed_stimuli: Vec<ActiveStimulus> = Vec::new();
        let mut remaining: Vec<ActiveStimulus> = Vec::new();

        for stim in self.active_stimuli.drain(..) {
            if stim.silent_ticks >= 1 {
                completed_ids.push(stim.fire_id);
                completed_stimuli.push(stim);
            } else {
                remaining.push(stim);
            }
        }
        self.active_stimuli = remaining;

        for stim in completed_stimuli {
            let eligibility = self.compute_eligibility(&stim.all_spikes);

            let mut output_tokens: Vec<(String, f64)> = Vec::new();
            let mut seen_seqs = std::collections::HashSet::new();
            for &(nid, _) in &stim.output_spikes {
                let idx = nid as usize;
                if let Some(seq) = self.output_idx.iter().position(|&oi| oi == idx) {
                    if seen_seqs.insert(seq) {
                        if let Some(token) = self.reverse_vocab.get(seq) {
                            output_tokens.push((token.clone(), 1.0));
                        }
                    }
                }
            }

            let token_strs: Vec<String> = output_tokens.iter().map(|(t, _)| t.clone()).collect();
            let output = tokenizer::recompose_tokens(&token_strs);

            eprintln!(
                "  [fire #{}] ticks={}, spikes={}, output_spikes={}",
                stim.fire_id, stim.elapsed_ticks,
                stim.all_spikes.len(), stim.output_spikes.len(),
            );

            self.fire_records.push(FireRecord {
                id: stim.fire_id,
                output,
                output_tokens,
                traces: stim.traces,
                spiked_neurons: stim.all_spikes,
                eligibility,
            });
            if self.fire_records.len() > 100 {
                self.fire_records.remove(0);
            }
        }

        if global_tick % 10_000 == 0 {
            for n in &mut self.neurons {
                n.homeostasis();
            }
        }

        if global_tick % 500 == 0 && !self.recent_spikes.is_empty() {
            let spikes: Vec<(NeuronId, u64)> = self.recent_spikes.drain(..).collect();
            self.sprout(&spikes);
        }

        completed_ids
    }

    /// 수면
    pub fn enter_sleep(&mut self) {
        let enter_tick = self.global_tick;
        let fires = self.fires_since_sleep;

        // 1. 시냅스 fatigue 감소 + sleep LTD (wake 초과분 정리)
        const SLEEP_LTD: f64 = 0.001;
        let mut scaled = 0usize;
        for n in &mut self.neurons {
            for s in &mut n.synapses {
                s.fatigue *= SLEEP_WEIGHT_SCALE;
                s.weight = (s.weight - SLEEP_LTD).max(0.0);
                scaled += 1;
            }
        }

        // 2. 전역 흥분 상태 리셋
        for n in &mut self.neurons {
            n.potential = 0.0;
            n.excitability = 1.0;
            n.fire_count_window = 0;
        }

        // 3. 해마 재생: 해마 뉴런을 주기적으로 자극 → 시냅스 통해 기억 피질로 전파
        let (hs, he) = self.hippocampus_range;
        const REPLAY_ROUNDS: usize = 5;
        const REPLAY_TICKS: u64 = 10;
        for _ in 0..REPLAY_ROUNDS {
            // 해마 뉴런 전체를 약하게 자극
            for idx in hs..he {
                if idx < self.neurons.len() {
                    self.neurons[idx].potential = 0.5;
                }
            }
            // 전파 (STDP 없이 발화+전달만)
            for _ in 0..REPLAY_TICKS {
                self.global_tick += 1;
                let threshold = self.threshold;
                for n in &mut self.neurons {
                    n.decay();
                }
                let fired: Vec<(NeuronId, Vec<(NeuronId, f64)>)> = self.neurons
                    .iter_mut()
                    .filter_map(|n| {
                        if n.potential >= threshold {
                            n.potential = 0.0;
                            let sign = if n.inhibitory { -1.0 } else { 1.0 };
                            let deliveries: Vec<_> = n.synapses.iter()
                                .map(|s| (s.target, s.effective_weight() * sign))
                                .collect();
                            Some((n.id, deliveries))
                        } else { None }
                    })
                    .collect();
                for (_, deliveries) in &fired {
                    for &(target, weight) in deliveries {
                        let tidx = target as usize;
                        if tidx < self.neurons.len() {
                            self.neurons[tidx].receive(weight);
                        }
                    }
                }
            }
            // 라운드 후 리셋
            for n in &mut self.neurons {
                n.potential = 0.0;
            }
        }

        // 4. 휴지기: 순수 감쇠 (fatigue 자연 회복 포함)
        for _ in 0..SLEEP_DURATION_TICKS {
            self.global_tick += 1;
            for n in &mut self.neurons {
                n.decay();
            }
        }

        self.fires_since_sleep = 0;
        self.last_sleep_tick = self.global_tick;
        self.recent_spikes.clear();

        eprintln!(
            "  [sleep] tick {} → {} (지속 {}틱, 누적 {}fire, {}시냅스 fatigue×{}, 해마재생 {}회)",
            enter_tick, self.global_tick, SLEEP_DURATION_TICKS,
            fires, scaled, SLEEP_WEIGHT_SCALE, REPLAY_ROUNDS,
        );
    }

    fn should_sleep(&self) -> bool {
        self.fires_since_sleep >= SLEEP_FIRE_INTERVAL
            || self.global_tick.saturating_sub(self.last_sleep_tick) >= SLEEP_TICK_INTERVAL
    }

    /// 발화 (동기 방식)
    pub fn fire(&mut self, text: &str) -> u64 {
        if self.should_sleep() {
            self.enter_sleep();
        }
        self.fires_since_sleep += 1;

        let fire_id = self.inject_stimulus(text);
        loop {
            let completed = self.tick();
            if completed.contains(&fire_id) { break; }
        }

        if self.threshold < 1.0 {
            let new_threshold = (0.50 + (fire_id / 10_000) as f64 * 0.001).min(1.0);
            if new_threshold > self.threshold {
                self.threshold = new_threshold;
            }
        }
        if self.noise_range > 0.1 {
            self.noise_range = (0.2 - (fire_id / 10_000) as f64 * 0.001).max(0.1);
        }

        if fire_id % PRUNE_INTERVAL == 0 {
            let removed = self.prune(MIN_WEIGHT);
            if removed > 0 {
                eprintln!("  [prune] {removed}개 시냅스 제거");
            }
        }

        fire_id
    }

    fn compute_eligibility(&self, spikes: &[(NeuronId, u64)]) -> Vec<EligibilityTrace> {
        let spike_map: HashMap<NeuronId, u64> = spikes.iter().cloned().collect();
        let mut eligibility = Vec::new();

        for &(pre_id, pre_tick) in spikes {
            let pidx = pre_id as usize;
            if pidx >= self.neurons.len() { continue; }

            for (syn_idx, syn) in self.neurons[pidx].synapses.iter().enumerate() {
                if let Some(&post_tick) = spike_map.get(&syn.target) {
                    let dt = post_tick as f64 - pre_tick as f64;
                    let dw = if dt > 0.0 {
                        0.006 * (-dt.abs() / 20.0).exp()
                    } else if dt < 0.0 {
                        -0.006 * (-dt.abs() / 20.0).exp()
                    } else { 0.0 };
                    if dw != 0.0 {
                        eligibility.push(EligibilityTrace {
                            neuron_idx: pidx, synapse_idx: syn_idx, dw,
                        });
                    }
                }
            }
        }
        eligibility
    }

    fn sprout(&mut self, spikes: &[(NeuronId, u64)]) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        let sprout_weight = INITIAL_WEIGHT;
        const SPROUT_RADIUS: f32 = 5.0;
        const MAX_SPROUT_PER_NEURON: usize = 1;
        const SPROUT_COOLDOWN_TICKS: u64 = 500;
        const SPROUT_PROBABILITY: f64 = 0.1;
        let current_tick = self.global_tick;

        let spiked_info: Vec<(NeuronId, f32, f32)> = spikes.iter()
            .filter_map(|&(nid, _)| {
                let idx = nid as usize;
                if idx < self.neurons.len() {
                    Some((nid, self.neurons[idx].x, self.neurons[idx].y))
                } else { None }
            }).collect();

        let mut new_synapses: Vec<(usize, NeuronId)> = Vec::new();

        // BCM target rate 근처(발화율 1%~5%)인 뉴런만 발아 허용
        const SPROUT_MIN_RATE: f64 = 0.01;
        const SPROUT_MAX_RATE: f64 = 0.05;
        for &(nid, x, y) in &spiked_info {
            let idx = nid as usize;
            let is_input = self.input_idx.contains(&idx);
            let is_output = self.is_output_neuron(idx);
            // 발화율 체크: 과활성 뉴런은 발아 금지
            let fire_rate = self.neurons[idx].fire_count_window as f64 / 10_000.0;
            if fire_rate > SPROUT_MAX_RATE || fire_rate < SPROUT_MIN_RATE { continue; }
            if !rng.random_bool(SPROUT_PROBABILITY) { continue; }
            if let Some(&last) = self.sprout_cooldown.get(&nid) {
                if current_tick - last < SPROUT_COOLDOWN_TICKS { continue; }
            }

            let existing: std::collections::HashSet<NeuronId> =
                self.neurons[idx].synapses.iter().map(|s| s.target).collect();

            let mut candidates: Vec<(NeuronId, f32)> = self.neurons.iter()
                .filter_map(|n| {
                    let oidx = n.id as usize;
                    if n.id == nid || existing.contains(&n.id) { return None; }
                    let target_is_input = self.input_idx.contains(&oidx);
                    let target_is_output = self.output_idx.contains(&oidx);
                    if is_input && (target_is_input || target_is_output) { return None; }
                    if is_output && (target_is_output || target_is_input) { return None; }
                    let dist = ((x - n.x).powi(2) + (y - n.y).powi(2)).sqrt();
                    if dist <= SPROUT_RADIUS { Some((n.id, dist)) } else { None }
                }).collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut added = 0;
            for (target, _) in candidates {
                if added >= MAX_SPROUT_PER_NEURON { break; }
                new_synapses.push((idx, target));
                added += 1;
            }
            if added > 0 {
                self.sprout_cooldown.insert(nid, current_tick);
            }
        }

        for (pre_idx, post_id) in new_synapses {
            self.neurons[pre_idx].add_synapse(post_id, sprout_weight);
        }
    }

    pub fn feedback(&mut self, fire_id: u64, positive: bool, strength: f64) {
        let record = match self.fire_records.iter().find(|r| r.id == fire_id) {
            Some(r) => r, None => return,
        };
        let eligibility: Vec<(usize, usize, f64)> = record.eligibility.iter()
            .map(|e| (e.neuron_idx, e.synapse_idx, e.dw))
            .collect();

        let reward = if positive { strength } else { -strength };
        for (nidx, sidx, dw) in eligibility {
            if nidx >= self.neurons.len() { continue; }
            if sidx >= self.neurons[nidx].synapses.len() { continue; }
            let change = dw * reward;
            let w = &mut self.neurons[nidx].synapses[sidx].weight;
            *w = (*w + change).clamp(0.0, 1.0);
        }
    }

    /// teach: 정답 출력 뉴런 사전 활성화 → fire
    pub fn teach(&mut self, input: &str, target: &str) -> u64 {
        let target_jamo = tokenizer::decompose_to_jamo(target);
        let target_seqs: Vec<u32> = target_jamo.iter().map(|j| self.get_seq(j)).collect();
        for &seq in &target_seqs {
            let idx = self.output_neuron(seq);
            self.neurons[idx].potential += 0.6;
        }
        self.fire(input)
    }

    fn prune(&mut self, min_weight: f64) -> usize {
        let mut removed = 0;
        for neuron in &mut self.neurons {
            let before = neuron.synapses.len();
            neuron.synapses.retain(|s| s.weight >= min_weight);
            removed += before - neuron.synapses.len();
        }
        removed
    }

    // ── 상태 조회 ──

    pub fn synapse_count(&self) -> usize {
        self.neurons.iter().map(|n| n.synapses.len()).sum()
    }
    pub fn fire_count(&self) -> u64 { self.next_fire_id - 1 }
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    pub fn get_last_output(&self) -> String {
        self.fire_records.last().map(|r| r.output.clone()).unwrap_or_default()
    }
    pub fn get_last_record(&self) -> Option<&FireRecord> {
        self.fire_records.last()
    }

    pub fn nid_label(&self, nid: NeuronId) -> String {
        let idx = nid as usize;
        let region = self.neuron_region(idx);
        let name = match region {
            RegionType::Input => "입력",
            RegionType::Emotion => "감정",
            RegionType::Reason => "이성",
            RegionType::Memory => "기억",
            RegionType::Hippocampus => "해마",
            RegionType::Output => "출력",
        };
        format!("{name} #{nid}")
    }

    pub fn get_status(&self) -> serde_json::Value {
        let (es, ee) = self.emotion_range;
        let (rs, re) = self.reason_range;
        let (ms, me) = self.memory_range;
        let (hs, he) = self.hippocampus_range;
        let emotion_s: usize = (es..ee).map(|i| self.neurons[i].synapses.len()).sum();
        let reason_s: usize = (rs..re).map(|i| self.neurons[i].synapses.len()).sum();
        let memory_s: usize = (ms..me).map(|i| self.neurons[i].synapses.len()).sum();
        let hippo_s: usize = (hs..he).map(|i| self.neurons[i].synapses.len()).sum();
        let input_s: usize = self.input_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();
        let output_s: usize = self.output_idx.iter().map(|&i| self.neurons[i].synapses.len()).sum();

        // weight 분포 계산
        let mut w_bins = [0usize; 10]; // 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        let mut w_sum = 0.0f64;
        let mut w_count = 0usize;
        for n in &self.neurons {
            for s in &n.synapses {
                let bin = ((s.weight * 10.0) as usize).min(9);
                w_bins[bin] += 1;
                w_sum += s.weight;
                w_count += 1;
            }
        }
        let w_avg = if w_count > 0 { w_sum / w_count as f64 } else { 0.0 };

        serde_json::json!({
            "neurons": self.neurons.len(),
            "synapses": self.synapse_count(),
            "fire_count": self.fire_count(),
            "vocab_size": self.vocab_size(),
            "threshold": format!("{:.3}", self.threshold),
            "regions": {
                "입력": { "neurons": self.input_idx.len(), "synapses": input_s },
                "감정": { "neurons": EMOTION_COUNT, "synapses": emotion_s },
                "이성": { "neurons": REASON_COUNT, "synapses": reason_s },
                "기억": { "neurons": MEMORY_COUNT, "synapses": memory_s },
                "해마": { "neurons": HIPPOCAMPUS_COUNT, "synapses": hippo_s },
                "출력": { "neurons": self.output_idx.len(), "synapses": output_s },
            },
            "weight_dist": {
                "avg": format!("{:.4}", w_avg),
                "0.0-0.1": w_bins[0],
                "0.1-0.2": w_bins[1],
                "0.2-0.3": w_bins[2],
                "0.3-0.4": w_bins[3],
                "0.4-0.5": w_bins[4],
                "0.5-0.6": w_bins[5],
                "0.6-0.7": w_bins[6],
                "0.7-0.8": w_bins[7],
                "0.8-0.9": w_bins[8],
                "0.9-1.0": w_bins[9],
            },
        })
    }

    pub fn print_summary(&self) {
        println!("  어휘: {}개", self.vocab_size());
        println!("  뉴런: {}개, 시냅스: {}개", self.neurons.len(), self.synapse_count());
    }

    // ── 서버용 헬퍼 ──

    pub fn get_seq_pub(&mut self, token: &str) -> u32 { self.get_seq(token) }
    pub fn output_neuron_pub(&self, seq: u32) -> usize { self.output_neuron(seq) }
    pub fn neurons_mut(&mut self) -> &mut Vec<Neuron> { &mut self.neurons }
    pub fn has_active_stimuli(&self) -> bool { !self.active_stimuli.is_empty() }

    // ── 저장/불러오기 ──

    pub fn save(&self, path: &Path) {
        let snap = Snapshot {
            neurons: self.neurons.clone(),
            vocab: self.vocab.clone(),
            reverse_vocab: self.reverse_vocab.clone(),
            input_idx: self.input_idx.clone(),
            output_idx: self.output_idx.clone(),
            emotion_range: self.emotion_range,
            emotion_main_range: self.emotion_main_range,
            emotion_map_range: self.emotion_map_range,
            reason_range: self.reason_range,
            reason_main_range: self.reason_main_range,
            reason_map_range: self.reason_map_range,
            memory_range: self.memory_range,
            hippocampus_range: self.hippocampus_range,
            next_fire_id: self.next_fire_id,
            threshold: self.threshold,
            noise_range: self.noise_range,
        };
        let data = serde_json::to_vec(&snap).expect("직렬화 실패");
        std::fs::write(path, &data).expect("저장 실패");
        eprintln!("  [저장] {} ({:.1}MB, 뉴런 {}개, 시냅스 {}개)",
            path.display(), data.len() as f64 / 1048576.0,
            self.neurons.len(), self.synapse_count());
    }

    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let snap: Snapshot = serde_json::from_slice(&data).ok()?;
        let synapse_count: usize = snap.neurons.iter().map(|n| n.synapses.len()).sum();
        eprintln!("  [불러오기] {} (뉴런 {}개, 시냅스 {}개, 어휘 {}개)",
            path.display(), snap.neurons.len(), synapse_count, snap.vocab.len());
        Some(Network {
            neurons: snap.neurons,
            vocab: snap.vocab,
            reverse_vocab: snap.reverse_vocab,
            input_idx: snap.input_idx,
            output_idx: snap.output_idx,
            emotion_range: snap.emotion_range,
            emotion_main_range: snap.emotion_main_range,
            emotion_map_range: snap.emotion_map_range,
            reason_range: snap.reason_range,
            reason_main_range: snap.reason_main_range,
            reason_map_range: snap.reason_map_range,
            memory_range: snap.memory_range,
            hippocampus_range: snap.hippocampus_range,
            fire_records: Vec::new(),
            next_fire_id: snap.next_fire_id,
            debug: false,
            active_stimuli: Vec::new(),
            recent_spikes: Vec::new(),
            threshold: snap.threshold,
            noise_range: snap.noise_range,
            sprout_cooldown: HashMap::new(),
            global_tick: 0,
            fires_since_sleep: 0,
            last_sleep_tick: 0,
        })
    }
}

fn default_emotion_main_range() -> (usize, usize) { (0, 4000) }
fn default_emotion_map_range() -> (usize, usize) { (4000, 5000) }
fn default_reason_main_range() -> (usize, usize) { (5000, 9000) }
fn default_reason_map_range() -> (usize, usize) { (9000, 10000) }
fn default_memory_range() -> (usize, usize) { (10000, 12000) }
fn default_hippo_range() -> (usize, usize) { (12000, 12500) }

#[derive(Serialize, Deserialize)]
struct Snapshot {
    neurons: Vec<Neuron>,
    vocab: HashMap<String, u32>,
    reverse_vocab: Vec<String>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    emotion_range: (usize, usize),
    #[serde(default = "default_emotion_main_range")]
    emotion_main_range: (usize, usize),
    #[serde(default = "default_emotion_map_range")]
    emotion_map_range: (usize, usize),
    reason_range: (usize, usize),
    #[serde(default = "default_reason_main_range")]
    reason_main_range: (usize, usize),
    #[serde(default = "default_reason_map_range")]
    reason_map_range: (usize, usize),
    #[serde(default = "default_memory_range")]
    memory_range: (usize, usize),
    #[serde(default = "default_hippo_range")]
    hippocampus_range: (usize, usize),
    next_fire_id: u64,
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default = "default_noise")]
    noise_range: f64,
}
fn default_threshold() -> f64 { 0.50 }
fn default_noise() -> f64 { 0.2 }
