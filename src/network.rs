use crate::hippocampus::{Hippocampus, HippocampusState};
use crate::neuron::{Neuron, NeuronId, PASS_RATIO};
use crate::region::RegionType;
use crate::synapse::{PathMemory, SynapseId, SynapseStore};
use crate::tokenizer::{self, hash_to_index, TextTokens};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

// ── 상수 ──

const INITIAL_WEIGHT: f64 = 0.5;
const DECAY_RATE: f64 = 0.75;          // 감쇠율: 낮을수록 신호가 빨리 약해짐
const BACKWARD_LR: f64 = 0.01;        // 틱마다 시냅스 weight 조정
const FEEDBACK_LR: f64 = 0.1;         // 피드백 시 weight 조정
const MAX_WEIGHT: f64 = 1.0;
const MIN_WEIGHT: f64 = 0.01;
const MAX_TICKS: u64 = 50;
const CONSOLIDATION_INTERVAL: usize = 5;
const MIN_PATTERN_FREQ: u64 = 3;
const OUTPUT_EMISSION_DECAY: f64 = 0.3; // 출력 뉴런이 토큰 방출 후 activation에 곱하는 감쇠율
const RANDOM_SYNAPSE_WEIGHT: f64 = 0.1; // 초기 랜덤 시냅스 가중치 (낮게 시작)
const COFIRE_SYNAPSE_WEIGHT: f64 = 0.15; // co-firing으로 생성되는 시냅스 가중치

// ── 감정/이성 구역별 특성 ──
const EMOTION_THRESHOLD: f64 = 0.2;  // 감정: 낮은 임계값 (빠르게 반응)
const REASON_THRESHOLD: f64 = 0.35;  // 이성: 높은 임계값 (근거 필요)

// ── 상태 영속화 ──

#[derive(Serialize, Deserialize)]
struct NetworkState {
    neurons: HashMap<NeuronId, Neuron>,
    neuron_region: HashMap<NeuronId, RegionType>,
    region_neurons: Vec<(RegionType, Vec<NeuronId>)>,
    hippocampus: HippocampusState,
    next_fire_id: u64,
}

// ── 발화 기록 ──

pub struct FireRecord {
    pub id: u64,
    pub fired_synapses: Vec<SynapseId>,
    pub output_tokens: Vec<(String, SynapseId, u64)>,
    pub neurons_visited: Vec<NeuronId>,
    pub output: String,
    pub path_length: usize,
    pub rewarded: bool,
}

// ── 네트워크 ──

const COOLDOWN_HISTORY: usize = 10;    // 최근 N회 사용 이력 추적
const COOLDOWN_MAX_PENALTY: f64 = 0.3;  // 1회 전 사용 시 최대 감소율 (30% 감소)
const COOLDOWN_MIN_PENALTY: f64 = 0.05; // 10회 전 사용 시 최소 감소율 (5% 감소)
const RELEVANCE_BONUS: f64 = 2.0;      // 입력-출력 토큰 관련성 보너스 배율

// ── STDP (Spike-Timing-Dependent Plasticity) ──
const STDP_A_PLUS: f64 = 0.005;       // LTP 강화 크기 (pre→post 인과)
const STDP_A_MINUS: f64 = 0.007;      // LTD 약화 크기 (반인과, 약간 더 강하게)
const STDP_TAU: f64 = 10.0;           // 시간 상수 (틱 단위, 클수록 넓은 시간창)

pub struct Network {
    neurons: HashMap<NeuronId, Neuron>,
    neuron_region: HashMap<NeuronId, RegionType>,
    region_neurons: HashMap<RegionType, Vec<NeuronId>>,
    synapse_store: SynapseStore,
    fire_records: Vec<FireRecord>,
    next_fire_id: u64,
    hippocampus: Hippocampus,
    shutdown: Arc<AtomicBool>,
    // 최근 사용 억제: 시냅스ID → 최근 N회 fire에서 사용된 이력 (앞=최근, 뒤=오래됨)
    cooldown_history: HashMap<SynapseId, VecDeque<bool>>,
    fire_generation: u64, // 현재 fire 세대 (이력 관리용)
    // 현재 fire의 입력 토큰 (compose_response에서 관련성 판단용)
    current_input_tokens: HashSet<String>,
}

impl Network {
    pub fn new(db_path: PathBuf, max_cached_synapses: usize, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            neurons: HashMap::new(),
            neuron_region: HashMap::new(),
            region_neurons: HashMap::new(),
            synapse_store: SynapseStore::new(db_path, max_cached_synapses),
            fire_records: Vec::new(),
            next_fire_id: 1,
            hippocampus: Hippocampus::new(CONSOLIDATION_INTERVAL, MIN_PATTERN_FREQ),
            shutdown,
            cooldown_history: HashMap::new(),
            fire_generation: 0,
            current_input_tokens: HashSet::new(),
        }
    }

    pub fn try_load_state(&mut self) -> bool {
        let data = match self.synapse_store.load_network_state() {
            Some(d) => d,
            None => return false,
        };
        let state: NetworkState = match serde_json::from_slice(&data) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  상태 파싱 실패 (구조 변경?): {e}");
                eprintln!("  → data/network.redb를 삭제하고 다시 시작하세요.");
                return false;
            }
        };

        self.neurons = state.neurons;
        self.neuron_region = state.neuron_region;
        self.region_neurons = state.region_neurons.into_iter().collect();
        self.next_fire_id = state.next_fire_id;
        self.hippocampus.import_state(state.hippocampus);

        println!(
            "  [로드] 뉴런 {}개, 구역 {}개, 해마 패턴 {}개 복원",
            self.neurons.len(),
            self.region_neurons.len(),
            self.hippocampus.pattern_count(),
        );
        true
    }

    pub fn save_state(&self) {
        // 네트워크 상태를 먼저 저장 (SIGTERM 시에도 최소한 상태는 보존)
        let state = NetworkState {
            neurons: self.neurons.clone(),
            neuron_region: self.neuron_region.clone(),
            region_neurons: self.region_neurons.iter().map(|(k, v)| (*k, v.clone())).collect(),
            hippocampus: self.hippocampus.export_state(),
            next_fire_id: self.next_fire_id,
        };
        match serde_json::to_vec(&state) {
            Ok(bytes) => {
                self.synapse_store.save_network_state(&bytes);
                // stderr 사용 (stdout 파이프 block 방지)
                let _ = writeln!(
                    std::io::stderr(),
                    "  [저장] 뉴런 {}개, 해마 패턴 {}개 → DB 저장 완료",
                    self.neurons.len(),
                    self.hippocampus.pattern_count(),
                );
            }
            Err(e) => eprintln!("  상태 직렬화 실패: {e}"),
        }
        // dirty 시냅스 저장 (네트워크 상태 이후 — SIGTERM 시 시냅스만 유실)
        self.synapse_store.flush_dirty();
        // DB 전체 pruning (약한/중복 시냅스 정리)
        let (removed, remaining) = self.synapse_store.prune_db(MIN_WEIGHT);
        if removed > 0 {
            let _ = writeln!(
                std::io::stderr(),
                "  [prune] DB 전체 정리: {removed}개 제거 → {remaining}개 남음"
            );
        }
    }

    pub fn add_region(&mut self, region: RegionType, neuron_count: usize) {
        let threshold = match region {
            RegionType::Emotion => EMOTION_THRESHOLD,
            RegionType::Reason => REASON_THRESHOLD,
            _ => crate::neuron::DEFAULT_THRESHOLD,
        };

        let neurons: Vec<NeuronId> = (0..neuron_count)
            .map(|_| {
                let id = Uuid::new_v4().to_string();
                let mut neuron = Neuron::new(id.clone());
                neuron.threshold = threshold;
                self.neurons.insert(id.clone(), neuron);
                self.neuron_region.insert(id.clone(), region);
                id
            })
            .collect();
        self.region_neurons.entry(region).or_default().extend(neurons);
    }

    /// 초기 랜덤 빈 시냅스 씨앗 뿌리기
    /// - Output↔Output: 내부 순환용
    /// - 구역 간: region.targets()에 따라 랜덤 연결
    /// 토큰 없이 구조적 연결만 — 학습/해마가 강화/약화로 다듬음
    pub fn seed_random_synapses(&mut self, per_region_count: usize) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        // Output→Output 내부 순환
        if let Some(output_nids) = self.region_neurons.get(&RegionType::Output).cloned() {
            let n = output_nids.len();
            let count = per_region_count.min(n * 2);
            let mut created = 0;
            for _ in 0..count * 3 {
                if created >= count { break; }
                let a = rng.random_range(0..n);
                let b = rng.random_range(0..n);
                if a == b { continue; }
                let src = &output_nids[a];
                let dst = &output_nids[b];
                // 이미 연결 있으면 skip
                let neuron = self.neurons.get(src).unwrap();
                let already = neuron.outgoing.iter().any(|sid| {
                    self.synapse_store.get(sid).is_some_and(|s| s.post_neuron == *dst)
                });
                if already { continue; }
                let neuron = self.neurons.get_mut(src).unwrap();
                neuron.create_synapse(
                    &self.synapse_store, dst.clone(),
                    RANDOM_SYNAPSE_WEIGHT, None, None,
                );
                created += 1;
            }
            println!("  [seed] Output↔Output: {created}개 랜덤 시냅스");
        }

        // 구역 간 랜덤 연결 (targets에 따라)
        let regions: Vec<RegionType> = self.region_neurons.keys().copied().collect();
        let mut total_inter = 0;
        for region in &regions {
            let targets = region.targets();
            let src_nids = match self.region_neurons.get(region) {
                Some(ns) => ns.clone(),
                None => continue,
            };
            for target_region in targets {
                if target_region == region { continue; } // Output→Output은 위에서 처리
                let dst_nids = match self.region_neurons.get(target_region) {
                    Some(ns) => ns.clone(),
                    None => continue,
                };
                let count = (per_region_count / 2).max(1);
                let mut created = 0;
                for _ in 0..count * 3 {
                    if created >= count { break; }
                    let src = &src_nids[rng.random_range(0..src_nids.len())];
                    let dst = &dst_nids[rng.random_range(0..dst_nids.len())];
                    let neuron = self.neurons.get(src).unwrap();
                    let already = neuron.outgoing.iter().any(|sid| {
                        self.synapse_store.get(sid).is_some_and(|s| s.post_neuron == *dst)
                    });
                    if already { continue; }
                    let neuron = self.neurons.get_mut(src).unwrap();
                    neuron.create_synapse(
                        &self.synapse_store, dst.clone(),
                        RANDOM_SYNAPSE_WEIGHT, None, None,
                    );
                    created += 1;
                    total_inter += 1;
                }
            }
        }
        println!("  [seed] 구역 간: {total_inter}개 랜덤 시냅스");
    }

    /// 해마가 감지한 co-firing 쌍에 시냅스 생성/강화
    fn connect_cofiring_pairs(&mut self, pairs: &[((NeuronId, NeuronId), u64)]) {
        let mut created = 0;
        let mut strengthened = 0;

        for ((nid_a, nid_b), freq) in pairs {
            // 양방향 확인: A→B 있으면 강화, 없으면 생성
            let bonus = COFIRE_SYNAPSE_WEIGHT * (*freq as f64 / 10.0).min(1.0);

            // A→B
            let neuron_a = self.neurons.get(nid_a).unwrap();
            let existing_ab = neuron_a.outgoing.iter().find(|sid| {
                self.synapse_store.get(sid).is_some_and(|s| s.post_neuron == *nid_b)
            }).cloned();

            if let Some(sid) = existing_ab {
                if let Some(syn) = self.synapse_store.get(&sid) {
                    let new_w = (syn.weight + bonus).min(MAX_WEIGHT);
                    self.synapse_store.update_weight(&sid, new_w);
                    strengthened += 1;
                }
            } else {
                let neuron = self.neurons.get_mut(nid_a).unwrap();
                neuron.create_synapse(
                    &self.synapse_store, nid_b.clone(),
                    COFIRE_SYNAPSE_WEIGHT, None, None,
                );
                created += 1;
            }
        }

        if created > 0 || strengthened > 0 {
            eprintln!("  [해마 co-fire] 시냅스 생성 {created}개, 강화 {strengthened}개");
        }
    }

    fn reset_activations(&mut self) {
        for neuron in self.neurons.values_mut() {
            neuron.reset();
        }
    }

    // ═══════════════════════════════════════════════
    //  입력 처리: 토큰 → 뉴런 활성화 + 저장
    // ═══════════════════════════════════════════════

    /// 입력 토큰을 시냅스로 저장 — 입력에서 사방으로 퍼짐
    /// Input → Storage, Emotion, Reason (region targets에 따라)
    /// n-gram은 Storage에만 (기억 저장), 단어는 모든 대상 구역에
    fn store_input_tokens(&mut self, tokens: &TextTokens) {
        let input_neurons = self.region_neurons.get(&RegionType::Input).unwrap().clone();
        let storage_neurons = self.region_neurons.get(&RegionType::Storage).unwrap().clone();
        let emotion_neurons = self.region_neurons.get(&RegionType::Emotion).unwrap().clone();
        let reason_neurons = self.region_neurons.get(&RegionType::Reason).unwrap().clone();

        // 원문 → Storage + Emotion (전체 문맥 보존)
        self.ensure_token_synapse(&input_neurons, &storage_neurons, &tokens.original, INITIAL_WEIGHT * 0.5);
        self.ensure_token_synapse(&input_neurons, &emotion_neurons, &tokens.original, INITIAL_WEIGHT * 0.3);

        // 단어 → 모든 대상 구역으로 퍼짐 (사방 전파)
        for word in &tokens.words {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, word, INITIAL_WEIGHT);
            self.ensure_token_synapse(&input_neurons, &emotion_neurons, word, INITIAL_WEIGHT);
            self.ensure_token_synapse(&input_neurons, &reason_neurons, word, INITIAL_WEIGHT * 0.8);
        }

        // 단일 글자 → Storage + Emotion (빠른 패턴 매칭)
        for ch in &tokens.chars {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, ch, INITIAL_WEIGHT * 0.8);
            self.ensure_token_synapse(&input_neurons, &emotion_neurons, ch, INITIAL_WEIGHT * 0.6);
        }

        // 초성 → Storage (최소 단위, 넓은 매칭)
        for cho in &tokens.jamo {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, cho, INITIAL_WEIGHT * 0.6);
        }

        // n-gram → Storage에만 (기억 경로, 폭발 방지)
        for bg in &tokens.bigrams {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, bg, INITIAL_WEIGHT * 0.7);
        }
        for tg in &tokens.trigrams {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, tg, INITIAL_WEIGHT * 0.6);
        }
        for fg in &tokens.fourgrams {
            self.ensure_token_synapse(&input_neurons, &storage_neurons, fg, INITIAL_WEIGHT * 0.5);
        }
    }

    /// 모든 토큰 타입으로 입력 뉴런 동시 활성화 (병렬 경쟁)
    /// 입력 뉴런만 활성화 — 나머지는 시냅스를 통해 자연스럽게 퍼짐
    fn activate_all_tokens(&mut self, tokens: &TextTokens) {
        let input_neurons = self.region_neurons.get(&RegionType::Input).unwrap().clone();

        let idx = hash_to_index(&tokens.original, input_neurons.len());
        self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.5);

        for word in &tokens.words {
            let idx = hash_to_index(word, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.9);
        }

        // 단일 글자
        for ch in &tokens.chars {
            let idx = hash_to_index(ch, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.7);
        }

        // 초성
        for cho in &tokens.jamo {
            let idx = hash_to_index(cho, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.5);
        }

        for bg in &tokens.bigrams {
            let idx = hash_to_index(bg, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.4);
        }

        for tg in &tokens.trigrams {
            let idx = hash_to_index(tg, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.5);
        }

        for fg in &tokens.fourgrams {
            let idx = hash_to_index(fg, input_neurons.len());
            self.neurons.get_mut(&input_neurons[idx]).unwrap().receive(0.6);
        }
    }

    /// 응답 토큰을 시냅스로 저장 — 모든 처리 구역에서 Output으로
    /// Storage, Emotion, Reason → Output (각각 다른 가중치)
    fn store_response_tokens(&mut self, input_tokens: &TextTokens, target_tokens: &TextTokens) {
        let storage_neurons = self.region_neurons.get(&RegionType::Storage).unwrap().clone();
        let emotion_neurons = self.region_neurons.get(&RegionType::Emotion).unwrap().clone();
        let reason_neurons = self.region_neurons.get(&RegionType::Reason).unwrap().clone();
        let output_neurons = self.region_neurons.get(&RegionType::Output).unwrap().clone();

        // 각 구역에서 Output으로의 응답 시냅스
        let regions: Vec<(&[NeuronId], f64)> = vec![
            (&storage_neurons, INITIAL_WEIGHT * 1.4),  // 기억: 가장 강한 응답 (0.7)
            (&emotion_neurons, INITIAL_WEIGHT),         // 감정: 직관적 응답 (0.5)
            (&reason_neurons, INITIAL_WEIGHT * 1.2),    // 이성: 숙고된 응답 (0.6)
        ];

        for (src_neurons, base_weight) in &regions {
            let src_idx = hash_to_index(&input_tokens.original, src_neurons.len());
            let src_neuron = &src_neurons[src_idx];

            // 원문 토큰 (전체 응답 텍스트)
            {
                let dst_idx = hash_to_index(&target_tokens.original, output_neurons.len());
                self.ensure_synapse_between(
                    src_neuron, &output_neurons[dst_idx],
                    &target_tokens.original, base_weight * 0.3,
                );
            }

            // 단어 토큰 (가장 높은 가중치 — 출력의 핵심)
            for word in &target_tokens.words {
                let dst_idx = hash_to_index(word, output_neurons.len());
                self.ensure_synapse_between(
                    src_neuron, &output_neurons[dst_idx],
                    word, base_weight * 1.2,
                );
            }

            // 단일 글자 토큰 (중간)
            for ch in &target_tokens.chars {
                let dst_idx = hash_to_index(ch, output_neurons.len());
                self.ensure_synapse_between(
                    src_neuron, &output_neurons[dst_idx],
                    ch, base_weight * 0.5,
                );
            }

            // 자모 토큰 (가장 낮은 가중치 — 보조 역할)
            for cho in &target_tokens.jamo {
                let dst_idx = hash_to_index(cho, output_neurons.len());
                self.ensure_synapse_between(
                    src_neuron, &output_neurons[dst_idx],
                    cho, base_weight * 0.3,
                );
            }
        }
    }

    fn ensure_token_synapse(
        &mut self,
        src_neurons: &[NeuronId],
        dst_neurons: &[NeuronId],
        token: &str,
        weight: f64,
    ) {
        let src_idx = hash_to_index(token, src_neurons.len());
        let dst_idx = hash_to_index(token, dst_neurons.len());
        let src = &src_neurons[src_idx];
        let dst = &dst_neurons[dst_idx];
        self.ensure_synapse_between(src, dst, token, weight);
    }

    fn ensure_synapse_between(
        &mut self,
        src: &NeuronId,
        dst: &NeuronId,
        token: &str,
        weight: f64,
    ) {
        let existing = self.synapse_store.find_by_token(token);
        for sid in &existing {
            if let Some(syn) = self.synapse_store.get(sid) {
                if syn.pre_neuron == *src && syn.post_neuron == *dst {
                    let new_w = (syn.weight + weight * 0.1).min(MAX_WEIGHT);
                    self.synapse_store.update_weight(sid, new_w);
                    return;
                }
            }
        }

        let neuron = self.neurons.get_mut(src).unwrap();
        neuron.create_synapse(
            &self.synapse_store,
            dst.clone(),
            weight,
            Some(token.to_string()),
            None,
        );
    }

    // ═══════════════════════════════════════════════
    //  패턴 병합: 자주 연속 출현하는 토큰을 하나의 시냅스로 통합
    // ═══════════════════════════════════════════════

    /// 출력 토큰 시퀀스에서 자주 같이 나오는 패턴을 감지하여 병합 시냅스 생성
    /// 예: "ㅂ" + "습니다" → "ㅂ습니다" 로 병합
    fn consolidate_patterns(&mut self, output_tokens: &[(String, SynapseId, u64)]) {
        if output_tokens.len() < 2 {
            return;
        }

        let storage_neurons = self.region_neurons.get(&RegionType::Storage).unwrap().clone();
        let output_neurons = self.region_neurons.get(&RegionType::Output).unwrap().clone();

        // 틱 순서대로 정렬된 토큰에서 인접 쌍 추출
        let mut sorted: Vec<_> = output_tokens.to_vec();
        sorted.sort_by_key(|(_, _, tick)| *tick);

        for window in sorted.windows(2) {
            let (tok_a, _sid_a, tick_a) = &window[0];
            let (tok_b, _sid_b, tick_b) = &window[1];

            // 같은 틱이거나 연속 틱에서 나온 토큰만 병합 대상
            if tick_b - tick_a > 2 {
                continue;
            }

            // 실제 자모(ㄱㅏㅎ 등)끼리만 병합 (음절 합성용)
            // 한글 완성 음절(요, 기, 나)은 1글자여도 자모가 아니므로 제외
            if !is_jamo_token(tok_a) || !is_jamo_token(tok_b) {
                continue;
            }

            // 병합 토큰 생성 (자모 2개 → bigram)
            let merged = format!("{}{}", tok_a, tok_b);

            // 기존에 이 병합 토큰 시냅스가 있으면 강화, 없으면 생성
            let existing = self.synapse_store.find_by_token(&merged);
            let mut found = false;
            for sid in &existing {
                if let Some(syn) = self.synapse_store.get(sid) {
                    // 기존 병합 시냅스 강화
                    let new_w = (syn.weight + BACKWARD_LR).min(MAX_WEIGHT);
                    self.synapse_store.update_weight(sid, new_w);
                    found = true;
                    break;
                }
            }

            if !found {
                // 새 병합 시냅스 생성 (Storage → Output)
                let src_idx = hash_to_index(&merged, storage_neurons.len());
                let dst_idx = hash_to_index(&merged, output_neurons.len());
                self.ensure_synapse_between(
                    &storage_neurons[src_idx],
                    &output_neurons[dst_idx],
                    &merged,
                    INITIAL_WEIGHT,
                );
            }
        }
    }

    // ═══════════════════════════════════════════════
    //  쿨다운 (최근 사용 억제)
    // ═══════════════════════════════════════════════

    /// 최근 10회 사용 이력 기반 차등 쿨다운 페널티 계산
    /// 1회 전 사용: 80% 감소, 10회 전 사용: 5% 감소, 누적 합산
    fn compute_cooldown_penalty(&self, sid: &SynapseId) -> f64 {
        let history = match self.cooldown_history.get(sid) {
            Some(h) => h,
            None => return 0.0,
        };
        let mut total_penalty = 0.0;
        for (i, &used) in history.iter().enumerate() {
            if used {
                // i=0 (1회 전) → MAX, i=9 (10회 전) → MIN
                let t = i as f64 / (COOLDOWN_HISTORY - 1).max(1) as f64;
                let penalty = COOLDOWN_MAX_PENALTY * (1.0 - t) + COOLDOWN_MIN_PENALTY * t;
                total_penalty += penalty;
            }
        }
        total_penalty.min(0.95) // 최대 95% 감소 (완전 차단은 안 함)
    }

    // ═══════════════════════════════════════════════
    //  발화
    // ═══════════════════════════════════════════════

    pub fn fire(&mut self, input: &str) -> u64 {
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;

        self.reset_activations();

        self.fire_generation += 1;

        let tokens = tokenizer::tokenize(input);
        self.store_input_tokens(&tokens);
        self.activate_all_tokens(&tokens);

        // 입력 토큰 저장 (compose_response에서 관련성 판단용)
        self.current_input_tokens.clear();
        self.current_input_tokens.insert(tokens.original.clone());
        for w in &tokens.words {
            self.current_input_tokens.insert(w.clone());
        }
        for b in &tokens.bigrams {
            self.current_input_tokens.insert(b.clone());
        }
        for ch in &tokens.chars {
            self.current_input_tokens.insert(ch.clone());
        }
        for cho in &tokens.jamo {
            self.current_input_tokens.insert(cho.clone());
        }

        let mut all_fired: Vec<SynapseId> = Vec::new();
        let mut all_output_tokens: Vec<(String, SynapseId, u64)> = Vec::new();
        let mut neurons_activated: Vec<NeuronId> = Vec::new();
        let mut neuron_fire_ticks: Vec<(NeuronId, u64)> = Vec::new(); // co-firing 추적용
        let mut emitted_tokens: HashSet<String> = HashSet::new();
        // 큐: 발화한 토큰이 항상 쌓이고, Output on일 때만 배출
        let mut token_queue: VecDeque<(String, SynapseId, u64)> = VecDeque::new();

        for tick in 0..MAX_TICKS {
            if self.shutdown.load(Ordering::Relaxed) { break; }
            let result = self.run_tick(&emitted_tokens, tick);

            all_fired.extend(result.fired_synapses.iter().cloned());
            neurons_activated.extend(result.activated_neurons.iter().cloned());
            for nid in &result.activated_neurons {
                neuron_fire_ticks.push((nid.clone(), tick));
            }

            // 발화한 토큰은 항상 큐에 쌓임
            for (tok, sid) in &result.fired_tokens_queue {
                token_queue.push_back((tok.clone(), sid.clone(), tick));
            }

            // Output이 on이면 큐에서 빼서 출력
            if result.output_on {
                while let Some((tok, sid, t)) = token_queue.pop_front() {
                    if !emitted_tokens.contains(&tok) {
                        emitted_tokens.insert(tok.clone());
                        all_output_tokens.push((tok, sid, t));
                    }
                }

                // Output 뉴런 감쇠 (말하면 약해짐)
                for (nid, _) in &result.emitted_output_neurons {
                    if let Some(n) = self.neurons.get_mut(nid) {
                        n.activation *= OUTPUT_EMISSION_DECAY;
                    }
                }
            }

            // 자연 감쇠: 모든 활성 뉴런이 사라지면 종료
            if result.active_count == 0 {
                break;
            }
        }

        let (output, used_output_tokens) = self.assemble_output(&all_output_tokens);

        print!("[{fire_id}] {output}");
        println!();

        // 해마 기록: 경로 패턴 + co-firing
        self.hippocampus.record(&neurons_activated);
        self.hippocampus.record_cofiring(&neuron_fire_ticks);

        let patterns = self.hippocampus.maybe_consolidate();
        if !patterns.is_empty() {
            let total = patterns.len();
            println!(
                "  [해마] 통합 (발화 #{}, 패턴 {}개)",
                self.hippocampus.fire_count(),
                total
            );
            let show = total.min(3);
            for (pattern, freq) in &patterns[..show] {
                self.store_memory(pattern.clone(), *freq);
            }
            for (pattern, freq) in &patterns[show..] {
                self.store_memory_silent(pattern.clone(), *freq);
            }
            if total > 3 {
                println!("    ... 외 {}개 패턴 저장", total - 3);
            }
        }

        // 해마 co-firing 통합: 자주 동시 발화하는 뉴런 쌍에 시냅스 생성/강화
        let cofire_pairs = self.hippocampus.consolidate_cofiring();
        if !cofire_pairs.is_empty() {
            self.connect_cofiring_pairs(&cofire_pairs);
        }

        if output.is_empty() {
            println!("  (신호 소멸)");
        }

        // 출력/경로 시냅스에 사용 이력 기록
        let mut used_sids: HashSet<SynapseId> = HashSet::new();
        for (_, sid, _) in &all_output_tokens {
            used_sids.insert(sid.clone());
        }
        for sid in &all_fired {
            used_sids.insert(sid.clone());
        }
        for (_, history) in self.cooldown_history.iter_mut() {
            history.push_front(false); // 이번 fire에서 미사용
            if history.len() > COOLDOWN_HISTORY {
                history.pop_back();
            }
        }
        for sid in &used_sids {
            let history = self.cooldown_history.entry(sid.clone()).or_insert_with(VecDeque::new);
            if let Some(front) = history.front_mut() {
                *front = true; // 이번 fire에서 사용됨
            } else {
                history.push_front(true);
            }
        }
        // 10회 동안 한 번도 안 쓰인 시냅스는 이력 제거
        self.cooldown_history.retain(|_, h| h.iter().any(|&used| used));

        // 패턴 병합: 연속 출력 토큰을 묶어서 기억 구역에 저장
        self.consolidate_patterns(&all_output_tokens);

        let path_length = all_fired.len();
        self.fire_records.push(FireRecord {
            id: fire_id,
            fired_synapses: all_fired,
            output_tokens: used_output_tokens,
            neurons_visited: neurons_activated,
            output,
            path_length,
            rewarded: false,
        });

        // 주기적 pruning: 10회마다 약한/중복 시냅스 제거
        if fire_id > 0 && fire_id % 10 == 0 {
            let (removed, remaining) = self.synapse_store.prune(MIN_WEIGHT);
            if removed > 0 {
                eprintln!("  [prune] {removed}개 제거 → {remaining}개 남음");
            }
        }

        fire_id
    }

    /// 출력 토큰 조립: 발화 순서 + 가중치 기반
    /// Word → Char → Jamo 순으로 시도, 짧은 토큰은 jamo 재조합
    /// 출력 조립 — (출력문자열, 실제 사용된 토큰 목록) 반환
    fn assemble_output(&self, tokens: &[(String, SynapseId, u64)]) -> (String, Vec<(String, SynapseId, u64)>) {
        if tokens.is_empty() {
            return (String::new(), Vec::new());
        }

        // sid 역참조용 맵
        let sid_map: HashMap<String, (SynapseId, u64)> = tokens.iter()
            .map(|(tok, sid, tick)| (tok.clone(), (sid.clone(), *tick)))
            .collect();

        // 토큰별 점수 수집 (비선형: 고weight 강조, 저weight 억제)
        let mut scored: Vec<(String, f64, u64)> = Vec::new();
        for (tok, sid, tick) in tokens {
            if let Some(syn) = self.synapse_store.get(sid) {
                let bonus = tokenizer::fire_bonus(tok);
                let linear = syn.weight * bonus;
                // sigmoid-like 비선형 변환: 0.15~1.0 구간을 S자로
                // 0.3 이하는 급격히 억제, 0.5 이상은 급격히 강조
                let score = 1.0 / (1.0 + (-12.0 * (linear - 0.4)).exp());
                scored.push((tok.clone(), score, *tick));
            }
        }

        // 단어(4+글자) 우선 선별, 그 다음 ngram(2-3), 마지막 자모(1)
        // 각 카테고리 내에서 가중치 내림차순
        let mut words: Vec<(String, f64, u64)> = Vec::new();
        let mut ngrams: Vec<(String, f64, u64)> = Vec::new();
        let mut jamos: Vec<(String, f64, u64)> = Vec::new();
        for s in &scored {
            let len = s.0.chars().count();
            if len >= 4 { words.push(s.clone()); }
            else if len >= 2 { ngrams.push(s.clone()); }
            else { jamos.push(s.clone()); }
        }
        words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ngrams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        jamos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 단어 최대 15개 + ngram 최대 8개 + 자모 최대 5개
        words.truncate(15);
        ngrams.truncate(8);
        jamos.truncate(5);

        scored = Vec::new();
        scored.extend(words);
        scored.extend(ngrams);
        scored.extend(jamos);

        // 발화 순서(tick) 기준 정렬, 같은 tick이면 가중치 내림차순
        scored.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // 모든 토큰을 발화 순서대로 모아서 재조합
        // 긴 토큰(4+글자)은 단어로, 짧은 토큰(1글자)은 jamo 버퍼, 2-3글자는 ngram
        let mut parts: Vec<String> = Vec::new();
        let mut jamo_buf: Vec<char> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut used_tokens: Vec<(String, SynapseId, u64)> = Vec::new();

        let min_w = INITIAL_WEIGHT * 0.3;

        for (tok, w, _) in &scored {
            if *w < min_w { continue; }
            if seen.contains(tok) { continue; }
            seen.insert(tok.clone());
            // 실제 사용된 토큰 기록
            if let Some((sid, tick)) = sid_map.get(tok) {
                used_tokens.push((tok.clone(), sid.clone(), *tick));
            }

            let char_len = tok.chars().count();
            if char_len >= 4 {
                // 긴 토큰 (단어급): jamo 버퍼 비우고 그대로 추가
                if !jamo_buf.is_empty() {
                    let composed = tokenizer::compose_jamo(&jamo_buf);
                    let merged = tokenizer::merge_trailing_jamo(&composed);
                    if !merged.is_empty() { parts.push(merged); }
                    jamo_buf.clear();
                }
                parts.push(tok.clone());
            } else if char_len <= 1 {
                // 짧은 토큰 (jamo/char급): jamo 버퍼에 쌓기
                for c in tok.chars() {
                    jamo_buf.push(c);
                }
            } else {
                // 2-3글자 (ngram급): jamo 버퍼 비우고 그대로 추가
                if !jamo_buf.is_empty() {
                    let composed = tokenizer::compose_jamo(&jamo_buf);
                    let merged = tokenizer::merge_trailing_jamo(&composed);
                    if !merged.is_empty() { parts.push(merged); }
                    jamo_buf.clear();
                }
                parts.push(tok.clone());
            }
        }

        // 남은 jamo 버퍼 처리
        if !jamo_buf.is_empty() {
            let composed = tokenizer::compose_jamo(&jamo_buf);
            let merged = tokenizer::merge_trailing_jamo(&composed);
            if !merged.is_empty() { parts.push(merged); }
        }

        let output = if parts.is_empty() {
            // 폴백: 가중치 상위 토큰 연결
            scored.iter().take(3).map(|(t, _, _)| t.as_str()).collect::<Vec<_>>().join("")
        } else {
            parts.join(" ")
        };
        (output, used_tokens)
    }

    /// (미사용) 다층 응답 조합: Original → Word 조합 → Char 조합 → Jamo 재조합
    fn compose_response(&mut self, tokens: &[(String, SynapseId, u64)]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut scored: Vec<(String, f64, u64)> = Vec::new();
        for (tok, sid, tick) in tokens {
            if let Some(syn) = self.synapse_store.get(sid) {
                let mut weight = syn.weight;
                // 짧은 토큰 보너스 (fire_bonus 기반)
                weight *= tokenizer::fire_bonus(tok);
                // 입력-출력 바인딩: 입력 토큰과 관련된 출력에 보너스
                if let Some(ref syn_tok) = syn.token {
                    if self.current_input_tokens.contains(syn_tok) {
                        weight *= RELEVANCE_BONUS;
                    }
                }
                scored.push((tok.clone(), weight, *tick));
            }
        }

        // 가중치 내림차순 (가장 강한 신호 우선)
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.2.cmp(&b.2))
        });

        // ── 1층: 긴 토큰 (문장급, 5+ 글자) ──
        if let Some((tok, w, _)) = scored.iter().find(|(t, _, _)| t.chars().count() >= 5) {
            if *w >= INITIAL_WEIGHT {
                return tok.clone();
            }
        }

        // ── 2층: Word 조합 (4+ 글자 단어를 이어붙여 새 문장) ──
        let words: Vec<(&str, f64)> = scored
            .iter()
            .filter(|(t, w, _)| t.chars().count() >= 4 && *w >= MIN_WEIGHT * 10.0)
            .map(|(t, w, _)| (t.as_str(), *w))
            .collect();

        if words.len() >= 2 {
            let selected: Vec<&str> = words.iter().take(8).map(|(t, _)| *t).collect();
            return selected.join(" ");
        }

        // ── 3층: Char 조합 (1글자 단위 조합) ──
        let chars: Vec<(&str, f64)> = scored
            .iter()
            .filter(|(t, w, _)| t.chars().count() == 1 && *w >= MIN_WEIGHT * 10.0)
            .map(|(t, w, _)| (t.as_str(), *w))
            .collect();

        if chars.len() >= 2 {
            let assembled: String = chars.iter().take(15).map(|(t, _)| *t).collect();
            return assembled;
        }

        // ── 4층: Jamo 재조합 (1글자 자모를 모아서 한글 음절로 합침) ──
        let jamo_tokens: Vec<(char, f64)> = scored
            .iter()
            .filter(|(t, w, _)| t.chars().count() == 1 && *w >= MIN_WEIGHT * 10.0)
            .map(|(t, w, _)| (t.chars().next().unwrap(), *w))
            .collect();

        if jamo_tokens.len() >= 2 {
            let jamo_chars: Vec<char> = jamo_tokens.iter().take(30).map(|(c, _)| *c).collect();
            let assembled = tokenizer::compose_jamo(&jamo_chars);
            if !assembled.is_empty() {
                return assembled;
            }
        }

        // ── 폴백: 가중치 상위 토큰 연결 ──
        scored
            .iter()
            .take(3)
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    // ═══════════════════════════════════════════════
    //  틱 실행
    // ═══════════════════════════════════════════════

    fn run_tick(&mut self, emitted_tokens: &HashSet<String>, current_tick: u64) -> TickResult {
        let mut result = TickResult::default();

        let active: Vec<(NeuronId, f64)> = self
            .neurons
            .iter()
            .filter(|(_, n)| n.activation > 0.05)
            .map(|(id, n)| (id.clone(), n.activation))
            .collect();

        if active.is_empty() {
            return result;
        }

        // 병렬 뉴런 계산: 각 활성 뉴런의 compute_fires를 병렬 실행
        let neurons = &self.neurons;
        let store = &self.synapse_store;
        let fires_per_neuron: Vec<(NeuronId, Vec<_>)> = active
            .par_iter()
            .map(|(nid, _)| {
                let neuron = neurons.get(nid).unwrap();
                let fires = neuron.compute_fires(store, emitted_tokens);
                (nid.clone(), fires)
            })
            .collect();

        // 결과 병합 (순차)
        let mut pending: HashMap<NeuronId, f64> = HashMap::new();
        let mut fired: Vec<(NeuronId, SynapseId, NeuronId, f64)> = Vec::new();
        let mut fired_tokens: Vec<(String, SynapseId)> = Vec::new();
        let output_nid_set: HashSet<&NeuronId> = self.region_neurons
            .get(&RegionType::Output)
            .map(|ns| ns.iter().collect())
            .unwrap_or_default();

        for (nid, fires) in fires_per_neuron {
            for (sid, post_id, forward_val, token) in fires {
                // 최근 사용 억제: 최근 10회 이력 기반 차등 감소
                let penalty = self.compute_cooldown_penalty(&sid);
                let forward_val = forward_val * (1.0 - penalty);
                *pending.entry(post_id.clone()).or_insert(0.0) += forward_val;
                fired.push((nid.clone(), sid.clone(), post_id.clone(), forward_val));
                result.fired_synapses.push(sid.clone());

                // Output 뉴런으로 향하는 시냅스의 토큰만 큐에 등록
                // (Output은 스위치 — 다른 구역에서 Output으로 도달한 토큰만 출력 대상)
                if let Some(tok) = token {
                    if output_nid_set.contains(&post_id) {
                        fired_tokens.push((tok, sid));
                    }
                }

                if !result.activated_neurons.contains(&post_id) {
                    result.activated_neurons.push(post_id);
                }
            }
        }

        // 감쇠
        for neuron in self.neurons.values_mut() {
            neuron.activation *= DECAY_RATE;
        }

        // pre 뉴런의 spike 기록: 발화한 뉴런은 현재 틱에 spike
        for (nid, _) in &active {
            if let Some(n) = self.neurons.get_mut(nid) {
                if n.activation >= n.threshold * PASS_RATIO {
                    n.last_spike_tick = Some(current_tick);
                }
            }
        }

        // 수신
        for (nid, val) in &pending {
            if let Some(n) = self.neurons.get_mut(nid) {
                n.receive(*val);
            }
        }

        // post 뉴런 spike 기록: 수신 후 임계값 넘은 뉴런
        for (nid, _) in &pending {
            if let Some(n) = self.neurons.get_mut(nid) {
                if n.activation >= n.threshold * PASS_RATIO && n.last_spike_tick.is_none() {
                    n.last_spike_tick = Some(current_tick);
                }
            }
        }

        // 출력 채널: Output 뉴런 중 하나라도 on이면 채널 열림
        let output_nids = self.region_neurons.get(&RegionType::Output).cloned().unwrap_or_default();
        result.output_on = output_nids.iter().any(|nid| {
            self.neurons.get(nid).is_some_and(|n| n.activation >= n.threshold)
        });

        // 발화한 토큰은 항상 큐에 등록 (output_on 여부와 무관)
        result.fired_tokens_queue = fired_tokens;

        // Output on인 뉴런 기록
        if result.output_on {
            for nid in &output_nids {
                if let Some(n) = self.neurons.get(nid) {
                    if n.activation >= n.threshold {
                        result.emitted_output_neurons.push((nid.clone(), n.activation));
                    }
                }
            }
        }

        // STDP: Spike-Timing-Dependent Plasticity
        for (pre_id, sid, post_id, _) in &fired {
            let pre_spike = self.neurons.get(pre_id).and_then(|n| n.last_spike_tick);
            let post_spike = self.neurons.get(post_id).and_then(|n| n.last_spike_tick);

            if let (Some(t_pre), Some(t_post)) = (pre_spike, post_spike) {
                let dt = t_post as f64 - t_pre as f64;

                let delta = if dt > 0.0 {
                    // LTP: pre가 먼저 발화 → 인과관계 → 강화
                    STDP_A_PLUS * (-dt / STDP_TAU).exp()
                } else if dt < 0.0 {
                    // LTD: post가 먼저 발화 → 반인과 → 약화
                    -STDP_A_MINUS * (dt / STDP_TAU).exp()
                } else {
                    // 동시 발화 → 약한 강화
                    STDP_A_PLUS * 0.5
                };

                if let Some(syn) = self.synapse_store.get(sid) {
                    let new_weight = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                    self.synapse_store.update_weight(sid, new_weight);
                }
            } else {
                // spike 정보 없으면 기존 backward 방식 fallback
                let response = self.neurons.get(post_id).map(|n| n.activation).unwrap_or(0.0);
                let threshold = self.neurons.get(pre_id).map(|n| n.threshold).unwrap_or(0.5);

                if let Some(syn) = self.synapse_store.get(sid) {
                    let delta = BACKWARD_LR * (response - threshold);
                    let new_weight = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                    self.synapse_store.update_weight(sid, new_weight);
                }
            }
        }

        result.active_count = self.neurons.values().filter(|n| n.activation > 0.05).count();

        result
    }

    // ═══════════════════════════════════════════════
    //  피드백
    // ═══════════════════════════════════════════════

    pub fn feedback(&mut self, fire_id: u64, positive: bool, strength: f64) {
        let strength = strength.clamp(0.0, 1.0);

        let record = self.fire_records.iter_mut().find(|r| r.id == fire_id);
        let record = match record {
            Some(r) => r,
            None => {
                println!("  발화 #{fire_id} 를 찾을 수 없습니다.");
                return;
            }
        };

        if record.rewarded {
            println!("  발화 #{fire_id} 는 이미 피드백 처리되었습니다.");
            return;
        }

        let mut updated = 0;
        let output_sids = record.output_tokens.clone();
        let path_sids = record.fired_synapses.clone();

        // 출력 시냅스 조정
        for (_, sid, _) in &output_sids {
            if let Some(syn) = self.synapse_store.get(sid) {
                let delta = strength * FEEDBACK_LR;
                let new_weight = if positive {
                    (syn.weight + delta).min(MAX_WEIGHT)
                } else {
                    // 틀릴 때 출력 시냅스 강하게 약화 (1.5배)
                    (syn.weight - delta * 1.5).max(MIN_WEIGHT)
                };
                self.synapse_store.update_weight(sid, new_weight);
                updated += 1;
            }
        }

        // 경로 시냅스도 조정 (출력보다 약하게)
        if !positive {
            for sid in &path_sids {
                if let Some(syn) = self.synapse_store.get(sid) {
                    let delta = strength * FEEDBACK_LR * 0.5;
                    let new_weight = (syn.weight - delta).max(MIN_WEIGHT);
                    self.synapse_store.update_weight(sid, new_weight);
                    updated += 1;
                }
            }
        }

        record.rewarded = true;

        let label = if positive { "강화" } else { "약화" };
        println!("  [{label}] 발화 #{fire_id} | 시냅스 {updated}개 (강도: {strength:.1})");
    }

    /// 부분 피드백: 응답의 각 단어별로 관련성 점수를 받아 개별 시냅스 조정
    /// token_scores: [("단어", 점수), ...] 점수 > 0 강화, < 0 약화
    pub fn feedback_partial(&mut self, fire_id: u64, token_scores: &[(String, f64)]) {
        let record = self.fire_records.iter_mut().find(|r| r.id == fire_id);
        let record = match record {
            Some(r) => r,
            None => {
                println!("  발화 #{fire_id} 를 찾을 수 없습니다.");
                return;
            }
        };

        if record.rewarded {
            return;
        }

        let output_sids = record.output_tokens.clone();
        let mut reinforced = 0;
        let mut weakened = 0;

        for (tok, sid, _) in &output_sids {
            if let Some(syn) = self.synapse_store.get(sid) {
                // 이 시냅스의 토큰이 token_scores에 매칭되는지 확인
                let score = if let Some(ref syn_tok) = syn.token {
                    token_scores.iter()
                        .find(|(t, _)| syn_tok.contains(t.as_str()) || t.contains(syn_tok.as_str()) || tok.contains(t.as_str()))
                        .map(|(_, s)| *s)
                        .unwrap_or(0.0)
                } else {
                    // Original 토큰: 전체 점수 평균 적용
                    let avg: f64 = if token_scores.is_empty() { 0.0 }
                        else { token_scores.iter().map(|(_, s)| s).sum::<f64>() / token_scores.len() as f64 };
                    avg
                };

                if score.abs() < 0.01 { continue; }

                let delta = score.abs() * FEEDBACK_LR;
                let new_weight = if score > 0.0 {
                    reinforced += 1;
                    (syn.weight + delta).min(MAX_WEIGHT)
                } else {
                    weakened += 1;
                    (syn.weight - delta).max(MIN_WEIGHT)
                };
                self.synapse_store.update_weight(sid, new_weight);
            }
        }

        record.rewarded = true;
        println!("  [부분피드백] 발화 #{fire_id} | +{reinforced} -{weakened}");
    }

    // ═══════════════════════════════════════════════
    //  teach: 입력-응답 쌍 학습
    // ═══════════════════════════════════════════════

    pub fn teach(&mut self, input: &str, target: &str) {
        let input_tokens = tokenizer::tokenize(input);
        let target_tokens = tokenizer::tokenize(target);

        self.store_input_tokens(&input_tokens);
        self.store_response_tokens(&input_tokens, &target_tokens);

        let (output, output_sids, path_length) = self.fire_silent(input);

        let meaning = token_match_score(&output, target);
        let efficiency = path_efficiency_score(path_length);
        let score = meaning * 0.8 + efficiency * 0.2;

        for (_, sid, _) in &output_sids {
            if let Some(syn) = self.synapse_store.get(sid) {
                let delta = if meaning >= 0.1 {
                    score * FEEDBACK_LR
                } else {
                    -(1.0 - score) * FEEDBACK_LR * 0.5
                };
                let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                self.synapse_store.update_weight(sid, new_w);
            }
        }

        println!(
            "  [teach] \"{}\" → \"{}\" | 의미 {:.0}% 효율 {:.0}% 종합 {:.0}% (경로 {})",
            input, output, meaning * 100.0, efficiency * 100.0, score * 100.0, path_length
        );
        println!("  [teach] 목표: \"{}\"", target);
    }

    // ═══════════════════════════════════════════════
    //  자동 학습
    // ═══════════════════════════════════════════════

    pub fn train(&mut self, pairs: &[(&str, &str)], duration_secs: u64) {
        let start = Instant::now();
        let mut rng = rand::rng();
        use rand::prelude::*;

        let mut count: u64 = 0;
        let mut positive_count: u64 = 0;
        let mut negative_count: u64 = 0;
        let mut best_score: f64 = 0.0;
        let mut best_output = String::new();
        let mut best_input = String::new();
        let mut best_target = String::new();
        let mut last_report = 0u64;

        let mut response_map: HashMap<String, Vec<String>> = HashMap::new();
        for &(input, target) in pairs {
            response_map.entry(input.to_string()).or_default().push(target.to_string());
        }
        for &(input, target) in pairs {
            if input != target {
                response_map.entry(target.to_string()).or_default().push(input.to_string());
            }
        }

        let all_inputs: Vec<String> = response_map.keys().cloned().collect();

        println!("  학습 시작 ({duration_secs}초)...");
        println!("  학습 입력: {}개, 총 연결: {}개\n", all_inputs.len(), pairs.len() * 2);

        println!("  [초기 등록] 모든 학습 쌍 저장 중...");
        for (&ref input, targets) in &response_map {
            let input_tokens = tokenizer::tokenize(input);
            self.store_input_tokens(&input_tokens);
            for target in targets {
                let target_tokens = tokenizer::tokenize(target);
                self.store_response_tokens(&input_tokens, &target_tokens);
            }
        }
        println!("  [초기 등록] 완료\n");

        while start.elapsed().as_secs() < duration_secs {
            let input = &all_inputs[rng.random_range(0..all_inputs.len())];
            let targets = &response_map[input];
            let target = &targets[rng.random_range(0..targets.len())];

            let (output, output_sids, path_length) = self.fire_silent(input);
            count += 1;

            let meaning = token_match_score(&output, target);
            let efficiency = path_efficiency_score(path_length);
            let score = meaning * 0.8 + efficiency * 0.2;

            for (_, sid, _) in &output_sids {
                if let Some(syn) = self.synapse_store.get(sid) {
                    let delta = if meaning >= 0.1 {
                        score * FEEDBACK_LR
                    } else {
                        -(1.0 - score) * FEEDBACK_LR * 0.5
                    };
                    let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                    self.synapse_store.update_weight(sid, new_w);
                }
            }

            if meaning >= 0.1 {
                positive_count += 1;
            } else {
                negative_count += 1;
            }

            if score > best_score {
                best_score = score;
                best_output = output;
                best_input = input.to_string();
                best_target = target.to_string();
            }

            let elapsed = start.elapsed().as_secs();
            if elapsed >= last_report + 10 {
                last_report = elapsed;
                print!(
                    "\r  [{:>3}s] 발화 {:<7} +{:<6} -{:<6} 최고 {:.0}% 시냅스 {} 토큰어휘 {}",
                    elapsed, count, positive_count, negative_count,
                    best_score * 100.0,
                    self.synapse_store.count(),
                    self.synapse_store.token_index_count(),
                );
                io::stdout().flush().ok();
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!("\n\n=== 학습 완료 ({:.1}초) ===", elapsed);
        println!("  총 발화: {}회 | 강화: {} | 약화: {}", count, positive_count, negative_count);
        if !best_output.is_empty() {
            println!(
                "  최고: \"{}\" → \"{}\" (목표: \"{}\", 일치 {:.0}%)",
                best_input, best_output, best_target, best_score * 100.0
            );
        }
        self.print_summary();
    }

    /// 조용한 발화 (학습용)
    fn fire_silent(&mut self, input: &str) -> (String, Vec<(String, SynapseId, u64)>, usize) {
        self.reset_activations();

        let tokens = tokenizer::tokenize(input);
        self.activate_all_tokens(&tokens);

        let mut all_output_tokens: Vec<(String, SynapseId, u64)> = Vec::new();
        let mut all_fired: Vec<SynapseId> = Vec::new();
        let mut emitted_tokens: HashSet<String> = HashSet::new();
        let mut neurons_visited: Vec<NeuronId> = Vec::new();
        let mut neuron_fire_ticks: Vec<(NeuronId, u64)> = Vec::new();
        let mut token_queue: VecDeque<(String, SynapseId, u64)> = VecDeque::new();

        for tick in 0..MAX_TICKS {
            if self.shutdown.load(Ordering::Relaxed) { break; }
            let result = self.run_tick(&emitted_tokens, tick);

            all_fired.extend(result.fired_synapses.iter().cloned());
            neurons_visited.extend(result.activated_neurons.iter().cloned());
            for nid in &result.activated_neurons {
                neuron_fire_ticks.push((nid.clone(), tick));
            }

            for (tok, sid) in &result.fired_tokens_queue {
                token_queue.push_back((tok.clone(), sid.clone(), tick));
            }

            if result.output_on {
                while let Some((tok, sid, t)) = token_queue.pop_front() {
                    if !emitted_tokens.contains(&tok) {
                        emitted_tokens.insert(tok.clone());
                        all_output_tokens.push((tok, sid, t));
                    }
                }
                for (nid, _) in &result.emitted_output_neurons {
                    if let Some(n) = self.neurons.get_mut(nid) {
                        n.activation *= OUTPUT_EMISSION_DECAY;
                    }
                }
            }

            if result.active_count == 0 {
                break;
            }
        }

        // 해마: 패턴 + co-firing 기록
        self.hippocampus.record(&neurons_visited);
        self.hippocampus.record_cofiring(&neuron_fire_ticks);
        let patterns = self.hippocampus.maybe_consolidate();
        for (pattern, freq) in patterns {
            self.store_memory_silent(pattern, freq);
        }
        let cofire_pairs = self.hippocampus.consolidate_cofiring();
        if !cofire_pairs.is_empty() {
            self.connect_cofiring_pairs(&cofire_pairs);
        }

        let path_length = all_fired.len();
        let (output, used_output_tokens) = self.assemble_output(&all_output_tokens);
        (output, used_output_tokens, path_length)
    }

    // ═══════════════════════════════════════════════
    //  기억 저장
    // ═══════════════════════════════════════════════

    fn store_memory(&mut self, pattern: Vec<NeuronId>, frequency: u64) {
        let storage_neurons = match self.region_neurons.get(&RegionType::Storage) {
            Some(ns) => ns.clone(),
            None => return,
        };
        let mut rng = rand::rng();
        use rand::prelude::*;
        let neuron_id = storage_neurons.choose(&mut rng).unwrap().clone();

        let emotion_neurons = self.region_neurons.get(&RegionType::Emotion)
            .cloned().unwrap_or_default();
        if emotion_neurons.is_empty() {
            return;
        }
        let target_id = emotion_neurons.choose(&mut rng).unwrap().clone();

        let weight = (INITIAL_WEIGHT + frequency as f64 * 0.02).min(MAX_WEIGHT);
        let memory = PathMemory {
            pattern: pattern.clone(),
            frequency,
        };
        let neuron = self.neurons.get_mut(&neuron_id).unwrap();
        neuron.create_synapse(&self.synapse_store, target_id, weight, None, Some(memory));

        let short_ids: Vec<String> = pattern.iter().map(|id| id[..8].to_string()).collect();
        println!(
            "    패턴: {} (빈도 {}) → 뉴런 {:.8}",
            short_ids.join("→"),
            frequency,
            neuron_id
        );
    }

    fn store_memory_silent(&mut self, pattern: Vec<NeuronId>, frequency: u64) {
        let storage_neurons = match self.region_neurons.get(&RegionType::Storage) {
            Some(ns) => ns.clone(),
            None => return,
        };
        let mut rng = rand::rng();
        use rand::prelude::*;
        let neuron_id = storage_neurons.choose(&mut rng).unwrap().clone();

        let emotion_neurons = self.region_neurons.get(&RegionType::Emotion)
            .cloned().unwrap_or_default();
        if emotion_neurons.is_empty() {
            return;
        }
        let target_id = emotion_neurons.choose(&mut rng).unwrap().clone();

        let memory = PathMemory { pattern, frequency };
        let weight = (INITIAL_WEIGHT + frequency as f64 * 0.02).min(MAX_WEIGHT);
        let neuron = self.neurons.get_mut(&neuron_id).unwrap();
        neuron.create_synapse(&self.synapse_store, target_id, weight, None, Some(memory));
    }

    // ═══════════════════════════════════════════════
    //  상태 출력
    // ═══════════════════════════════════════════════

    // ═══════════════════════════════════════════════
    //  API 메서드
    // ═══════════════════════════════════════════════

    pub fn get_last_output(&self) -> String {
        self.fire_records.last().map(|r| r.output.clone()).unwrap_or_default()
    }

    pub fn get_last_path_length(&self) -> usize {
        self.fire_records.last().map(|r| r.path_length).unwrap_or(0)
    }

    /// 마지막 발화의 출력 토큰 상세 정보 (실제 output에 포함된 토큰만)
    /// 반환: (토큰, 타입명, 가중치, 시냅스ID)
    pub fn get_last_output_tokens(&self) -> Vec<(String, usize, f64, String)> {
        let record = match self.fire_records.last() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let mut result = Vec::new();
        for (tok, sid, _tick) in &record.output_tokens {
            if let Some(syn) = self.synapse_store.get(sid) {
                let char_len = tok.chars().count();
                result.push((tok.clone(), char_len, syn.weight, sid.clone()));
            }
        }
        result
    }

    pub fn teach_api(&mut self, input: &str, target: &str) -> serde_json::Value {
        let input_tokens = tokenizer::tokenize(input);
        let target_tokens = tokenizer::tokenize(target);

        self.store_input_tokens(&input_tokens);
        self.store_response_tokens(&input_tokens, &target_tokens);

        let (output, output_sids, path_length) = self.fire_silent(input);

        let meaning = token_match_score(&output, target);
        let efficiency = path_efficiency_score(path_length);
        let score = meaning * 0.8 + efficiency * 0.2;

        for (_, sid, _) in &output_sids {
            if let Some(syn) = self.synapse_store.get(sid) {
                let delta = if meaning >= 0.1 {
                    score * FEEDBACK_LR
                } else {
                    -(1.0 - score) * FEEDBACK_LR * 0.5
                };
                let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                self.synapse_store.update_weight(sid, new_w);
            }
        }

        serde_json::json!({
            "output": output,
            "target": target,
            "semantic_score": meaning,
            "efficiency_score": efficiency,
            "combined_score": score,
            "path_length": path_length,
        })
    }

    pub fn get_status(&self) -> serde_json::Value {
        serde_json::json!({
            "neurons": self.neurons.len(),
            "synapses": self.synapse_store.count(),
            "cached": self.synapse_store.cached_count(),
            "token_vocab": self.synapse_store.token_index_count(),
            "fire_count": self.hippocampus.fire_count(),
            "patterns": self.hippocampus.pattern_count(),
        })
    }

    pub fn neuron_count(&self) -> usize { self.neurons.len() }
    pub fn synapse_count(&self) -> usize { self.synapse_store.count() }
    pub fn cached_count(&self) -> usize { self.synapse_store.cached_count() }
    pub fn fire_count(&self) -> u64 { self.hippocampus.fire_count() as u64 }
    pub fn pattern_count(&self) -> usize { self.hippocampus.pattern_count() }
    pub fn token_vocab_count(&self) -> usize { self.synapse_store.token_index_count() }

    pub fn print_summary(&self) {
        use RegionType::*;
        for region in &[Input, Emotion, Reason, Storage, Output] {
            let neuron_count = self
                .region_neurons
                .get(region)
                .map(|n| n.len())
                .unwrap_or(0);
            let synapse_count: usize = self
                .region_neurons
                .get(region)
                .map(|neurons| {
                    neurons
                        .iter()
                        .filter_map(|n| self.neurons.get(n))
                        .map(|n| n.outgoing.len())
                        .sum()
                })
                .unwrap_or(0);

            if *region == Storage {
                let all_sids: Vec<SynapseId> = self
                    .region_neurons
                    .get(region)
                    .map(|neurons| {
                        neurons
                            .iter()
                            .filter_map(|n| self.neurons.get(n))
                            .flat_map(|n| n.outgoing.iter().cloned())
                            .collect()
                    })
                    .unwrap_or_default();
                let memory_count = self.synapse_store.cached_memory_count(&all_sids);
                println!(
                    "  [{region}] 뉴런: {neuron_count}, 시냅스: {synapse_count} (기억: {memory_count})"
                );
            } else {
                println!("  [{region}] 뉴런: {neuron_count}, 시냅스: {synapse_count}");
            }
        }
        println!(
            "  전체 시냅스: {} (캐시: {}, 토큰어휘: {})",
            self.synapse_store.count(),
            self.synapse_store.cached_count(),
            self.synapse_store.token_index_count(),
        );
        println!(
            "  해마: 발화 {}회, 추적 패턴 {}개",
            self.hippocampus.fire_count(),
            self.hippocampus.pattern_count()
        );
    }
}

// ── 틱 결과 ──

#[derive(Default)]
struct TickResult {
    fired_synapses: Vec<SynapseId>,
    fired_tokens_queue: Vec<(String, SynapseId)>, // 이번 틱에 발화한 토큰 (큐에 쌓임)
    output_on: bool,                               // Output 채널이 열려있는지
    emitted_output_neurons: Vec<(NeuronId, f64)>,  // on 상태인 Output 뉴런
    activated_neurons: Vec<NeuronId>,
    active_count: usize,
}

// ── 유틸 ──

/// 토큰이 실제 자모(ㄱ~ㅎ, ㅏ~ㅣ)인지 확인
/// 한글 완성 음절(가~힣)이나 다른 문자는 false
fn is_jamo_token(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() != 1 {
        return false;
    }
    let code = chars[0] as u32;
    // 한글 호환 자모: ㄱ(0x3131) ~ ㅣ(0x3163)
    (0x3131..=0x3163).contains(&code)
}

fn path_efficiency_score(path_length: usize) -> f64 {
    1.0 / (1.0 + path_length as f64 / 10.0)
}

fn token_match_score(output: &str, target: &str) -> f64 {
    if output.is_empty() || target.is_empty() {
        return 0.0;
    }

    if output == target {
        return 1.0;
    }

    let out_words: HashSet<&str> = output.split_whitespace().collect();
    let tgt_words: HashSet<&str> = target.split_whitespace().collect();

    if tgt_words.is_empty() {
        return 0.0;
    }

    let intersection = out_words.intersection(&tgt_words).count();
    let union = out_words.union(&tgt_words).count();

    let jaccard = if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    };

    let contains_bonus = if output.contains(target) || target.contains(output) {
        0.3
    } else {
        0.0
    };

    (jaccard * 0.7 + contains_bonus).min(1.0)
}

impl Drop for Network {
    fn drop(&mut self) {
        self.save_state();
    }
}
