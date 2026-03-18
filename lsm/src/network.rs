use crate::fire_engine::{EngineNeuron, EngineSynapse, FireEngine};
use crate::hippocampus::{ConsolidationResult, HippoInput, HippoStats, HippocampusState};
use crate::neuron::{Neuron, NeuronId, PASS_RATIO};
use crate::region::RegionType;
use crate::synapse::{PathMemory, SynapseId, SynapseStore};
use crate::tokenizer::{self, hash_to_index, TextTokens};
use crossbeam::channel::{Receiver, Sender};
use crossbeam::queue::SegQueue;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

// ── 상수 ──

const INITIAL_WEIGHT: f64 = 0.5;
const DECAY_RATE: f64 = 0.5;           // 감쇠율: 낮을수록 신호가 빨리 약해짐
const BACKWARD_LR: f64 = 0.01;        // 틱마다 시냅스 weight 조정
const FEEDBACK_LR: f64 = 0.1;         // 피드백 시 weight 조정
const MAX_WEIGHT: f64 = 1.0;
const MIN_WEIGHT: f64 = 0.1;
const MAX_TICKS: u64 = 50;
const OUTPUT_EMISSION_DECAY: f64 = 0.3; // 출력 뉴런이 토큰 방출 후 activation에 곱하는 감쇠율
const RANDOM_SYNAPSE_WEIGHT: f64 = 0.1; // 초기 랜덤 시냅스 가중치 (낮게 시작)
const COFIRE_SYNAPSE_WEIGHT: f64 = 0.15; // co-firing으로 생성되는 시냅스 가중치

// ── 감정/이성 구역별 특성 ──
const EMOTION_THRESHOLD: f64 = 0.3;  // 감정: 낮은 임계값 (빠르게 반응)
const REASON_THRESHOLD: f64 = 0.5;   // 이성: 높은 임계값 (근거 필요)

// ── 상태 영속화 ──

#[derive(Serialize, Deserialize)]
struct NetworkState {
    neurons: HashMap<NeuronId, Neuron>,
    neuron_region: HashMap<NeuronId, RegionType>,
    region_neurons: Vec<(RegionType, Vec<NeuronId>)>,
    #[serde(default)]
    hippocampus: Option<HippocampusState>,
    next_fire_id: u64,
}

// ── 디버그 트리 ──

#[derive(Debug, Clone, Serialize)]
pub struct DebugNode {
    pub neuron_id: String,
    pub region: String,
    pub activation: f64,
    pub children: Vec<DebugEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugEdge {
    pub synapse_id: String,
    pub weight: f64,
    pub modifier: f64,
    pub forward: f64,
    pub token: Option<String>,
    pub target: DebugNode,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugTickInfo {
    pub tick: u64,
    pub active_neurons: usize,
    pub fired_synapses: usize,
    pub tokens_emitted: Vec<String>,
    pub propagations: Vec<DebugPropagation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugPropagation {
    pub from_neuron: String,
    pub from_region: String,
    pub synapse_id: String,
    pub weight: f64,
    pub modifier: f64,
    pub forward: f64,
    pub to_neuron: String,
    pub to_region: String,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FireDebugResult {
    pub fire_id: u64,
    pub input: String,
    pub output: String,
    pub total_ticks: u64,
    pub total_path: usize,
    pub elapsed_ms: u64,
    pub ticks: Vec<DebugTickInfo>,
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

// ── 축삭 발아 (Axonal Sprouting) ──
const SPROUT_RATE: f64 = 0.01;        // 기본 발아 확률 (매우 낮게)
const SPROUT_SIGMA: f64 = 3.0;        // 가우시안 반경 (약 3칸 이내 주요 범위)
const SPROUT_WEIGHT: f64 = 0.05;      // 새 시냅스 초기 weight
const MAX_SPROUT_PER_TICK: usize = 2; // 틱당 최대 발아 수 (폭발 방지)
const SPROUT_SEARCH_RADIUS: usize = 5; // 탐색 반경 (성능용)

/// fire 후처리 지연 실행용 데이터
struct PostProcessData {
    neurons_activated: Vec<NeuronId>,
    neuron_fire_ticks: Vec<(NeuronId, u64)>,
    all_output_tokens: Vec<(String, SynapseId, u64)>,
    all_fired: Vec<SynapseId>,
    used_output_tokens: Vec<(String, SynapseId, u64)>,
    fire_id: u64,
}

pub struct Network {
    neurons: HashMap<NeuronId, Neuron>,
    neuron_region: HashMap<NeuronId, RegionType>,
    region_neurons: HashMap<RegionType, Vec<NeuronId>>,
    synapse_store: SynapseStore,
    fire_records: Vec<FireRecord>,
    next_fire_id: u64,
    // 해마 스레드 통신
    hippo_tx: Sender<HippoInput>,
    consolidation_rx: Receiver<ConsolidationResult>,
    hippo_stats: Arc<HippoStats>,
    shutdown: Arc<AtomicBool>,
    // 최근 사용 억제: 시냅스ID → 최근 N회 fire에서 사용된 이력 (앞=최근, 뒤=오래됨)
    cooldown_history: HashMap<SynapseId, VecDeque<bool>>,
    fire_generation: u64, // 현재 fire 세대 (이력 관리용)
    // 현재 fire의 입력 토큰 (compose_response에서 관련성 판단용)
    current_input_tokens: HashSet<String>,
    // 후처리 지연 실행: fire 응답 후 별도로 처리
    pending_post_process: Option<PostProcessData>,
}

impl Network {
    /// 시냅스 weight 업데이트 + 뉴런 캐시 동기화
    fn sync_weight(&mut self, pre_neuron: &str, sid: &str, new_weight: f64) {
        self.synapse_store.update_weight(sid, new_weight);
        if let Some(neuron) = self.neurons.get_mut(pre_neuron) {
            if let Some(os) = neuron.outgoing_cache.iter_mut().find(|o| o.id == sid) {
                os.weight = new_weight;
            }
        }
    }

    /// 시냅스 modifier 업데이트 + 뉴런 캐시 동기화
    /// Output 구역으로 향하는 시냅스는 modifier를 항상 0으로 유지
    fn sync_modifier(&mut self, pre_neuron: &str, sid: &str, new_modifier: f64) {
        // Output 구역 시냅스는 modifier 0 강제
        let is_output_target = self.synapse_store.get(sid)
            .and_then(|s| self.neuron_region.get(&s.post_neuron))
            .map(|r| *r == RegionType::Output)
            .unwrap_or(false);

        let final_mod = if is_output_target { 0.0 } else { new_modifier };

        self.synapse_store.update_modifier(sid, final_mod);
        if let Some(neuron) = self.neurons.get_mut(pre_neuron) {
            if let Some(os) = neuron.outgoing_cache.iter_mut().find(|o| o.id == sid) {
                os.modifier = final_mod.clamp(-1.0, 1.0);
            }
        }
    }

    pub fn new(
        db_path: PathBuf,
        max_cached_synapses: usize,
        shutdown: Arc<AtomicBool>,
        hippo_tx: Sender<HippoInput>,
        consolidation_rx: Receiver<ConsolidationResult>,
        hippo_stats: Arc<HippoStats>,
    ) -> Self {
        Self {
            neurons: HashMap::new(),
            neuron_region: HashMap::new(),
            region_neurons: HashMap::new(),
            synapse_store: SynapseStore::new(db_path, max_cached_synapses),
            fire_records: Vec::new(),
            next_fire_id: 1,
            hippo_tx,
            consolidation_rx,
            hippo_stats,
            shutdown,
            cooldown_history: HashMap::new(),
            fire_generation: 0,
            current_input_tokens: HashSet::new(),
            pending_post_process: None,
        }
    }

    /// 상태 로드. 성공 시 (true, Option<HippocampusState>) 반환
    pub fn try_load_state(&mut self) -> (bool, Option<HippocampusState>) {
        let data = match self.synapse_store.load_network_state() {
            Some(d) => d,
            None => return (false, None),
        };
        let state: NetworkState = match serde_json::from_slice(&data) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  상태 파싱 실패 (구조 변경?): {e}");
                eprintln!("  → data/network.redb를 삭제하고 다시 시작하세요.");
                return (false, None);
            }
        };

        self.neurons = state.neurons;
        self.neuron_region = state.neuron_region;
        self.region_neurons = state.region_neurons.into_iter().collect();
        self.next_fire_id = state.next_fire_id;
        let hippo_state = state.hippocampus; // 해마 상태는 외부(서버)에서 스레드로 전달

        // 레거시 데이터 마이그레이션: outgoing(ID만) → outgoing_cache(weight/modifier 포함)
        // + 기존 outgoing_cache에 post_neuron이 비어있으면 채우기
        for neuron in self.neurons.values_mut() {
            neuron.migrate_outgoing(&self.synapse_store);
            neuron.fill_missing_post_neurons(&self.synapse_store);
        }

        // Output 구역 시냅스 modifier → 0 강제 (메모리만, DB는 save 시 반영)
        let output_nids: HashSet<NeuronId> = self.region_neurons
            .get(&RegionType::Output)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        for neuron in self.neurons.values_mut() {
            for os in &mut neuron.outgoing_cache {
                if output_nids.contains(&os.post_neuron) && os.modifier != 0.0 {
                    os.modifier = 0.0;
                }
            }
        }

        // 시냅스 캐시 워밍업 (token_index 구축)
        self.synapse_store.warm_cache();

        let pattern_count = hippo_state.as_ref().map(|s| s.pattern_counts.len()).unwrap_or(0);
        println!(
            "  [로드] 뉴런 {}개, 구역 {}개, 해마 패턴 {}개 복원",
            self.neurons.len(),
            self.region_neurons.len(),
            pattern_count,
        );
        (true, hippo_state)
    }

    /// 경량 저장: 네트워크 상태 + dirty 시냅스만 (자동저장용, 블로킹 최소화)
    pub fn save_state_light(&self) {
        let hippo_state = {
            let (tx, rx) = crossbeam::channel::bounded(1);
            if self.hippo_tx.send(HippoInput::ExportState(tx)).is_ok() {
                rx.recv_timeout(std::time::Duration::from_secs(5)).ok()
            } else {
                None
            }
        };

        let state = NetworkState {
            neurons: self.neurons.clone(),
            neuron_region: self.neuron_region.clone(),
            region_neurons: self.region_neurons.iter().map(|(k, v)| (*k, v.clone())).collect(),
            hippocampus: hippo_state,
            next_fire_id: self.next_fire_id,
        };
        match serde_json::to_vec(&state) {
            Ok(bytes) => {
                self.synapse_store.save_network_state(&bytes);
                let _ = writeln!(
                    std::io::stderr(),
                    "  [경량저장] 뉴런 {}개 → DB 저장 완료",
                    self.neurons.len(),
                );
            }
            Err(e) => eprintln!("  상태 직렬화 실패: {e}"),
        }
        // dirty만 flush (캐시 전체 동기화 + prune 생략)
        self.synapse_store.flush_dirty();
    }

    /// 전체 저장: 캐시 동기화 + flush + prune (수동 저장/종료 시)
    pub fn save_state(&self) {
        let hippo_state = {
            let (tx, rx) = crossbeam::channel::bounded(1);
            if self.hippo_tx.send(HippoInput::ExportState(tx)).is_ok() {
                rx.recv_timeout(std::time::Duration::from_secs(5)).ok()
            } else {
                None
            }
        };

        let state = NetworkState {
            neurons: self.neurons.clone(),
            neuron_region: self.neuron_region.clone(),
            region_neurons: self.region_neurons.iter().map(|(k, v)| (*k, v.clone())).collect(),
            hippocampus: hippo_state,
            next_fire_id: self.next_fire_id,
        };
        match serde_json::to_vec(&state) {
            Ok(bytes) => {
                self.synapse_store.save_network_state(&bytes);
                let _ = writeln!(
                    std::io::stderr(),
                    "  [저장] 뉴런 {}개, 해마 패턴 {}개 → DB 저장 완료",
                    self.neurons.len(),
                    self.hippo_stats.pattern_count.load(Ordering::Relaxed),
                );
            }
            Err(e) => eprintln!("  상태 직렬화 실패: {e}"),
        }
        // 뉴런 캐시 → synapse_store 동기화 (STDP가 캐시만 업데이트하므로)
        for neuron in self.neurons.values() {
            for os in &neuron.outgoing_cache {
                self.synapse_store.update_weight(&os.id, os.weight);
                self.synapse_store.update_modifier(&os.id, os.modifier);
            }
        }
        // dirty 시냅스 저장
        self.synapse_store.flush_dirty();
        // DB 전체 pruning (약한/중복 시냅스 정리)
        let (removed, remaining, _removed_pairs) = self.synapse_store.prune_db(MIN_WEIGHT);
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
                let already = neuron.outgoing_cache.iter().any(|os| {
                    self.synapse_store.get(&os.id).is_some_and(|s| s.post_neuron == *dst)
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
                    let already = neuron.outgoing_cache.iter().any(|os| {
                        self.synapse_store.get(&os.id).is_some_and(|s| s.post_neuron == *dst)
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
            let existing_ab = neuron_a.outgoing_cache.iter().find(|os| {
                os.post_neuron == *nid_b
            }).map(|os| os.id.clone());

            if let Some(sid) = existing_ab {
                if let Some(syn) = self.synapse_store.get(&sid) {
                    let new_w = (syn.weight + bonus).min(MAX_WEIGHT);
                    let pre = nid_a.clone();
                    self.sync_weight(&pre, &sid, new_w);
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
                    let pre = src.clone();
                    self.sync_weight(&pre, sid, new_w);
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
                    let pre = syn.pre_neuron.clone();
                    self.sync_weight(&pre, sid, new_w);
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

        let fire_start = Instant::now();
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

        let tick_elapsed = fire_start.elapsed();
        let (output, used_output_tokens) = self.assemble_output(&all_output_tokens);
        let assemble_elapsed = fire_start.elapsed() - tick_elapsed;

        print!("[{fire_id}] {output}");
        println!();

        if output.is_empty() {
            println!("  (신호 소멸)");
        }

        eprintln!(
            "  [fire #{fire_id}] ticks={:?} assemble={:?} total_before_post={:?} fired={}",
            tick_elapsed, assemble_elapsed, fire_start.elapsed(), all_fired.len()
        );

        let path_length = all_fired.len();

        // 후처리는 pending에 저장 → 서버에서 응답 전송 후 run_pending_post_process() 호출
        self.pending_post_process = Some(PostProcessData {
            neurons_activated: neurons_activated.clone(),
            neuron_fire_ticks,
            all_output_tokens: all_output_tokens.clone(),
            all_fired: all_fired.clone(),
            used_output_tokens: used_output_tokens.clone(),
            fire_id,
        });

        self.fire_records.push(FireRecord {
            id: fire_id,
            fired_synapses: all_fired,
            output_tokens: used_output_tokens,
            neurons_visited: neurons_activated,
            output,
            path_length,
            rewarded: false,
        });

        if self.fire_records.len() > 50 {
            self.fire_records.drain(..self.fire_records.len() - 50);
        }

        fire_id
    }

    /// Atomic CAS 기반 병렬 발화 (run_tick_parallel 사용).
    /// 기존 fire()와 동일한 로직이지만 틱 내부에서 atomic 연산으로 병렬 처리.
    pub fn fire_parallel(&mut self, input: &str) -> u64 {
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;

        self.reset_activations();
        self.fire_generation += 1;

        let tokens = tokenizer::tokenize(input);
        self.store_input_tokens(&tokens);
        self.activate_all_tokens(&tokens);

        // 입력 토큰 저장
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

        // ── Atomic 배열 + 인덱스 매핑 구축 ──
        let idx_to_nid: Vec<NeuronId> = self.neurons.keys().cloned().collect();
        let nid_to_idx: HashMap<NeuronId, usize> = idx_to_nid
            .iter()
            .enumerate()
            .map(|(i, nid)| (nid.clone(), i))
            .collect();
        let atomic_activations: Vec<AtomicU64> = idx_to_nid
            .iter()
            .map(|nid| {
                let val = self.neurons.get(nid).map(|n| n.activation).unwrap_or(0.0);
                AtomicU64::new(val.to_bits())
            })
            .collect();

        let mut all_fired: Vec<SynapseId> = Vec::new();
        let mut all_output_tokens: Vec<(String, SynapseId, u64)> = Vec::new();
        let mut neurons_activated: Vec<NeuronId> = Vec::new();
        let mut neuron_fire_ticks: Vec<(NeuronId, u64)> = Vec::new();
        let mut emitted_tokens: HashSet<String> = HashSet::new();
        let mut token_queue: VecDeque<(String, SynapseId, u64)> = VecDeque::new();

        let fire_start = Instant::now();
        for tick in 0..MAX_TICKS {
            if self.shutdown.load(Ordering::Relaxed) { break; }
            let result = self.run_tick_parallel(
                &emitted_tokens, tick,
                &atomic_activations, &nid_to_idx, &idx_to_nid,
            );

            all_fired.extend(result.fired_synapses.iter().cloned());
            neurons_activated.extend(result.activated_neurons.iter().cloned());
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
                    // atomic 배열도 동기화
                    if let Some(&idx) = nid_to_idx.get(nid) {
                        let val = self.neurons.get(nid).map(|n| n.activation).unwrap_or(0.0);
                        atomic_activations[idx].store(val.to_bits(), Ordering::Relaxed);
                    }
                }
            }

            if result.active_count == 0 {
                break;
            }
        }

        let tick_elapsed = fire_start.elapsed();
        let (output, used_output_tokens) = self.assemble_output(&all_output_tokens);
        let assemble_elapsed = fire_start.elapsed() - tick_elapsed;

        print!("[{fire_id}] {output}");
        println!();

        if output.is_empty() {
            println!("  (신호 소멸)");
        }

        eprintln!(
            "  [fire_parallel #{fire_id}] ticks={:?} assemble={:?} total_before_post={:?} fired={}",
            tick_elapsed, assemble_elapsed, fire_start.elapsed(), all_fired.len()
        );

        let path_length = all_fired.len();

        self.pending_post_process = Some(PostProcessData {
            neurons_activated: neurons_activated.clone(),
            neuron_fire_ticks,
            all_output_tokens: all_output_tokens.clone(),
            all_fired: all_fired.clone(),
            used_output_tokens: used_output_tokens.clone(),
            fire_id,
        });

        self.fire_records.push(FireRecord {
            id: fire_id,
            fired_synapses: all_fired,
            output_tokens: used_output_tokens,
            neurons_visited: neurons_activated,
            output,
            path_length,
            rewarded: false,
        });

        // 최근 50개만 유지 (consolidate_path_modifiers 폭발 방지)
        if self.fire_records.len() > 50 {
            self.fire_records.drain(..self.fire_records.len() - 50);
        }

        fire_id
    }

    /// 디버그 발화: 틱별 전파 경로를 리스트로 기록
    pub fn fire_debug(&mut self, input: &str) -> FireDebugResult {
        let start = std::time::Instant::now();
        let fire_id = self.next_fire_id;
        self.next_fire_id += 1;

        self.reset_activations();
        self.fire_generation += 1;

        let tokens = tokenizer::tokenize(input);
        self.store_input_tokens(&tokens);
        self.activate_all_tokens(&tokens);

        self.current_input_tokens.clear();
        self.current_input_tokens.insert(tokens.original.clone());
        for w in &tokens.words { self.current_input_tokens.insert(w.clone()); }
        for b in &tokens.bigrams { self.current_input_tokens.insert(b.clone()); }
        for ch in &tokens.chars { self.current_input_tokens.insert(ch.clone()); }
        for cho in &tokens.jamo { self.current_input_tokens.insert(cho.clone()); }

        let mut all_output_tokens: Vec<(String, SynapseId, u64)> = Vec::new();
        let mut emitted_tokens: HashSet<String> = HashSet::new();
        let mut token_queue: VecDeque<(String, SynapseId, u64)> = VecDeque::new();
        let mut debug_ticks: Vec<DebugTickInfo> = Vec::new();
        let mut total_path: usize = 0;

        for tick in 0..MAX_TICKS {
            if self.shutdown.load(Ordering::Relaxed) { break; }

            // 활성 뉴런 수집
            let active: Vec<(NeuronId, f64)> = self.neurons.iter()
                .filter(|(_, n)| n.activation > 0.05)
                .map(|(id, n)| (id.clone(), n.activation))
                .collect();

            if active.is_empty() { break; }

            let result = self.run_tick(&emitted_tokens, tick);

            // 전파 정보 수집
            let mut propagations: Vec<DebugPropagation> = Vec::new();
            // run_tick의 fired 정보를 재구성: active 뉴런의 outgoing에서
            for (nid, _) in &active {
                if let Some(neuron) = self.neurons.get(nid) {
                    let from_region = self.neuron_region.get(nid)
                        .map(|r| format!("{:?}", r)).unwrap_or("?".into());
                    for os in &neuron.outgoing_cache {
                        let forward = neuron.activation * os.weight + os.modifier;
                        if forward <= 0.0 { continue; }
                        // 이 시냅스가 실제 fired에 포함되었는지 확인
                        if result.fired_synapses.contains(&os.id) {
                            let to_region = self.neuron_region.get(&os.post_neuron)
                                .map(|r| format!("{:?}", r)).unwrap_or("?".into());
                            propagations.push(DebugPropagation {
                                from_neuron: nid[..8].to_string(),
                                from_region: from_region.clone(),
                                synapse_id: os.id[..8].to_string(),
                                weight: (os.weight * 100.0).round() / 100.0,
                                modifier: (os.modifier * 100.0).round() / 100.0,
                                forward: (forward * 100.0).round() / 100.0,
                                to_neuron: os.post_neuron[..8].to_string(),
                                to_region,
                                token: os.token.clone(),
                            });
                        }
                    }
                }
            }

            let mut tick_tokens: Vec<String> = Vec::new();
            for (tok, sid) in &result.fired_tokens_queue {
                token_queue.push_back((tok.clone(), sid.clone(), tick));
            }
            if result.output_on {
                while let Some((tok, sid, t)) = token_queue.pop_front() {
                    if !emitted_tokens.contains(&tok) {
                        emitted_tokens.insert(tok.clone());
                        tick_tokens.push(tok.clone());
                        all_output_tokens.push((tok, sid, t));
                    }
                }
            }

            total_path += result.fired_synapses.len();

            debug_ticks.push(DebugTickInfo {
                tick,
                active_neurons: active.len(),
                fired_synapses: result.fired_synapses.len(),
                tokens_emitted: tick_tokens,
                propagations,
            });

            if result.active_count == 0 { break; }
        }

        let (output, _) = self.assemble_output(&all_output_tokens);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        FireDebugResult {
            fire_id,
            input: input.to_string(),
            output,
            total_ticks: debug_ticks.len() as u64,
            total_path,
            elapsed_ms,
            ticks: debug_ticks,
        }
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
        let t0 = Instant::now();

        let active: Vec<(NeuronId, f64)> = self
            .neurons
            .iter()
            .filter(|(_, n)| n.activation > 0.05)
            .map(|(id, n)| (id.clone(), n.activation))
            .collect();

        if active.is_empty() {
            return result;
        }

        let t1 = Instant::now();

        // 병렬 뉴런 계산: 각 활성 뉴런의 compute_fires를 병렬 실행
        let neurons = &self.neurons;
        let fires_per_neuron: Vec<(NeuronId, Vec<_>)> = active
            .par_iter()
            .map(|(nid, _)| {
                let neuron = neurons.get(nid).unwrap();
                let fires = neuron.compute_fires(emitted_tokens);
                (nid.clone(), fires)
            })
            .collect();
        let t2 = Instant::now();

        // 결과 병합 (순차)
        let mut pending: HashMap<NeuronId, f64> = HashMap::new();
        // (pre_neuron, synapse_id, post_neuron, forward, current_weight)
        let mut fired: Vec<(NeuronId, SynapseId, NeuronId, f64, f64)> = Vec::new();
        let mut fired_tokens: Vec<(String, SynapseId)> = Vec::new();
        let output_nid_set: HashSet<&NeuronId> = self.region_neurons
            .get(&RegionType::Output)
            .map(|ns| ns.iter().collect())
            .unwrap_or_default();

        for (nid, fires) in fires_per_neuron {
            // compute_fires returns: (sid, post_id, forward, token)
            // outgoing_cache에서 weight 가져오기
            let neuron = self.neurons.get(&nid).unwrap();
            for (sid, post_id, forward_val, token) in fires {
                let current_weight = neuron.outgoing_cache.iter()
                    .find(|o| o.id == sid)
                    .map(|o| o.weight)
                    .unwrap_or(0.5);

                // 최근 사용 억제: 최근 10회 이력 기반 차등 감소
                let penalty = self.compute_cooldown_penalty(&sid);
                let forward_val = forward_val * (1.0 - penalty);
                *pending.entry(post_id.clone()).or_insert(0.0) += forward_val;
                fired.push((nid.clone(), sid.clone(), post_id.clone(), forward_val, current_weight));
                result.fired_synapses.push(sid.clone());

                // Output 뉴런으로 향하는 시냅스의 토큰만 큐에 등록
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

        let t3 = Instant::now();

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

        let t4 = Instant::now();

        // 축삭 발아: 활성 뉴런에서 근처 뉴런으로 확률적 시냅스 생성
        self.try_sprout(&active);
        let t4b = Instant::now();

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
        // synapse_store.get 없이 fired에 포함된 current_weight 사용
        let mut stdp_updates: Vec<(NeuronId, SynapseId, f64)> = Vec::new();
        for (pre_id, sid, post_id, _, current_weight) in &fired {
            let pre_spike = self.neurons.get(pre_id).and_then(|n| n.last_spike_tick);
            let post_spike = self.neurons.get(post_id).and_then(|n| n.last_spike_tick);

            if let (Some(t_pre), Some(t_post)) = (pre_spike, post_spike) {
                let dt = t_post as f64 - t_pre as f64;

                let delta = if dt > 0.0 {
                    STDP_A_PLUS * (-dt / STDP_TAU).exp()
                } else if dt < 0.0 {
                    -STDP_A_MINUS * (dt / STDP_TAU).exp()
                } else {
                    STDP_A_PLUS * 0.5
                };

                let new_weight = (current_weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                stdp_updates.push((pre_id.clone(), sid.clone(), new_weight));
            } else {
                let response = self.neurons.get(post_id).map(|n| n.activation).unwrap_or(0.0);
                let threshold = self.neurons.get(pre_id).map(|n| n.threshold).unwrap_or(0.5);

                let delta = BACKWARD_LR * (response - threshold);
                let new_weight = (current_weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                stdp_updates.push((pre_id.clone(), sid.clone(), new_weight));
            }
        }
        // STDP weight: 뉴런 캐시만 즉시 반영 (DB는 save 시 일괄)
        // pre_neuron별로 그룹핑하여 outgoing_cache 1회 순회로 다중 업데이트
        let mut grouped: HashMap<NeuronId, Vec<(SynapseId, f64)>> = HashMap::new();
        for (pre_id, sid, new_weight) in stdp_updates {
            grouped.entry(pre_id).or_default().push((sid, new_weight));
        }
        for (pre_id, updates) in grouped {
            if let Some(neuron) = self.neurons.get_mut(&pre_id) {
                for os in &mut neuron.outgoing_cache {
                    if let Some(pos) = updates.iter().position(|(sid, _)| *sid == os.id) {
                        os.weight = updates[pos].1;
                    }
                }
            }
        }

        result.active_count = self.neurons.values().filter(|n| n.activation > 0.05).count();

        let t5 = Instant::now();

        // 틱 성능 로그 (첫 5틱 + 매 50틱마다)
        if current_tick < 5 || current_tick % 50 == 0 {
            eprintln!(
                "  [tick {current_tick}] active={} fired={} | collect={:?} compute={:?} merge={:?} decay+recv={:?} sprout={:?} stdp={:?}",
                active.len(),
                result.fired_synapses.len(),
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t4b - t4,
                t5 - t4b,
            );
        }

        result
    }

    // ═══════════════════════════════════════════════
    //  병렬 틱 (Atomic CAS 기반)
    // ═══════════════════════════════════════════════

    /// Atomic CAS 기반 병렬 run_tick.
    /// 모든 활성 뉴런의 발화 계산 + 타겟 activation 업데이트를 동시에 수행.
    /// Mutex 없이 AtomicU64 (f64 bit-cast) + CAS loop 만 사용.
    fn run_tick_parallel(
        &mut self,
        emitted_tokens: &HashSet<String>,
        current_tick: u64,
        atomic_activations: &[AtomicU64],
        nid_to_idx: &HashMap<NeuronId, usize>,
        idx_to_nid: &[NeuronId],
    ) -> TickResult {
        let mut result = TickResult::default();
        let t0 = Instant::now();

        // 1. Atomic 배열에서 활성 뉴런 수집
        let active: Vec<(usize, NeuronId, f64)> = idx_to_nid
            .iter()
            .enumerate()
            .filter_map(|(idx, nid)| {
                let val = f64::from_bits(atomic_activations[idx].load(Ordering::Relaxed));
                if val > 0.05 {
                    Some((idx, nid.clone(), val))
                } else {
                    None
                }
            })
            .collect();

        if active.is_empty() {
            return result;
        }

        let t1 = Instant::now();

        // 2. Output 뉴런 집합 (인덱스 기반)
        let output_nid_set: HashSet<usize> = self.region_neurons
            .get(&RegionType::Output)
            .map(|ns| ns.iter().filter_map(|nid| nid_to_idx.get(nid).copied()).collect())
            .unwrap_or_default();

        // 3. 쿨다운 페널티를 사전 계산 (순차, 읽기 전용)
        //    key: SynapseId → penalty
        let cooldown_penalties: HashMap<SynapseId, f64> = self.cooldown_history
            .keys()
            .map(|sid| (sid.clone(), self.compute_cooldown_penalty(sid)))
            .collect();

        // 4. Atomic → neuron struct 동기화 (감쇠 전 값으로, compute_fires가 사용)
        for (idx, nid, _) in &active {
            let val = f64::from_bits(atomic_activations[*idx].load(Ordering::Relaxed));
            if let Some(n) = self.neurons.get_mut(nid) {
                n.activation = val;
            }
        }

        // 5. 감쇠 적용 (순차 버전과 동일: decay → receive 순서)
        //    atomic 배열만 감쇠, neuron struct는 감쇠 전 값 유지 (compute_fires용)
        //    이후 CAS add로 새 신호가 감쇠된 배열에 추가됨 → 새 신호는 감쇠 안됨
        for idx in 0..atomic_activations.len() {
            loop {
                let old_bits = atomic_activations[idx].load(Ordering::Relaxed);
                let old_val = f64::from_bits(old_bits);
                let new_val = old_val * DECAY_RATE;
                let new_bits = new_val.to_bits();
                if atomic_activations[idx]
                    .compare_exchange_weak(old_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }
        }

        let neurons = &self.neurons;

        // 6. 병렬 발화 + atomic activation 업데이트 + lock-free 결과 수집
        let fired_queue: SegQueue<(NeuronId, SynapseId, NeuronId, f64, f64)> = SegQueue::new();
        let fired_synapses_queue: SegQueue<SynapseId> = SegQueue::new();
        let fired_tokens_queue: SegQueue<(String, SynapseId)> = SegQueue::new();
        let activated_neurons_queue: SegQueue<NeuronId> = SegQueue::new();

        active.par_iter().for_each(|(_, nid, _)| {
            let neuron = match neurons.get(nid) {
                Some(n) => n,
                None => return,
            };

            let fires = neuron.compute_fires(emitted_tokens);

            for (sid, post_id, forward_val, token) in fires {
                let current_weight = neuron.outgoing_cache.iter()
                    .find(|o| o.id == sid)
                    .map(|o| o.weight)
                    .unwrap_or(0.5);

                // 쿨다운 페널티 적용
                let penalty = cooldown_penalties.get(&sid).copied().unwrap_or(0.0);
                let forward_val = forward_val * (1.0 - penalty);

                // Atomic CAS add: 타겟 뉴런 activation 업데이트
                if let Some(&target_idx) = nid_to_idx.get(&post_id) {
                    // CAS loop for atomic f64 add
                    loop {
                        let old_bits = atomic_activations[target_idx].load(Ordering::Relaxed);
                        let old_val = f64::from_bits(old_bits);
                        let new_val = (old_val + forward_val).min(1.0); // MAX_ACTIVATION
                        let new_bits = new_val.to_bits();
                        if atomic_activations[target_idx]
                            .compare_exchange_weak(
                                old_bits, new_bits,
                                Ordering::Relaxed, Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }

                // Lock-free 결과 수집
                fired_queue.push((nid.clone(), sid.clone(), post_id.clone(), forward_val, current_weight));
                fired_synapses_queue.push(sid.clone());
                activated_neurons_queue.push(post_id.clone());

                // Output 뉴런으로 향하는 시냅스의 토큰
                if let Some(tok) = token {
                    if let Some(&target_idx) = nid_to_idx.get(&post_id) {
                        if output_nid_set.contains(&target_idx) {
                            fired_tokens_queue.push((tok, sid));
                        }
                    }
                }
            }
        });

        let t2 = Instant::now();

        // 6. SegQueue → Vec 변환
        let mut fired: Vec<(NeuronId, SynapseId, NeuronId, f64, f64)> = Vec::new();
        while let Some(f) = fired_queue.pop() {
            fired.push(f);
        }
        while let Some(sid) = fired_synapses_queue.pop() {
            result.fired_synapses.push(sid);
        }
        let mut fired_tokens: Vec<(String, SynapseId)> = Vec::new();
        while let Some(ft) = fired_tokens_queue.pop() {
            fired_tokens.push(ft);
        }
        let mut activated_set: HashSet<NeuronId> = HashSet::new();
        while let Some(nid) = activated_neurons_queue.pop() {
            if activated_set.insert(nid.clone()) {
                result.activated_neurons.push(nid);
            }
        }

        let t3 = Instant::now();

        // 7. Atomic → neuron struct 동기화 (activation + spike 기록)
        //    감쇠는 이미 step 4에서 적용됨, 여기선 compute 결과만 반영
        for (idx, nid) in idx_to_nid.iter().enumerate() {
            let val = f64::from_bits(atomic_activations[idx].load(Ordering::Relaxed));
            if let Some(n) = self.neurons.get_mut(nid) {
                n.activation = val;
            }
        }

        // pre 뉴런 spike 기록
        for (_, nid, _) in &active {
            if let Some(n) = self.neurons.get_mut(nid) {
                if n.activation >= n.threshold * PASS_RATIO {
                    n.last_spike_tick = Some(current_tick);
                }
            }
        }

        // post 뉴런 spike 기록
        for nid in &result.activated_neurons {
            if let Some(n) = self.neurons.get_mut(nid) {
                if n.activation >= n.threshold * PASS_RATIO && n.last_spike_tick.is_none() {
                    n.last_spike_tick = Some(current_tick);
                }
            }
        }

        let t4 = Instant::now();

        // 9. 축삭 발아 (순차)
        let active_for_sprout: Vec<(NeuronId, f64)> = active.iter()
            .map(|(_, nid, act)| (nid.clone(), *act))
            .collect();
        self.try_sprout(&active_for_sprout);
        let t4b = Instant::now();

        // 10. 출력 채널
        let output_nids = self.region_neurons.get(&RegionType::Output).cloned().unwrap_or_default();
        result.output_on = output_nids.iter().any(|nid| {
            self.neurons.get(nid).is_some_and(|n| n.activation >= n.threshold)
        });
        result.fired_tokens_queue = fired_tokens;

        if result.output_on {
            for nid in &output_nids {
                if let Some(n) = self.neurons.get(nid) {
                    if n.activation >= n.threshold {
                        result.emitted_output_neurons.push((nid.clone(), n.activation));
                    }
                }
            }
        }

        // 11. STDP (순차 — 동일 로직)
        let mut stdp_updates: Vec<(NeuronId, SynapseId, f64)> = Vec::new();
        for (pre_id, sid, post_id, _, current_weight) in &fired {
            let pre_spike = self.neurons.get(pre_id).and_then(|n| n.last_spike_tick);
            let post_spike = self.neurons.get(post_id).and_then(|n| n.last_spike_tick);

            if let (Some(t_pre), Some(t_post)) = (pre_spike, post_spike) {
                let dt = t_post as f64 - t_pre as f64;
                let delta = if dt > 0.0 {
                    STDP_A_PLUS * (-dt / STDP_TAU).exp()
                } else if dt < 0.0 {
                    -STDP_A_MINUS * (dt / STDP_TAU).exp()
                } else {
                    STDP_A_PLUS * 0.5
                };
                let new_weight = (current_weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                stdp_updates.push((pre_id.clone(), sid.clone(), new_weight));
            } else {
                let response = self.neurons.get(post_id).map(|n| n.activation).unwrap_or(0.0);
                let threshold = self.neurons.get(pre_id).map(|n| n.threshold).unwrap_or(0.5);
                let delta = BACKWARD_LR * (response - threshold);
                let new_weight = (current_weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                stdp_updates.push((pre_id.clone(), sid.clone(), new_weight));
            }
        }

        // STDP weight 반영 (뉴런 캐시만)
        let mut grouped: HashMap<NeuronId, Vec<(SynapseId, f64)>> = HashMap::new();
        for (pre_id, sid, new_weight) in stdp_updates {
            grouped.entry(pre_id).or_default().push((sid, new_weight));
        }
        for (pre_id, updates) in grouped {
            if let Some(neuron) = self.neurons.get_mut(&pre_id) {
                for os in &mut neuron.outgoing_cache {
                    if let Some(pos) = updates.iter().position(|(sid, _)| *sid == os.id) {
                        os.weight = updates[pos].1;
                    }
                }
            }
        }

        result.active_count = self.neurons.values().filter(|n| n.activation > 0.05).count();

        // 12. Neuron struct → atomic 배열 재동기화 (STDP가 weight를 바꿨으므로 activation은 이미 동기화됨)
        //     sprout로 새 뉴런이 생길 수 있으므로 idx_to_nid에 있는 것만 동기화
        for (idx, nid) in idx_to_nid.iter().enumerate() {
            if let Some(n) = self.neurons.get(nid) {
                atomic_activations[idx].store(n.activation.to_bits(), Ordering::Relaxed);
            }
        }

        let t5 = Instant::now();

        if current_tick < 5 || current_tick % 50 == 0 {
            eprintln!(
                "  [par tick {current_tick}] active={} fired={} | collect={:?} compute+atomic={:?} drain={:?} decay+sync={:?} sprout={:?} stdp={:?}",
                active.len(),
                result.fired_synapses.len(),
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t4b - t4,
                t5 - t4b,
            );
        }

        result
    }

    // ═══════════════════════════════════════════════
    //  축삭 발아 (Axonal Sprouting)
    // ═══════════════════════════════════════════════

    /// 뉴런 ID → 구역 내 2D 좌표
    fn neuron_grid_pos(&self, nid: &NeuronId) -> Option<(usize, usize)> {
        let region = self.neuron_region.get(nid)?;
        let neurons = self.region_neurons.get(region)?;
        let idx = neurons.iter().position(|n| n == nid)?;
        let (cols, _) = region.grid_dims();
        Some((idx % cols, idx / cols))
    }

    /// 활성 뉴런에서 근처 뉴런으로 축삭 발아 시도
    fn try_sprout(&mut self, active_nids: &[(NeuronId, f64)]) {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut sprout_count = 0;

        // 활성 뉴런이 많으면 랜덤 샘플링 (최대 10개만 시도)
        let max_candidates = 10.min(active_nids.len());
        let candidates: Vec<&(NeuronId, f64)> = if active_nids.len() <= max_candidates {
            active_nids.iter().collect()
        } else {
            let mut indices: Vec<usize> = (0..active_nids.len()).collect();
            indices.partial_shuffle(&mut rng, max_candidates);
            indices[..max_candidates].iter().map(|&i| &active_nids[i]).collect()
        };

        for (nid, _) in candidates {
            if sprout_count >= MAX_SPROUT_PER_TICK {
                break;
            }

            let src_region = match self.neuron_region.get(nid) {
                Some(r) => *r,
                None => continue,
            };
            let src_pos = match self.neuron_grid_pos(nid) {
                Some(p) => p,
                None => continue,
            };

            // 같은 구역 + 연결 가능한 구역 모두 탐색
            let mut target_regions: Vec<RegionType> = vec![src_region];
            target_regions.extend_from_slice(src_region.targets());

            for &tgt_region in &target_regions {
                if sprout_count >= MAX_SPROUT_PER_TICK {
                    break;
                }

                let tgt_neurons = match self.region_neurons.get(&tgt_region) {
                    Some(ns) => ns,
                    None => continue,
                };
                let (tgt_cols, tgt_rows) = tgt_region.grid_dims();
                let r = SPROUT_SEARCH_RADIUS;

                // 같은 구역: 그리드 반경 내 뉴런만 순회
                if src_region == tgt_region {
                    let x_min = (src_pos.0 as isize - r as isize).max(0) as usize;
                    let x_max = (src_pos.0 + r + 1).min(tgt_cols);
                    let y_min = (src_pos.1 as isize - r as isize).max(0) as usize;
                    let y_max = (src_pos.1 + r + 1).min(tgt_rows);

                    for y in y_min..y_max {
                        for x in x_min..x_max {
                            let tgt_idx = y * tgt_cols + x;
                            if tgt_idx >= tgt_neurons.len() { continue; }
                            let tgt_nid = &tgt_neurons[tgt_idx];
                            if tgt_nid == nid { continue; }

                            let dx = src_pos.0 as f64 - x as f64;
                            let dy = src_pos.1 as f64 - y as f64;
                            let dist_sq = dx * dx + dy * dy;

                            let prob = SPROUT_RATE * (-dist_sq / (2.0 * SPROUT_SIGMA * SPROUT_SIGMA)).exp();
                            if rng.random::<f64>() >= prob { continue; }

                            let already = self.neurons.get(nid)
                                .map(|n| n.outgoing_cache.iter().any(|os| os.post_neuron == *tgt_nid))
                                .unwrap_or(false);
                            if already { continue; }

                            if let Some(neuron) = self.neurons.get_mut(nid) {
                                neuron.create_synapse(&self.synapse_store, tgt_nid.clone(), SPROUT_WEIGHT, None, None);
                                sprout_count += 1;
                            }
                            if sprout_count >= MAX_SPROUT_PER_TICK { break; }
                        }
                        if sprout_count >= MAX_SPROUT_PER_TICK { break; }
                    }
                } else {
                    // 다른 구역: 랜덤 3개만 시도 (전체 순회 대신)
                    let sample_count = 3.min(tgt_neurons.len());
                    for _ in 0..sample_count {
                        let tgt_idx = rng.random_range(0..tgt_neurons.len());
                        let tgt_nid = &tgt_neurons[tgt_idx];

                        let (src_cols, _) = src_region.grid_dims();
                        let sx = src_pos.0 as f64 / src_cols as f64;
                        let sy = src_pos.1 as f64 / tgt_rows as f64;
                        let tx = (tgt_idx % tgt_cols) as f64 / tgt_cols as f64;
                        let ty = (tgt_idx / tgt_cols) as f64 / tgt_rows as f64;
                        let dx = (sx - tx) * 8.0;
                        let dy = (sy - ty) * 8.0;
                        let dist_sq = dx * dx + dy * dy;

                        let prob = SPROUT_RATE * (-dist_sq / (2.0 * SPROUT_SIGMA * SPROUT_SIGMA)).exp();
                        if rng.random::<f64>() >= prob { continue; }

                        let already = self.neurons.get(nid)
                            .map(|n| n.outgoing_cache.iter().any(|os| os.post_neuron == *tgt_nid))
                            .unwrap_or(false);
                        if already { continue; }

                        if let Some(neuron) = self.neurons.get_mut(nid) {
                            neuron.create_synapse(&self.synapse_store, tgt_nid.clone(), SPROUT_WEIGHT, None, None);
                            sprout_count += 1;
                        }
                        if sprout_count >= MAX_SPROUT_PER_TICK { break; }
                    }
                }
            }
        }

    }

    // ═══════════════════════════════════════════════
    //  뉴런 사멸 + 신생 (Apoptosis + Neurogenesis)
    // ═══════════════════════════════════════════════

    /// prune 후 호출: 제거된 시냅스의 뉴런들을 점검하고,
    /// outgoing 시냅스가 0인 고아 뉴런은 사멸 → 랜덤 구역에 새 뉴런 신생
    fn cleanup_and_neurogenesis(&mut self, removed_pairs: &[(NeuronId, NeuronId)], removed_sids: &[SynapseId]) {
        // 영향받은 뉴런 ID 수집
        let mut affected: HashSet<NeuronId> = HashSet::new();
        for (pre, post) in removed_pairs {
            affected.insert(pre.clone());
            affected.insert(post.clone());
        }

        let removed_sid_set: HashSet<&SynapseId> = removed_sids.iter().collect();

        // 각 뉴런의 outgoing 정리 + 고아 판정
        let mut dead_neurons: Vec<NeuronId> = Vec::new();
        for nid in &affected {
            if let Some(neuron) = self.neurons.get_mut(nid) {
                neuron.outgoing_cache.retain(|os| !removed_sid_set.contains(&os.id));
                neuron.outgoing.retain(|sid| !removed_sid_set.contains(sid));
                if neuron.outgoing_cache.is_empty() {
                    // incoming 체크: outgoing_cache.post_neuron 직접 비교 (synapse_store 접근 없음)
                    let has_incoming = self.neurons.values().any(|n| {
                        n.outgoing_cache.iter().any(|os| os.post_neuron == *nid)
                    });
                    if !has_incoming {
                        dead_neurons.push(nid.clone());
                    }
                }
            }
        }

        if dead_neurons.is_empty() {
            return;
        }

        let mut rng = rand::rng();
        let all_regions: Vec<RegionType> = vec![
            RegionType::Input, RegionType::Emotion, RegionType::Reason,
            RegionType::Storage, RegionType::Output,
        ];

        for dead_nid in &dead_neurons {
            // 사멸: 뉴런 제거
            let dead_region = self.neuron_region.remove(dead_nid);
            self.neurons.remove(dead_nid);
            if let Some(region) = dead_region {
                if let Some(nids) = self.region_neurons.get_mut(&region) {
                    nids.retain(|n| n != dead_nid);
                }
            }

            // 신생: 랜덤 구역에 새 뉴런 생성
            let new_region = all_regions[rng.random_range(0..all_regions.len())];
            let threshold = match new_region {
                RegionType::Emotion => EMOTION_THRESHOLD,
                RegionType::Reason => REASON_THRESHOLD,
                _ => crate::neuron::DEFAULT_THRESHOLD,
            };

            let new_id = Uuid::new_v4().to_string();
            let mut new_neuron = Neuron::new(new_id.clone());
            new_neuron.threshold = threshold;

            // 근처 뉴런과 시냅스 연결 (sprouting과 동일한 가우시안 거리 기반)
            let region_nids = self.region_neurons.get(&new_region).cloned().unwrap_or_default();
            let (cols, _rows) = new_region.grid_dims();
            let new_idx = region_nids.len(); // 맨 끝에 추가
            let new_pos = (new_idx % cols, new_idx / cols);

            // 근처 뉴런과 양방향 약한 시냅스 생성
            let mut connections = 0;
            for (idx, other_nid) in region_nids.iter().enumerate() {
                if connections >= 3 { break; } // 초기 연결 최대 3개
                let other_pos = (idx % cols, idx / cols);
                let dx = new_pos.0 as f64 - other_pos.0 as f64;
                let dy = new_pos.1 as f64 - other_pos.1 as f64;
                let dist_sq = dx * dx + dy * dy;

                let prob = SPROUT_RATE * 5.0 * (-dist_sq / (2.0 * SPROUT_SIGMA * SPROUT_SIGMA)).exp();
                if rng.random::<f64>() < prob {
                    // 새 뉴런 → 기존 뉴런
                    new_neuron.create_synapse(
                        &self.synapse_store,
                        other_nid.clone(),
                        SPROUT_WEIGHT,
                        None, None,
                    );
                    // 기존 뉴런 → 새 뉴런
                    if let Some(other) = self.neurons.get_mut(other_nid) {
                        other.create_synapse(
                            &self.synapse_store,
                            new_id.clone(),
                            SPROUT_WEIGHT,
                            None, None,
                        );
                    }
                    connections += 1;
                }
            }

            // 같은 구역의 targets 구역과도 연결 시도
            for &tgt_region in new_region.targets() {
                if connections >= 3 { break; }
                let tgt_nids = self.region_neurons.get(&tgt_region).cloned().unwrap_or_default();
                if tgt_nids.is_empty() { continue; }
                let tgt_idx = rng.random_range(0..tgt_nids.len());
                let tgt_nid = &tgt_nids[tgt_idx];
                new_neuron.create_synapse(
                    &self.synapse_store,
                    tgt_nid.clone(),
                    SPROUT_WEIGHT,
                    None, None,
                );
                connections += 1;
            }

            self.neurons.insert(new_id.clone(), new_neuron);
            self.neuron_region.insert(new_id.clone(), new_region);
            self.region_neurons.entry(new_region).or_default().push(new_id.clone());

            let dead_region_name = dead_region.map(|r| format!("{:?}", r)).unwrap_or("?".into());
            eprintln!(
                "  [neurogenesis] {:?} 뉴런 사멸 → {:?} 구역에 새 뉴런 탄생 (연결 {connections}개)",
                dead_region_name, new_region
            );
        }
    }

    // ═══════════════════════════════════════════════
    //  출력 토큰 간 시냅스 연결
    // ═══════════════════════════════════════════════

    /// 같은 fire에서 출력된 토큰들의 뉴런끼리 시냅스 생성
    /// → "이 토큰들은 함께 나온다" 맥락 학습
    fn link_output_tokens(&mut self, output_tokens: &[(String, SynapseId, u64)]) {
        if output_tokens.len() < 2 {
            return;
        }

        // 출력 시냅스 → post_neuron 수집 (sid→post 역인덱스로 빠르게 조회)
        let mut sid_to_post: HashMap<SynapseId, NeuronId> = HashMap::new();
        for neuron in self.neurons.values() {
            for os in &neuron.outgoing_cache {
                sid_to_post.insert(os.id.clone(), os.post_neuron.clone());
            }
        }
        let mut output_neurons: Vec<(NeuronId, String)> = Vec::new();
        let mut seen: HashSet<NeuronId> = HashSet::new();
        for (tok, sid, _) in output_tokens {
            if let Some(post_nid) = sid_to_post.get(sid) {
                if seen.insert(post_nid.clone()) {
                    output_neurons.push((post_nid.clone(), tok.clone()));
                }
            }
        }

        // 출력 뉴런 쌍마다 시냅스 연결
        for i in 0..output_neurons.len() {
            for j in (i + 1)..output_neurons.len() {
                let (nid_a, _) = &output_neurons[i];
                let (nid_b, _) = &output_neurons[j];

                // A→B 연결 (outgoing_cache의 post_neuron 직접 비교, DB 접근 없음)
                let already_ab = self.neurons.get(nid_a)
                    .map(|n| n.outgoing_cache.iter().any(|os| os.post_neuron == *nid_b))
                    .unwrap_or(true);

                if !already_ab {
                    if let Some(neuron) = self.neurons.get_mut(nid_a) {
                        neuron.create_synapse(
                            &self.synapse_store,
                            nid_b.clone(),
                            SPROUT_WEIGHT, // 약한 초기 연결
                            None,
                            None,
                        );
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════
    //  경로 역추적 (해마 통합 시 실행)
    // ═══════════════════════════════════════════════

    /// 최근 fire 기록의 피드백 정보를 기반으로 중간 경로 시냅스 modifier 조절
    /// 해마 통합 시점에 호출 — 오프라인 경로 최적화
    fn consolidate_path_modifiers(&mut self) {
        // 최근 rewarded 레코드의 인덱스만 수집
        let rewarded_indices: Vec<usize> = self.fire_records.iter()
            .enumerate()
            .rev()
            .filter(|(_, r)| r.rewarded)
            .take(5)
            .map(|(i, _)| i)
            .collect();
        if rewarded_indices.is_empty() {
            return;
        }

        // 필요한 sid만 수집 (전체 200k 대신 관련된 수천 개만)
        let mut needed_sids: HashSet<SynapseId> = HashSet::new();
        for &idx in &rewarded_indices {
            let record = &self.fire_records[idx];
            for sid in &record.fired_synapses {
                needed_sids.insert(sid.clone());
            }
            for (_, sid, _) in &record.output_tokens {
                needed_sids.insert(sid.clone());
            }
        }

        // 필요한 sid만 해시맵으로 구축 — O(neurons × avg_outgoing) 대신 O(needed)
        let mut sid_info: HashMap<SynapseId, (NeuronId, NeuronId, f64)> =
            HashMap::with_capacity(needed_sids.len());
        for (nid, neuron) in &self.neurons {
            for os in &neuron.outgoing_cache {
                if needed_sids.contains(&os.id) {
                    sid_info.insert(os.id.clone(), (nid.clone(), os.post_neuron.clone(), os.modifier));
                }
            }
        }

        // pre_neuron별 modifier 갱신을 그룹핑 (outgoing_cache 탐색 최소화)
        let mut updates_by_neuron: HashMap<NeuronId, Vec<(SynapseId, f64)>> = HashMap::new();
        let mut store_updates: Vec<(String, f64)> = Vec::new();

        for &idx in &rewarded_indices {
            let record = &self.fire_records[idx];
            let output_nids: HashSet<NeuronId> = record.output_tokens.iter()
                .filter_map(|(_, sid, _)| sid_info.get(sid).map(|(_, post, _)| post.clone()))
                .collect();

            let output_sids: HashSet<&SynapseId> = record.output_tokens.iter()
                .map(|(_, sid, _)| sid)
                .collect();

            let avg_output_mod: f64 = if record.output_tokens.is_empty() {
                0.0
            } else {
                let sum: f64 = record.output_tokens.iter()
                    .filter_map(|(_, osid, _)| sid_info.get(osid).map(|(_, _, m)| *m))
                    .sum();
                sum / record.output_tokens.len() as f64
            };

            for sid in &record.fired_synapses {
                if output_sids.contains(sid) {
                    continue;
                }
                if let Some((pre, post, modifier)) = sid_info.get(sid) {
                    let is_output_target = self.neuron_region.get(post)
                        .map(|r| *r == RegionType::Output)
                        .unwrap_or(false);
                    if is_output_target {
                        continue;
                    }
                    let leads_to_output = output_nids.contains(post);
                    let factor = if leads_to_output { 0.3 } else { 0.1 };
                    let new_mod = (modifier + avg_output_mod * factor).clamp(-1.0, 1.0);

                    updates_by_neuron.entry(pre.clone())
                        .or_default()
                        .push((sid.clone(), new_mod));
                    store_updates.push((sid.clone(), new_mod));
                }
            }
        }

        // outgoing_cache 갱신: 뉴런별로 한번에 처리
        for (nid, updates) in &updates_by_neuron {
            if let Some(neuron) = self.neurons.get_mut(nid) {
                let sid_mod: HashMap<&str, f64> = updates.iter()
                    .map(|(sid, m)| (sid.as_str(), *m))
                    .collect();
                for os in &mut neuron.outgoing_cache {
                    if let Some(&new_mod) = sid_mod.get(os.id.as_str()) {
                        os.modifier = new_mod;
                    }
                }
            }
        }

        // synapse_store batch 업데이트 (단일 lock)
        self.synapse_store.update_modifiers_batch(&store_updates);
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

        let output_sids = record.output_tokens.clone();
        let path_sids = record.fired_synapses.clone();
        record.rewarded = true;

        let mut updated = 0;
        // 출력 시냅스 modifier 조정 (weight는 유지) — 수집 후 일괄 적용
        let mut mod_updates: Vec<(NeuronId, SynapseId, f64)> = Vec::new();
        for (_, sid, _) in &output_sids {
            if let Some(syn) = self.synapse_store.get(sid) {
                let delta = strength * FEEDBACK_LR;
                let new_mod = if positive {
                    syn.modifier + delta
                } else {
                    syn.modifier - delta * 1.5
                };
                mod_updates.push((syn.pre_neuron.clone(), sid.clone(), new_mod));
                updated += 1;
            }
        }

        // 경로 시냅스도 modifier 조정 (출력보다 약하게)
        if !positive {
            for sid in &path_sids {
                if let Some(syn) = self.synapse_store.get(sid) {
                    let delta = strength * FEEDBACK_LR * 0.5;
                    let new_mod = syn.modifier - delta;
                    mod_updates.push((syn.pre_neuron.clone(), sid.clone(), new_mod));
                    updated += 1;
                }
            }
        }

        for (pre, sid, new_mod) in mod_updates {
            self.sync_modifier(&pre, &sid, new_mod);
        }

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
        record.rewarded = true;

        let mut reinforced = 0;
        let mut weakened = 0;
        let mut mod_updates: Vec<(NeuronId, SynapseId, f64)> = Vec::new();

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
                let new_mod = if score > 0.0 {
                    reinforced += 1;
                    syn.modifier + delta
                } else {
                    weakened += 1;
                    syn.modifier - delta
                };
                mod_updates.push((syn.pre_neuron.clone(), sid.clone(), new_mod));
            }
        }

        for (pre, sid, new_mod) in mod_updates {
            self.sync_modifier(&pre, &sid, new_mod);
        }
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

        let weight_updates: Vec<(NeuronId, SynapseId, f64)> = output_sids.iter()
            .filter_map(|(_, sid, _)| {
                self.synapse_store.get(sid).map(|syn| {
                    let delta = if meaning >= 0.1 {
                        score * FEEDBACK_LR
                    } else {
                        -(1.0 - score) * FEEDBACK_LR * 0.5
                    };
                    let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                    (syn.pre_neuron.clone(), sid.clone(), new_w)
                })
            })
            .collect();
        for (pre, sid, new_w) in weight_updates {
            self.sync_weight(&pre, &sid, new_w);
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

            let weight_updates: Vec<(NeuronId, SynapseId, f64)> = output_sids.iter()
                .filter_map(|(_, sid, _)| {
                    self.synapse_store.get(sid).map(|syn| {
                        let delta = if meaning >= 0.1 {
                            score * FEEDBACK_LR
                        } else {
                            -(1.0 - score) * FEEDBACK_LR * 0.5
                        };
                        let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                        (syn.pre_neuron.clone(), sid.clone(), new_w)
                    })
                })
                .collect();
            for (pre, sid, new_w) in weight_updates {
                self.sync_weight(&pre, &sid, new_w);
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

        // 해마 스레드에 활성화 데이터 전송 (비동기)
        let _ = self.hippo_tx.try_send(HippoInput::ActivationData {
            neurons_activated: neurons_visited.clone(),
            neuron_fire_ticks: neuron_fire_ticks.clone(),
        });
        // 통합 결과 적용 (있으면)
        self.apply_consolidation_results();

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

        let weight_updates: Vec<(NeuronId, SynapseId, f64)> = output_sids.iter()
            .filter_map(|(_, sid, _)| {
                self.synapse_store.get(sid).map(|syn| {
                    let delta = if meaning >= 0.1 {
                        score * FEEDBACK_LR
                    } else {
                        -(1.0 - score) * FEEDBACK_LR * 0.5
                    };
                    let new_w = (syn.weight + delta).clamp(MIN_WEIGHT, MAX_WEIGHT);
                    (syn.pre_neuron.clone(), sid.clone(), new_w)
                })
            })
            .collect();
        for (pre, sid, new_w) in weight_updates {
            self.sync_weight(&pre, &sid, new_w);
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
        use RegionType::*;
        let mut regions = serde_json::Map::new();
        for region in &[Input, Emotion, Reason, Storage, Output] {
            let neuron_count = self.region_neurons.get(region).map(|n| n.len()).unwrap_or(0);
            let synapse_count: usize = self.region_neurons.get(region)
                .map(|neurons| neurons.iter()
                    .filter_map(|n| self.neurons.get(n))
                    .map(|n| n.outgoing_cache.len())
                    .sum())
                .unwrap_or(0);

            let mut info = serde_json::json!({
                "neurons": neuron_count,
                "synapses": synapse_count,
            });

            if *region == Storage {
                let all_sids: Vec<SynapseId> = self.region_neurons.get(region)
                    .map(|neurons| neurons.iter()
                        .filter_map(|n| self.neurons.get(n))
                        .flat_map(|n| n.outgoing_cache.iter().map(|os| os.id.clone()))
                        .collect())
                    .unwrap_or_default();
                let memory_count = self.synapse_store.cached_memory_count(&all_sids);
                info["memories"] = serde_json::json!(memory_count);
            }

            regions.insert(format!("{region}"), info);
        }

        serde_json::json!({
            "neurons": self.neurons.len(),
            "synapses": self.synapse_store.count(),
            "cached": self.synapse_store.cached_count(),
            "token_vocab": self.synapse_store.token_index_count(),
            "fire_count": self.hippo_stats.fire_count.load(Ordering::Relaxed),
            "patterns": self.hippo_stats.pattern_count.load(Ordering::Relaxed),
            "regions": regions,
        })
    }

    pub fn neuron_count(&self) -> usize { self.neurons.len() }
    pub fn synapse_count(&self) -> usize { self.synapse_store.count() }
    pub fn cached_count(&self) -> usize { self.synapse_store.cached_count() }
    pub fn fire_count(&self) -> u64 { self.hippo_stats.fire_count.load(Ordering::Relaxed) as u64 }
    pub fn pattern_count(&self) -> usize { self.hippo_stats.pattern_count.load(Ordering::Relaxed) as usize }
    pub fn token_vocab_count(&self) -> usize { self.synapse_store.token_index_count() }

    /// 해마 스레드에서 보내온 통합 결과 적용
    fn apply_consolidation_results(&mut self) {
        while let Ok(result) = self.consolidation_rx.try_recv() {
            if !result.patterns.is_empty() {
                let total = result.patterns.len();
                let cap = total.min(20);
                eprintln!(
                    "  [해마 결과 적용] 패턴 {}개 (저장 {}개), co-fire 쌍 {}개",
                    total, cap, result.cofire_pairs.len()
                );
                let show = cap.min(3);
                for (pattern, freq) in &result.patterns[..show] {
                    self.store_memory(pattern.clone(), *freq);
                }
                for (pattern, freq) in &result.patterns[show..cap] {
                    self.store_memory_silent(pattern.clone(), *freq);
                }
                self.consolidate_path_modifiers();
            }
            if !result.cofire_pairs.is_empty() {
                self.connect_cofiring_pairs(&result.cofire_pairs);
            }
        }
    }

    /// fire 후처리: 응답 전송 후 별도로 호출 (cooldown, link, prune + 해마 데이터 전송)
    pub fn run_pending_post_process(&mut self) {
        // 해마 스레드에서 온 통합 결과 적용 (non-blocking)
        self.apply_consolidation_results();

        let data = match self.pending_post_process.take() {
            Some(d) => d,
            None => return,
        };
        let pp_start = Instant::now();

        // 해마 스레드로 활성화 데이터 전송 (non-blocking, 꽉 차면 drop)
        let _ = self.hippo_tx.try_send(HippoInput::ActivationData {
            neurons_activated: data.neurons_activated,
            neuron_fire_ticks: data.neuron_fire_ticks,
        });
        let t1 = Instant::now();

        // cooldown 이력
        let mut used_sids: HashSet<SynapseId> = HashSet::new();
        for (_, sid, _) in &data.all_output_tokens {
            used_sids.insert(sid.clone());
        }
        for sid in &data.all_fired {
            used_sids.insert(sid.clone());
        }
        for (_, history) in self.cooldown_history.iter_mut() {
            history.push_front(false);
            if history.len() > COOLDOWN_HISTORY {
                history.pop_back();
            }
        }
        for sid in &used_sids {
            let history = self.cooldown_history.entry(sid.clone()).or_insert_with(VecDeque::new);
            if let Some(front) = history.front_mut() {
                *front = true;
            } else {
                history.push_front(true);
            }
        }
        self.cooldown_history.retain(|_, h| h.iter().any(|&used| used));
        let t2 = Instant::now();

        // 패턴 병합 + 출력 토큰 연결
        self.consolidate_patterns(&data.all_output_tokens);
        let t3 = Instant::now();
        self.link_output_tokens(&data.used_output_tokens);
        let t4 = Instant::now();

        // 주기적 pruning
        if data.fire_id > 0 && data.fire_id % 10 == 0 {
            let (removed, remaining, removed_pairs, removed_sids) = self.synapse_store.prune(MIN_WEIGHT);
            if removed > 0 {
                eprintln!("  [prune] {removed}개 제거 → {remaining}개 남음");
                self.cleanup_and_neurogenesis(&removed_pairs, &removed_sids);
            }
        }
        let t5 = Instant::now();

        let total = pp_start.elapsed();
        if total.as_millis() > 200 {
            eprintln!(
                "  [post #{fire_id}] post_process={total:?} | hippo_send={:?} cooldown={:?} patterns={:?} link={:?} prune={:?}",
                t1 - pp_start, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                fire_id = data.fire_id,
            );
        } else {
            eprintln!("  [post #{fire_id}] post_process={total:?}", fire_id = data.fire_id);
        }
    }

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
                        .map(|n| n.outgoing_cache.len())
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
                            .flat_map(|n| n.outgoing_cache.iter().map(|os| os.id.clone()))
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
            self.hippo_stats.fire_count.load(Ordering::Relaxed),
            self.hippo_stats.pattern_count.load(Ordering::Relaxed)
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
