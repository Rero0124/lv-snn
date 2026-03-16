use crate::neuron::NeuronId;
use crate::tokenizer::TokenType;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

pub type SynapseId = String;

const SYNAPSE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("synapses_v2");
const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("state");

/// 경로 기억: 뉴런 단위 경로 패턴
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathMemory {
    pub pattern: Vec<NeuronId>,
    pub frequency: u64,
}

/// 시냅스: 뉴런 간 연결 (토큰 단위)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub id: SynapseId,
    pub pre_neuron: NeuronId,
    pub post_neuron: NeuronId,
    pub weight: f64,
    pub token: Option<String>,
    pub token_type: Option<TokenType>,
    pub memory: Option<PathMemory>,
    pub active: bool,
}

impl Synapse {
    pub fn new(
        id: SynapseId,
        pre: NeuronId,
        post: NeuronId,
        weight: f64,
        token: Option<String>,
        token_type: Option<TokenType>,
        memory: Option<PathMemory>,
    ) -> Self {
        Self {
            id,
            pre_neuron: pre,
            post_neuron: post,
            weight,
            token,
            token_type,
            memory,
            active: true,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("synapse serialization failed")
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        serde_json::from_slice(bytes).ok()
    }
}

struct CachedEntry {
    synapse: Synapse,
    last_access: Instant,
    access_count: u64,
    dirty: bool,
}

struct CacheState {
    cache: HashMap<SynapseId, CachedEntry>,
    token_index: HashMap<String, Vec<SynapseId>>,
    db_count: usize,
}

pub struct SynapseStore {
    state: Mutex<CacheState>,
    db: Arc<Database>,
    max_cached: usize,
}

// SynapseStore는 Mutex로 보호되므로 Send + Sync
unsafe impl Sync for SynapseStore {}

impl SynapseStore {
    pub fn new(db_path: PathBuf, max_cached: usize) -> Self {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let db = Database::create(&db_path).expect("failed to open redb");
        {
            let write_txn = db.begin_write().unwrap();
            let _ = write_txn.open_table(SYNAPSE_TABLE).unwrap();
            let _ = write_txn.open_table(STATE_TABLE).unwrap();
            write_txn.commit().unwrap();
        }
        let db_count = {
            let read_txn = db.begin_read().unwrap();
            let table = read_txn.open_table(SYNAPSE_TABLE).unwrap();
            table.len().unwrap_or(0) as usize
        };
        Self {
            state: Mutex::new(CacheState {
                cache: HashMap::new(),
                token_index: HashMap::new(),
                db_count,
            }),
            db: Arc::new(db),
            max_cached,
        }
    }

    pub fn create(
        &self,
        pre: NeuronId,
        post: NeuronId,
        weight: f64,
        token: Option<String>,
        token_type: Option<TokenType>,
        memory: Option<PathMemory>,
    ) -> SynapseId {
        let id = Uuid::new_v4().to_string();
        let synapse = Synapse::new(id.clone(), pre, post, weight, token, token_type, memory);
        let mut st = self.state.lock().unwrap();
        if let Some(ref tok) = synapse.token {
            st.token_index.entry(tok.clone()).or_default().push(id.clone());
        }
        Self::put_cache_inner(&mut st, id.clone(), synapse, true);
        Self::evict_if_needed_inner(&mut st, &self.db, self.max_cached);
        id
    }

    pub fn get(&self, id: &str) -> Option<Synapse> {
        let mut st = self.state.lock().unwrap();
        if let Some(entry) = st.cache.get_mut(id) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            return Some(entry.synapse.clone());
        }
        // 캐시 미스 → DB에서 로드 (lock 보유 상태)
        if let Some(synapse) = Self::load_from_db(&self.db, id) {
            if let Some(ref tok) = synapse.token {
                st.token_index.entry(tok.clone()).or_default().push(id.to_string());
            }
            Self::put_cache_inner(&mut st, id.to_string(), synapse.clone(), false);
            Self::evict_if_needed_inner(&mut st, &self.db, self.max_cached);
            return Some(synapse);
        }
        None
    }

    pub fn update_weight(&self, id: &str, new_weight: f64) -> bool {
        let mut st = self.state.lock().unwrap();
        if let Some(entry) = st.cache.get_mut(id) {
            entry.synapse.weight = new_weight;
            entry.last_access = Instant::now();
            entry.access_count += 1;
            entry.dirty = true;
            return true;
        }
        if let Some(mut synapse) = Self::load_from_db(&self.db, id) {
            synapse.weight = new_weight;
            Self::remove_from_db_inner(&mut st, &self.db, id);
            Self::put_cache_inner(&mut st, id.to_string(), synapse, true);
            Self::evict_if_needed_inner(&mut st, &self.db, self.max_cached);
            return true;
        }
        false
    }

    /// 캐시에서 특정 토큰을 가진 시냅스 ID 목록
    pub fn find_by_token(&self, token: &str) -> Vec<SynapseId> {
        let st = self.state.lock().unwrap();
        st.token_index.get(token).cloned().unwrap_or_default()
    }

    pub fn is_cached(&self, id: &str) -> bool {
        let st = self.state.lock().unwrap();
        st.cache.contains_key(id)
    }

    pub fn count(&self) -> usize {
        let st = self.state.lock().unwrap();
        st.cache.len() + st.db_count
    }

    pub fn cached_count(&self) -> usize {
        let st = self.state.lock().unwrap();
        st.cache.len()
    }

    pub fn token_index_count(&self) -> usize {
        let st = self.state.lock().unwrap();
        st.token_index.len()
    }

    /// 캐시에서 기억 데이터가 있는 시냅스 수
    pub fn cached_memory_count(&self, synapse_ids: &[SynapseId]) -> usize {
        let st = self.state.lock().unwrap();
        synapse_ids
            .iter()
            .filter(|sid| {
                st.cache
                    .get(sid.as_str())
                    .is_some_and(|e| e.synapse.memory.is_some())
            })
            .count()
    }

    fn put_cache_inner(st: &mut CacheState, id: SynapseId, synapse: Synapse, dirty: bool) {
        st.cache.insert(
            id,
            CachedEntry {
                synapse,
                last_access: Instant::now(),
                access_count: 1,
                dirty,
            },
        );
    }

    fn evict_if_needed_inner(st: &mut CacheState, db: &Arc<Database>, max_cached: usize) {
        while st.cache.len() > max_cached {
            let victim = st
                .cache
                .iter()
                .min_by(|a, b| {
                    a.1.access_count
                        .cmp(&b.1.access_count)
                        .then(a.1.last_access.cmp(&b.1.last_access))
                })
                .map(|(id, _)| id.clone());
            if let Some(id) = victim {
                if let Some(entry) = st.cache.remove(&id) {
                    // 토큰 인덱스에서도 제거
                    if let Some(ref tok) = entry.synapse.token {
                        if let Some(ids) = st.token_index.get_mut(tok) {
                            ids.retain(|s| s != &id);
                            if ids.is_empty() {
                                st.token_index.remove(tok);
                            }
                        }
                    }
                    if entry.dirty {
                        Self::save_to_db_inner(st, db, &entry.synapse);
                    }
                }
            }
        }
    }

    fn save_to_db_inner(st: &mut CacheState, db: &Arc<Database>, synapse: &Synapse) {
        let bytes = synapse.to_bytes();
        let write_txn = db.begin_write().unwrap();
        {
            let mut table = write_txn.open_table(SYNAPSE_TABLE).unwrap();
            table.insert(synapse.id.as_str(), bytes.as_slice()).unwrap();
        }
        write_txn.commit().unwrap();
        st.db_count += 1;
    }

    fn load_from_db(db: &Arc<Database>, id: &str) -> Option<Synapse> {
        let read_txn = db.begin_read().ok()?;
        let table = read_txn.open_table(SYNAPSE_TABLE).ok()?;
        let guard = table.get(id).ok()??;
        Synapse::from_bytes(guard.value())
    }

    fn remove_from_db_inner(st: &mut CacheState, db: &Arc<Database>, id: &str) {
        if let Ok(write_txn) = db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(SYNAPSE_TABLE) {
                if table.remove(id).ok().flatten().is_some() {
                    st.db_count = st.db_count.saturating_sub(1);
                }
            }
            write_txn.commit().ok();
        }
    }

    /// 약한 시냅스 + 중복 시냅스 제거 (pruning)
    /// 반환: (제거된 수, 남은 수)
    pub fn prune(&self, min_weight: f64) -> (usize, usize) {
        let mut st = self.state.lock().unwrap();

        // 1) 캐시에서 약한 시냅스 수집
        let weak_ids: Vec<SynapseId> = st.cache.iter()
            .filter(|(_, e)| e.synapse.weight <= min_weight)
            .map(|(id, _)| id.clone())
            .collect();

        // 2) 중복 시냅스 수집 (같은 pre→post+token 중 가장 강한 것만 유지)
        let mut best: HashMap<(NeuronId, NeuronId, Option<String>), (SynapseId, f64)> = HashMap::new();
        let mut dup_ids: Vec<SynapseId> = Vec::new();

        for (id, entry) in st.cache.iter() {
            let key = (
                entry.synapse.pre_neuron.clone(),
                entry.synapse.post_neuron.clone(),
                entry.synapse.token.clone(),
            );
            if let Some((best_id, best_w)) = best.get_mut(&key) {
                if entry.synapse.weight > *best_w {
                    dup_ids.push(best_id.clone());
                    *best_id = id.clone();
                    *best_w = entry.synapse.weight;
                } else {
                    dup_ids.push(id.clone());
                }
            } else {
                best.insert(key, (id.clone(), entry.synapse.weight));
            }
        }

        // 3) 합치기
        let mut to_remove: Vec<SynapseId> = weak_ids;
        for id in dup_ids {
            if !to_remove.contains(&id) {
                to_remove.push(id);
            }
        }

        if to_remove.is_empty() {
            let remaining = st.cache.len() + st.db_count;
            return (0, remaining);
        }

        // 4) 캐시에서 제거
        for id in &to_remove {
            if let Some(entry) = st.cache.remove(id) {
                if let Some(ref tok) = entry.synapse.token {
                    if let Some(ids) = st.token_index.get_mut(tok) {
                        ids.retain(|s| s != id);
                        if ids.is_empty() {
                            st.token_index.remove(tok);
                        }
                    }
                }
            }
        }

        // 5) DB에서도 제거 (배치)
        if let Ok(write_txn) = self.db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(SYNAPSE_TABLE) {
                for id in &to_remove {
                    if table.remove(id.as_str()).ok().flatten().is_some() {
                        st.db_count = st.db_count.saturating_sub(1);
                    }
                }
            }
            write_txn.commit().ok();
        }

        let removed = to_remove.len();
        let remaining = st.cache.len() + st.db_count;
        (removed, remaining)
    }

    /// DB 전체를 스캔해서 약한 시냅스 제거
    /// 캐시에 없는 DB 시냅스도 정리
    pub fn prune_db(&self, min_weight: f64) -> (usize, usize) {
        // 먼저 캐시 pruning
        let (cache_removed, _) = self.prune(min_weight);

        // DB 스캔
        let mut db_remove_ids: Vec<String> = Vec::new();
        // 중복 감지용
        let mut db_best: HashMap<(NeuronId, NeuronId, Option<String>), (String, f64)> = HashMap::new();

        if let Ok(read_txn) = self.db.begin_read() {
            if let Ok(table) = read_txn.open_table(SYNAPSE_TABLE) {
                if let Ok(iter) = table.iter() {
                    for entry in iter {
                        if let Ok((key, val)) = entry {
                            let id = key.value().to_string();
                            if let Some(syn) = Synapse::from_bytes(val.value()) {
                                // 약한 시냅스
                                if syn.weight <= min_weight {
                                    db_remove_ids.push(id);
                                    continue;
                                }
                                // 중복 체크
                                let dup_key = (syn.pre_neuron.clone(), syn.post_neuron.clone(), syn.token.clone());
                                if let Some((best_id, best_w)) = db_best.get_mut(&dup_key) {
                                    if syn.weight > *best_w {
                                        db_remove_ids.push(best_id.clone());
                                        *best_id = id;
                                        *best_w = syn.weight;
                                    } else {
                                        db_remove_ids.push(id);
                                    }
                                } else {
                                    db_best.insert(dup_key, (id, syn.weight));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut st = self.state.lock().unwrap();
        if !db_remove_ids.is_empty() {
            if let Ok(write_txn) = self.db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(SYNAPSE_TABLE) {
                    for id in &db_remove_ids {
                        if table.remove(id.as_str()).ok().flatten().is_some() {
                            st.db_count = st.db_count.saturating_sub(1);
                        }
                    }
                }
                write_txn.commit().ok();
            }
        }

        let total_removed = cache_removed + db_remove_ids.len();
        let remaining = st.cache.len() + st.db_count;
        (total_removed, remaining)
    }

    /// 모든 dirty 시냅스를 DB에 배치 저장하고 clean으로 표시
    pub fn flush_dirty(&self) {
        let mut st = self.state.lock().unwrap();
        let dirty_synapses: Vec<Synapse> = st.cache.values()
            .filter(|e| e.dirty)
            .map(|e| e.synapse.clone())
            .collect();
        if dirty_synapses.is_empty() {
            return;
        }
        let _ = writeln!(std::io::stderr(), "  [DB] dirty 시냅스 {}개 저장 중...", dirty_synapses.len());
        if let Ok(write_txn) = self.db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(SYNAPSE_TABLE) {
                for syn in &dirty_synapses {
                    let bytes = syn.to_bytes();
                    table.insert(syn.id.as_str(), bytes.as_slice()).ok();
                }
            }
            if write_txn.commit().is_ok() {
                for entry in st.cache.values_mut() {
                    entry.dirty = false;
                }
                let _ = writeln!(std::io::stderr(), "  [DB] dirty 시냅스 저장 완료");
            }
        }
    }

    /// 네트워크 상태를 DB에 저장
    pub fn save_network_state(&self, data: &[u8]) {
        match self.db.begin_write() {
            Ok(write_txn) => {
                match write_txn.open_table(STATE_TABLE) {
                    Ok(mut table) => {
                        if let Err(e) = table.insert("network", data) {
                            eprintln!("  [DB] 상태 insert 실패: {e}");
                        }
                    }
                    Err(e) => eprintln!("  [DB] STATE_TABLE 열기 실패: {e}"),
                }
                if let Err(e) = write_txn.commit() {
                    eprintln!("  [DB] 상태 커밋 실패: {e}");
                }
            }
            Err(e) => eprintln!("  [DB] write 트랜잭션 시작 실패: {e}"),
        }
    }

    /// DB에서 네트워크 상태 로드
    pub fn load_network_state(&self) -> Option<Vec<u8>> {
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(STATE_TABLE).ok()?;
        let guard = table.get("network").ok()??;
        Some(guard.value().to_vec())
    }
}

impl Drop for SynapseStore {
    fn drop(&mut self) {
        let st = self.state.get_mut().unwrap();
        let dirty: Vec<_> = st.cache.values()
            .filter(|e| e.dirty)
            .collect();
        if dirty.is_empty() {
            return;
        }
        let _ = writeln!(std::io::stderr(), "  [DB] dirty 시냅스 {}개 저장 중...", dirty.len());
        if let Ok(write_txn) = self.db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(SYNAPSE_TABLE) {
                for entry in &dirty {
                    let bytes = entry.synapse.to_bytes();
                    table.insert(entry.synapse.id.as_str(), bytes.as_slice()).ok();
                }
            }
            write_txn.commit().ok();
        }
    }
}
