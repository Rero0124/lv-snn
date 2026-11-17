//! 기준 모델: Shiu et al. 2024 의 LIF (model.py 의 Brian2 모델을 그대로 옮김)
//!
//!   dv/dt = (v0 - v + g) / t_mbr     (불응기 동안 정지)
//!   dg/dt = -g / tau                 (불응기 동안 정지)
//!   발화: v > v_th → v = v_rst, g = 0, 불응기 2.2ms
//!   시냅스: 발화 1.8ms 뒤 post 의 g += 시냅스수 × w_syn (받는 뉴런이 불응기 중이면 버려짐)
//!   자극: 150Hz 포아송, 이벤트마다 v += w_syn × 250 (불응기 없음)

use crate::data::Connectome;

const DT: f64 = 0.1; // ms
const T_MBR: f64 = 20.0;
const TAU: f64 = 5.0;
const V_TH: f64 = 7.0; // v_th - v_0 = -45 - (-52) mV
const W_SYN: f64 = 0.275;
const REFRACTORY_STEPS: i64 = 22;
const DELAY_STEPS: usize = 18;
const POISSON_HZ: f64 = 150.0;
const POISSON_KICK: f64 = W_SYN * 250.0;

struct SplitMix(u64);
impl SplitMix {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 한 번의 시행. 반환: 뉴런별 발화 횟수
pub fn run_trial(c: &Connectome, stim: &[usize], duration_ms: f64, seed: u64) -> Vec<u32> {
    run_trial_with_rest(c, stim, duration_ms, 0.0, seed).0
}

/// 자극 duration_ms 뒤에 자극 없이 rest_ms 를 더 돌린다. 반환: (자극 중 뉴런별 발화 수, 휴지 구간 총 발화 수)
pub fn run_trial_with_rest(c: &Connectome, stim: &[usize], duration_ms: f64, rest_ms: f64, seed: u64) -> (Vec<u32>, u64) {
    let (counts, rest, _) = run_full(c, stim, duration_ms, rest_ms, 0.0, seed);
    (counts, rest)
}

/// spont_per_ms: 뉴런당 1ms 마다 자발 발화할 확률 (snn 의 자발 발화와 같은 조건을 기준 모델에 주기 위함).
/// 반환: (자극 중 뉴런별 발화 수, 휴지 구간 총 발화 수, 500ms 구간별 총 발화 수)
pub fn run_full(c: &Connectome, stim: &[usize], duration_ms: f64, rest_ms: f64, spont_per_ms: f64, seed: u64) -> (Vec<u32>, u64, Vec<u64>) {
    let n = c.n();
    let stim_steps = (duration_ms / DT).round() as i64;
    let steps = stim_steps + (rest_ms / DT).round() as i64;
    let mut rest_spikes = 0u64;
    let mut windows: Vec<u64> = Vec::new();
    let p_spont_step = spont_per_ms * DT * c.n() as f64; // 한 스텝에 전체에서 자발 발화가 하나 생길 확률
    let e_m = (-DT / T_MBR).exp();
    let e_s = (-DT / TAU).exp();
    let k_g = TAU / (TAU - T_MBR) * (e_s - e_m); // g 가 v 에 기여하는 정확해 계수

    let mut v = vec![0.0f64; n]; // v - v0
    let mut g = vec![0.0f64; n];
    let mut refr_until = vec![0i64; n];
    let mut is_stim = vec![false; n];
    for &i in stim { is_stim[i] = true; }
    let mut counts = vec![0u32; n];
    let ring_len = DELAY_STEPS + 1;
    let mut ring: Vec<Vec<f64>> = (0..ring_len).map(|_| vec![0.0; n]).collect();
    let mut ring_dirty = vec![false; ring_len];
    let mut rng = SplitMix(seed);
    let p_poisson = POISSON_HZ * DT / 1000.0;
    let mut spiked: Vec<usize> = Vec::new();

    for step in 0..steps {
        // 1. 상태 갱신 + 임계 판정
        spiked.clear();
        for i in 0..n {
            if step < refr_until[i] { continue; }
            let (vi, gi) = (v[i], g[i]);
            if vi == 0.0 && gi == 0.0 { continue; }
            let nv = vi * e_m + gi * k_g;
            let ng = gi * e_s;
            v[i] = if nv.abs() < 1e-12 { 0.0 } else { nv };
            g[i] = if ng.abs() < 1e-12 { 0.0 } else { ng };
            if nv > V_TH { spiked.push(i); }
        }
        // 2. 지연이 끝난 시냅스 입력 전달
        let slot = (step as usize) % ring_len;
        if ring_dirty[slot] {
            let buf = &mut ring[slot];
            for i in 0..n {
                // 불응기 중에 도착한 입력은 버려진다. 저자 원본(Brian2)을 뉴런 하나로 돌려 확인한 동작:
                // 4ms 간격 입력에서 원본은 16ms 마다 발화하는데, 불응기 중 입력을 쌓아 두면 12ms 마다 발화한다
                if buf[i] != 0.0 { if step >= refr_until[i] { g[i] += buf[i]; } buf[i] = 0.0; }
            }
            ring_dirty[slot] = false;
        }
        // 3. 포아송 자극
        if step < stim_steps {
            for &i in stim {
                if rng.next_f64() < p_poisson { v[i] += POISSON_KICK; }
            }
        }
        // 자발 발화: 무작위 뉴런 하나를 강제로 임계 위로
        if p_spont_step > 0.0 && rng.next_f64() < p_spont_step {
            let i = (rng.next_f64() * n as f64) as usize;
            v[i.min(n - 1)] += POISSON_KICK;
        }
        let w = (step / 5000) as usize;
        if windows.len() <= w { windows.resize(w + 1, 0); }
        windows[w] += spiked.len() as u64;
        // 4. 발화 처리: 전달 예약 + 리셋
        if !spiked.is_empty() {
            let target = (step as usize + DELAY_STEPS) % ring_len;
            ring_dirty[target] = true;
            for &i in &spiked {
                if step < stim_steps { counts[i] += 1; } else { rest_spikes += 1; }
                for k in c.out_start[i]..c.out_start[i + 1] {
                    ring[target][c.out_post[k] as usize] += c.out_count[k] as f64 * W_SYN;
                }
                v[i] = 0.0;
                g[i] = 0.0;
                if !is_stim[i] { refr_until[i] = step + REFRACTORY_STEPS; }
            }
        }
    }
    (counts, rest_spikes, windows)
}
