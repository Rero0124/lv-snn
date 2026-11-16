//! 세포 종류별 특성이 문헌의 정성적 사실과 맞는지 확인하는 점검 (--typecheck).
//!   1) 같은 입력에서 Pv 가 추체세포보다 훨씬 빠르게 발화하는가
//!   2) 일정한 입력에서 추체세포는 발화율이 떨어지고(적응) Pv 는 유지되는가
//!   3) 연달아 활성화될 때 피로형 시냅스는 약해지고 촉진형은 강해지는가
//!   4) Vip 를 구동하면 Sst 가 줄고 흥분성 뉴런의 활동이 늘어나는가 (탈억제)

use crate::neuron::{CellType, Neuron};
use crate::synapse::Synapse;

fn typed(id: u32, t: CellType) -> Neuron {
    let mut n = Neuron::new(id, 0.0, 0.0);
    n.set_type(t);
    n.spont_scale = 0.0;
    n
}

/// 매 틱 drive 만큼 입력을 주며 ticks 틱 돌려 발화 틱 목록을 돌려준다
fn drive_single(t: CellType, drive: f64, ticks: u64) -> Vec<u64> {
    let mut n = typed(0, t);
    let mut spikes = Vec::new();
    for tick in 1..=ticks {
        n.receive(drive);
        n.decay();
        if n.try_fire(tick, 0.5, 0.0, true).is_some() { spikes.push(tick); }
    }
    spikes
}

pub fn run() {
    println!("== 1·2. 일정한 입력에서의 발화 (300틱) ==");
    println!("{:<10} {:>8} {:>10} {:>10} {:>12}", "종류", "입력", "처음 30틱", "마지막 30틱", "마지막/처음");
    for &(name, t) in &[("추체", CellType::Pyramidal), ("Pv", CellType::Pv), ("Sst", CellType::Sst), ("Vip", CellType::Vip)] {
        for &drive in &[0.5, 1.0] {
            let s = drive_single(t, drive, 300);
            let first = s.iter().filter(|&&x| x <= 30).count();
            let last = s.iter().filter(|&&x| x > 270).count();
            println!("{:<10} {:>8.1} {:>10} {:>10} {:>12.2}", name, drive, first, last,
                if first > 0 { last as f64 / first as f64 } else { 0.0 });
        }
    }

    println!("\n== 3. 연달아 활성화될 때의 전달량 (2틱 간격 10번, 가중치 1.0 기준) ==");
    for &(name, facilitating) in &[("피로형", false), ("촉진형", true)] {
        let mut syn = Synapse::new(0, 1.0);
        syn.facilitating = facilitating;
        let mut seq = Vec::new();
        for _ in 0..10 {
            seq.push(syn.effective_weight());
            syn.fire_fatigue();
            syn.recover();
            syn.recover();
        }
        println!("{:<8} {}", name, seq.iter().map(|w| format!("{w:.2}")).collect::<Vec<_>>().join(" "));
    }

    println!("\n== 5·6. 두 구획 추체세포: 입력 위치와 억제 위치 (300틱 발화 수, 괄호 안은 버스트) ==");
    println!("{:<34} {:>8}", "조건", "발화");
    let two = |basal: f64, apical: f64, soma_inh: f64, dend_inh: f64| -> (usize, usize) {
        let mut n = typed(0, CellType::Pyramidal);
        n.has_dendrite = true;
        let (mut spikes, mut bursts) = (0, 0);
        for tick in 1..=300u64 {
            n.receive(basal - soma_inh);
            n.receive_dend(apical - dend_inh);
            n.decay();
            if n.try_fire(tick, 0.5, 0.0, true).is_some() { spikes += 1; if n.last_burst { bursts += 1; } }
        }
        (spikes, bursts)
    };
    for &(label, b, a, si, di) in &[
        ("세포체 입력만", 0.25, 0.0, 0.0, 0.0),
        ("수상돌기 입력만", 0.0, 0.12, 0.0, 0.0),
        ("둘 다", 0.25, 0.12, 0.0, 0.0),
        ("둘 다 + 세포체 억제(Pv 식)", 0.25, 0.12, 0.12, 0.0),
        ("둘 다 + 수상돌기 억제(Sst 식)", 0.25, 0.12, 0.0, 0.12),
        ("세포체 입력만 + 세포체 억제", 0.25, 0.0, 0.12, 0.0),
        ("세포체 입력만 + 수상돌기 억제", 0.25, 0.0, 0.0, 0.12),
    ] {
        let (sp, bu) = two(b, a, si, di);
        println!("{:<34} {:>8} ({})", label, sp, bu);
    }

    println!("\n== 4. 탈억제 회로 (추체 40, Pv 8, Sst 6, Vip 4) ==");
    println!("{:<22} {:>8} {:>8} {:>8} {:>8}", "조건", "추체", "Pv", "Sst", "Vip");
    for &(label, vip_drive, sst_drive) in &[("기본", 0.0, 0.0), ("Vip 구동", 0.8, 0.0), ("Sst 구동", 0.0, 0.8)] {
        let counts = motif(vip_drive, sst_drive);
        println!("{:<22} {:>8} {:>8} {:>8} {:>8}", label, counts[0], counts[1], counts[2], counts[3]);
    }
}

/// 종류별 연결 규칙으로 만든 작은 회로. 반환: [추체, Pv, Sst, Vip] 총 발화 수 (300틱)
fn motif(vip_drive: f64, sst_drive: f64) -> [u32; 4] {
    let kinds: Vec<CellType> = std::iter::repeat(CellType::Pyramidal).take(40)
        .chain(std::iter::repeat(CellType::Pv).take(8))
        .chain(std::iter::repeat(CellType::Sst).take(6))
        .chain(std::iter::repeat(CellType::Vip).take(4)).collect();
    let mut neurons: Vec<Neuron> = kinds.iter().enumerate().map(|(i, &t)| typed(i as u32, t)).collect();
    let n = neurons.len();
    // 결정적인 의사 난수
    let mut state = 12345u64;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; (state >> 11) as f64 / (1u64 << 53) as f64 };
    for pre in 0..n {
        for post in 0..n {
            if pre == post { continue; }
            // 연결 규칙: 추체 → 전부 / Pv → 추체, Pv / Sst → 추체, Pv, Vip (Sst 끼리는 안 함) / Vip → Sst
            let (allowed, prob, w) = match (kinds[pre], kinds[post]) {
                (CellType::Pyramidal, CellType::Pyramidal) => (true, 0.1, 0.10),
                (CellType::Pyramidal, _) => (true, 0.4, 0.15),
                (CellType::Pv, CellType::Pyramidal) | (CellType::Pv, CellType::Pv) => (true, 0.5, 0.15),
                (CellType::Sst, CellType::Pyramidal) | (CellType::Sst, CellType::Pv) | (CellType::Sst, CellType::Vip) => (true, 0.5, 0.15),
                (CellType::Vip, CellType::Sst) => (true, 0.8, 0.60),
                _ => (false, 0.0, 0.0),
            };
            if allowed && next() < prob {
                let mut s = Synapse::new(post as u32, w);
                // 흥분성 → Sst 는 촉진형
                s.facilitating = kinds[pre] == CellType::Pyramidal && kinds[post] == CellType::Sst;
                neurons[pre].synapses.push(s);
            }
        }
    }
    let mut counts = [0u32; 4];
    for tick in 1..=300u64 {
        for (i, nr) in neurons.iter_mut().enumerate() {
            match kinds[i] {
                CellType::Pyramidal => nr.receive(0.25 + 0.1 * ((i % 5) as f64) / 5.0),
                CellType::Vip => nr.receive(vip_drive),
                CellType::Sst => nr.receive(sst_drive),
                _ => {}
            }
        }
        let mut fired = Vec::new();
        for nr in neurons.iter_mut() {
            nr.decay();
            if let Some(d) = nr.try_fire(tick, 0.5, 0.0, true) { fired.push((nr.id as usize, d)); }
        }
        for (pre, deliveries) in fired {
            let slot = match kinds[pre] { CellType::Pyramidal => 0, CellType::Pv => 1, CellType::Sst => 2, _ => 3 };
            counts[slot] += 1;
            for (target, w) in deliveries { neurons[target as usize].receive(w); }
        }
    }
    counts
}
