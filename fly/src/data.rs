//! FlyWire 783 커넥톰 로더 (scripts/prepare.py 가 만든 바이너리)

use std::collections::HashMap;
use std::path::Path;

/// CSR 형태의 커넥톰. count 는 부호 있는 시냅스 수 (+흥분 / -억제)
pub struct Connectome {
    pub ids: Vec<u64>,
    pub out_start: Vec<usize>,
    pub out_post: Vec<u32>,
    pub out_count: Vec<i32>,
    /// pre 뉴런 기준 부호 (커넥톰은 Dale 법칙을 따름: 한 뉴런의 출력은 전부 같은 부호)
    pub inhibitory: Vec<bool>,
}

impl Connectome {
    pub fn load(dir: &Path) -> Self {
        let raw = std::fs::read(dir.join("ids.bin"))
            .expect("data/ids.bin 없음 — scripts/prepare.py 를 먼저 실행");
        let ids: Vec<u64> = raw.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let n = ids.len();

        let raw = std::fs::read(dir.join("conn.bin")).expect("data/conn.bin 없음");
        let m = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let body = &raw[8..];
        assert_eq!(body.len(), m * 12, "conn.bin 크기 불일치");

        let rd = |k: usize, off: usize| -> [u8; 4] { body[k * 12 + off..k * 12 + off + 4].try_into().unwrap() };
        let mut degree = vec![0usize; n];
        for k in 0..m {
            degree[u32::from_le_bytes(rd(k, 0)) as usize] += 1;
        }
        let mut out_start = vec![0usize; n + 1];
        for i in 0..n {
            out_start[i + 1] = out_start[i] + degree[i];
        }
        let mut cursor = out_start.clone();
        let mut out_post = vec![0u32; m];
        let mut out_count = vec![0i32; m];
        let mut inhibitory = vec![false; n];
        for k in 0..m {
            let pre = u32::from_le_bytes(rd(k, 0)) as usize;
            let count = i32::from_le_bytes(rd(k, 8));
            let at = cursor[pre];
            cursor[pre] += 1;
            out_post[at] = u32::from_le_bytes(rd(k, 4));
            out_count[at] = count;
            if count < 0 {
                inhibitory[pre] = true;
            }
        }
        Connectome { ids, out_start, out_post, out_count, inhibitory }
    }

    pub fn n(&self) -> usize { self.ids.len() }

    pub fn index_of(&self) -> HashMap<u64, usize> {
        self.ids.iter().enumerate().map(|(i, &id)| (id, i)).collect()
    }
}

/// Shiu et al. example.ipynb 의 오른쪽 반구 단맛 감각뉴런 (v630 기준 21개, v783 에는 20개 존재)
pub const SUGAR_GRN: [u64; 21] = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
];

/// 주둥이 내밀기 운동뉴런 MN9
pub const MN9: u64 = 720575940660219265;
