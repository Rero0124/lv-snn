//! 문자 ↔ 32×32 이진 비트맵 변환
//!
//! 입력: 텍스트 → 각 글자를 fontdue로 렌더 → 32×32 이진 픽셀
//! 출력: 32×32 이진 패턴 → 레퍼런스 사전에서 해밍거리 최소 글자 복원
//!
//! 뉴런 매핑:
//!   각 글자 = 32×32 = 1024개 입력 뉴런(혹은 출력 뉴런)
//!   픽셀 값 > 임계치 → 해당 뉴런 활성
//!
//! 여러 글자가 연속으로 들어오면 순차 주입 (문장 = 글자 시퀀스)

use fontdue::{Font, FontSettings};
use std::sync::OnceLock;

pub const GRID: usize = 32;
pub const PIXELS: usize = GRID * GRID; // 1024

/// 폰트 바이너리를 빌드 시 번들
const FONT_BYTES: &[u8] = include_bytes!("../assets/NanumGothic-Regular.ttf");

static FONT: OnceLock<Font> = OnceLock::new();

fn font() -> &'static Font {
    FONT.get_or_init(|| {
        Font::from_bytes(FONT_BYTES, FontSettings::default())
            .expect("NanumGothic 폰트 로드 실패")
    })
}

/// 글자 하나를 32×32 이진 비트맵으로 렌더
/// 픽셀 값 ≥ 128이면 true (활성).
pub fn char_to_bitmap(c: char) -> [bool; PIXELS] {
    let mut out = [false; PIXELS];

    // 공백/제어 문자는 빈 비트맵
    if c.is_whitespace() || c.is_control() {
        return out;
    }

    let f = font();
    // 28px로 렌더해서 32×32 안에 여백이 생기도록
    let (metrics, bmp) = f.rasterize(c, 28.0);

    if metrics.width == 0 || metrics.height == 0 {
        return out;
    }

    // 중앙 정렬을 위한 오프셋
    let ox = (GRID.saturating_sub(metrics.width)) / 2;
    let oy = (GRID.saturating_sub(metrics.height)) / 2;

    for y in 0..metrics.height.min(GRID) {
        for x in 0..metrics.width.min(GRID) {
            let v = bmp[y * metrics.width + x];
            if v >= 128 {
                let gx = x + ox;
                let gy = y + oy;
                if gx < GRID && gy < GRID {
                    out[gy * GRID + gx] = true;
                }
            }
        }
    }
    out
}

/// 해밍 거리 (비트 다른 개수)
#[inline]
fn hamming(a: &[bool; PIXELS], b: &[bool; PIXELS]) -> u32 {
    let mut d = 0u32;
    for i in 0..PIXELS {
        if a[i] != b[i] {
            d += 1;
        }
    }
    d
}

/// 레퍼런스 사전: (글자, 비트맵) 리스트
pub struct BitmapDict {
    entries: Vec<(char, [bool; PIXELS])>,
}

impl BitmapDict {
    /// 기본 한글 음절 + ASCII 소문자/숫자/공백으로 초기화
    /// 자주 쓰이는 한글 음절 약 2,400개(KS X 1001 완성형 일부) 수준.
    pub fn new_korean_basic() -> Self {
        let mut entries = Vec::new();

        // ASCII 출력가능 문자
        for c in 0x20u32..0x7Fu32 {
            if let Some(ch) = char::from_u32(c) {
                entries.push((ch, char_to_bitmap(ch)));
            }
        }

        // 한글 음절 전체 (U+AC00 ~ U+D7A3, 11172자)
        // 메모리: 11172 * 1024비트 = 약 1.4MB (bool 저장 시 11MB)
        // 지금은 전체 로드, 필요 시 축소 가능
        for c in 0xAC00u32..=0xD7A3u32 {
            if let Some(ch) = char::from_u32(c) {
                let bmp = char_to_bitmap(ch);
                // 빈 비트맵(폰트에 없는 글리프)은 제외
                if bmp.iter().any(|&b| b) {
                    entries.push((ch, bmp));
                }
            }
        }

        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 입력 비트맵에 가장 가까운 글자 찾기
    /// 빈 비트맵(활성 픽셀 < min_pixels)이면 None 반환
    pub fn nearest(&self, bmp: &[bool; PIXELS], min_pixels: usize) -> Option<char> {
        let active = bmp.iter().filter(|&&b| b).count();
        if active < min_pixels {
            return None;
        }

        let mut best: Option<(char, u32)> = None;
        for (ch, ref_bmp) in &self.entries {
            let d = hamming(bmp, ref_bmp);
            match best {
                None => best = Some((*ch, d)),
                Some((_, bd)) if d < bd => best = Some((*ch, d)),
                _ => {}
            }
        }
        best.map(|(c, _)| c)
    }
}

/// 텍스트 → 글자별 비트맵 시퀀스
pub fn text_to_bitmaps(text: &str) -> Vec<[bool; PIXELS]> {
    text.chars()
        .filter(|c| !c.is_whitespace())
        .map(char_to_bitmap)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_korean_char() {
        let bmp = char_to_bitmap('가');
        let active = bmp.iter().filter(|&&b| b).count();
        assert!(active > 10, "한글 글자는 활성 픽셀이 충분히 있어야 함: {}", active);
    }

    #[test]
    fn nearest_roundtrip() {
        let dict = BitmapDict::new_korean_basic();
        let bmp = char_to_bitmap('한');
        let found = dict.nearest(&bmp, 5);
        assert_eq!(found, Some('한'));
    }
}
