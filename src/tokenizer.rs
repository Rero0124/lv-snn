use serde::{Deserialize, Serialize};

/// 토큰 길이 기반 발화 보너스
/// 단어(4+글자)가 출력 핵심이므로 가장 높게, 자모는 보조 역할
pub fn fire_bonus(token: &str) -> f64 {
    let len = token.chars().count();
    match len {
        1 => 0.6,       // 자모/단일 글자: 보조 (발화는 쉽지만 출력 기여 낮음)
        2 => 0.8,       // bigram
        3 => 0.9,       // trigram
        4..=5 => 1.2,   // 짧은 단어: 핵심 출력 단위
        6..=10 => 1.0,  // 일반 단어
        _ => 0.7,       // 긴 문장/구
    }
}

/// 텍스트를 여러 단위로 분해한 결과
pub struct TextTokens {
    pub original: String,
    pub chars: Vec<String>,
    pub jamo: Vec<String>,
    pub words: Vec<String>,
    pub bigrams: Vec<String>,
    pub trigrams: Vec<String>,
    pub fourgrams: Vec<String>,
}

const CHOSEONGS: [char; 19] = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
];

const JUNGSEONGS: [char; 21] = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
];

const JONGSEONGS: [char; 28] = [
    '\0', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
];

/// 한글 글자 → 자모 분해 (초성, 중성, 종성)
fn decompose_hangul(c: char) -> Option<Vec<char>> {
    let code = c as u32;
    if !(0xAC00..=0xD7A3).contains(&code) {
        return None;
    }
    let offset = code - 0xAC00;
    let cho_idx = (offset / 588) as usize;
    let jung_idx = ((offset % 588) / 28) as usize;
    let jong_idx = (offset % 28) as usize;

    let mut result = vec![CHOSEONGS[cho_idx], JUNGSEONGS[jung_idx]];
    if jong_idx > 0 {
        result.push(JONGSEONGS[jong_idx]);
    }
    Some(result)
}

/// 자모 배열 → 한글 문자열 재조합
/// 초성+중성(+종성) 패턴을 인식해서 음절로 합침
pub fn compose_jamo(jamo_chars: &[char]) -> String {
    let mut result = String::new();
    let mut i = 0;

    while i < jamo_chars.len() {
        let c = jamo_chars[i];

        // 초성인지 확인
        let cho = CHOSEONGS.iter().position(|&x| x == c);
        if let Some(cho_idx) = cho {
            // 다음이 중성인지 확인
            if i + 1 < jamo_chars.len() {
                let jung = JUNGSEONGS.iter().position(|&x| x == jamo_chars[i + 1]);
                if let Some(jung_idx) = jung {
                    // 그 다음이 종성인지 확인 (종성이면서 그 뒤에 중성이 오지 않는 경우)
                    let mut jong_idx = 0usize;
                    let mut consumed = 2;

                    if i + 2 < jamo_chars.len() {
                        let jong = JONGSEONGS.iter().position(|&x| x == jamo_chars[i + 2]);
                        if let Some(j) = jong {
                            if j > 0 {
                                // 종성 뒤에 중성이 오면 종성이 아니라 다음 초성
                                let next_is_jung = i + 3 < jamo_chars.len()
                                    && JUNGSEONGS.iter().any(|&x| x == jamo_chars[i + 3]);
                                if !next_is_jung {
                                    jong_idx = j;
                                    consumed = 3;
                                }
                            }
                        }
                    }

                    let code = 0xAC00 + (cho_idx as u32) * 588 + (jung_idx as u32) * 28 + jong_idx as u32;
                    if let Some(syllable) = char::from_u32(code) {
                        result.push(syllable);
                        i += consumed;
                        continue;
                    }
                }
            }
        }

        // 조합 안 되면 그대로 출력
        result.push(c);
        i += 1;
    }

    result
}

/// 출력 후처리: 종성 없는 음절 + 자음 자모 → 종성으로 합침
/// 예: "이러" + "ㄴ" → "이런", "하" + "ㄹ" → "할"
pub fn merge_trailing_jamo(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut result = String::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        let code = c as u32;

        // 한글 음절이고 종성이 없는 경우
        if (0xAC00..=0xD7A3).contains(&code) && (code - 0xAC00) % 28 == 0 {
            // 다음 글자가 종성 가능한 자음 자모인지 확인
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                let jong = JONGSEONGS.iter().position(|&x| x == next);

                if let Some(jong_idx) = jong {
                    if jong_idx > 0 {
                        // 그 다음이 중성(모음)이면 합치지 않음 (다음 초성임)
                        let next_is_jung = i + 2 < chars.len()
                            && JUNGSEONGS.iter().any(|&x| x == chars[i + 2]);

                        if !next_is_jung {
                            // 종성 합치기
                            let new_code = code + jong_idx as u32;
                            if let Some(merged) = char::from_u32(new_code) {
                                result.push(merged);
                                i += 2;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        result.push(c);
        i += 1;
    }

    result
}

/// 텍스트 → 다중 토큰 분해
pub fn tokenize(input: &str) -> TextTokens {
    let input = input.trim();
    let original = input.to_string();

    let words: Vec<String> = input
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    let char_vec: Vec<char> = input.chars().collect();

    // 단일 글자 (공백 제외, 중복 제거)
    let mut chars: Vec<String> = Vec::new();
    let mut seen_chars: std::collections::HashSet<char> = std::collections::HashSet::new();
    for &c in &char_vec {
        if !c.is_whitespace() && seen_chars.insert(c) {
            chars.push(c.to_string());
        }
    }

    // 자모 분해 (초성+중성+종성, 중복 제거)
    let mut jamo: Vec<String> = Vec::new();
    let mut seen_jamo: std::collections::HashSet<char> = std::collections::HashSet::new();
    for &c in &char_vec {
        if let Some(parts) = decompose_hangul(c) {
            for j in parts {
                if seen_jamo.insert(j) {
                    jamo.push(j.to_string());
                }
            }
        }
    }

    let bigrams = if char_vec.len() >= 2 {
        char_vec.windows(2)
            .map(|w| w.iter().collect::<String>())
            .collect()
    } else {
        Vec::new()
    };

    let trigrams = if char_vec.len() >= 3 {
        char_vec.windows(3)
            .map(|w| w.iter().collect::<String>())
            .collect()
    } else {
        Vec::new()
    };

    let fourgrams = if char_vec.len() >= 4 {
        char_vec.windows(4)
            .map(|w| w.iter().collect::<String>())
            .collect()
    } else {
        Vec::new()
    };

    TextTokens {
        original,
        chars,
        jamo,
        words,
        bigrams,
        trigrams,
        fourgrams,
    }
}

/// 문자열 → 뉴런 인덱스 (결정적 해시)
pub fn hash_to_index(text: &str, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let mut hash: u64 = 5381;
    for byte in text.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    (hash as usize) % len
}
