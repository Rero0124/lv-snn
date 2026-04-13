use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionType {
    Input,
    Emotion,
    Reason,
    Memory,
    Hippocampus,
    Output,
}

impl fmt::Display for RegionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionType::Input => write!(f, "입력"),
            RegionType::Emotion => write!(f, "감정"),
            RegionType::Reason => write!(f, "이성"),
            RegionType::Memory => write!(f, "기억"),
            RegionType::Hippocampus => write!(f, "해마"),
            RegionType::Output => write!(f, "출력"),
        }
    }
}

/// 구역 순서 (뉴런 ID 계산용)
pub const REGIONS: [RegionType; 6] = [
    RegionType::Input,
    RegionType::Emotion,
    RegionType::Reason,
    RegionType::Memory,
    RegionType::Hippocampus,
    RegionType::Output,
];
