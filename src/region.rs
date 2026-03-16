use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionType {
    Input,
    Output,
    Emotion,
    Reason,
    Storage,
}

impl fmt::Display for RegionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionType::Input => write!(f, "입력"),
            RegionType::Output => write!(f, "출력"),
            RegionType::Emotion => write!(f, "감정"),
            RegionType::Reason => write!(f, "이성"),
            RegionType::Storage => write!(f, "기억"),
        }
    }
}

impl RegionType {
    /// 연결 가능한 대상 구역
    pub fn targets(&self) -> &[RegionType] {
        use RegionType::*;
        match self {
            Input => &[Emotion, Reason, Storage],
            Emotion => &[Reason, Storage, Output],
            Reason => &[Emotion, Storage, Output],
            Storage => &[Emotion, Reason, Output],
            Output => &[Output], // 내부 순환: 발화 후 감쇠로 자연 소멸
        }
    }
}
