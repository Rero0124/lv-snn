use crate::neuron::NeuronId;
use serde::{Deserialize, Serialize};

/// 시냅스: 목적지 뉴런 + 가중치
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub target: NeuronId,
    pub weight: f64,
}

impl Synapse {
    pub fn new(target: NeuronId, weight: f64) -> Self {
        Self { target, weight }
    }
}
