# LV-SNN

A brain-inspired Spiking Neural Network (SNN) for Korean conversational AI.

Text is decomposed into multi-level tokens (original, word, n-gram, character, jamo) stored as synapses. Signals propagate tick-by-tick across brain-like regions, with STDP and hippocampal pattern consolidation enabling autonomous learning.

## Architecture

```
Input text → Tokenize (original/word/n-gram/char/jamo)
           → Activate Input region neurons
           → Tick propagation: Input → Emotion/Reason/Storage → Output
           → Output gate → compose_response (4-layer assembly)
           → Hippocampus path recording + pattern consolidation
```

### Regions (region.rs)

| Region | Neurons | Role | Connects to |
|--------|---------|------|-------------|
| Input | 128 | Token hash → neuron mapping, signal origin | Emotion, Reason, Storage |
| Emotion | 64 | Low threshold (0.5), fast intuitive response | Reason, Storage, Output |
| Reason | 64 | High threshold (0.7), evidence-based processing | Emotion, Storage, Output |
| Storage | 128 | Token synapse storage, hippocampal pattern store | Emotion, Reason, Output |
| Output | 64 | Response token collection, dissipates when gate closes | (no outgoing connections) |

### Key Mechanisms

- **3-State Neuron** (neuron.rs): Decay (< threshold x 0.85) / Pass / Diverge (> threshold x 1.5, 1.3x signal boost). Top-K (10) synapses per neuron
- **STDP** (network.rs): Spike-timing-dependent plasticity. Pre→post causal = LTP (strengthen), anti-causal = LTD (weaken)
- **Hippocampus** (hippocampus.rs): Tracks neuron-level path patterns (length 3) → consolidates frequent patterns to Storage → removes transferred patterns from hippocampus
- **Output Gate** (network.rs): Output neuron threshold set high (1.0) → gate opens for 5 ticks after sufficient signal accumulation
- **Cooldown**: Recently used synapse signal reduction (0.15) for response diversity
- **Pattern Merging** (consolidate_patterns): Merges consecutively output jamo/char tokens into single synapses (e.g., "ㅂ"+"습니다" → "ㅂ습니다")
- **Jamo Reassembly** (tokenizer.rs): Reassembles jamo-level output into Hangul syllables (compose_jamo)
- **Multi-layer Response Assembly** (compose_response): Original (L1) → Word combination (L2) → Char assembly (L3) → Jamo reassembly (L4) → Fallback

### Feedback

- **positive**: Output synapse weight + strength x 0.3
- **negative**: Output synapse weight - strength x 0.3 x 1.5 (strong) + path synapse - strength x 0.3 x 0.5
- **partial**: Per-token scoring for fine-grained synapse adjustment

## Build & Run

```bash
cargo build --release

# Server mode
./target/release/lv-snn --serve              # http://127.0.0.1:3000
./target/release/lv-snn --serve --port 8080  # Custom port

# Interactive mode
./target/release/lv-snn
```

## HTTP API (server.rs)

| Endpoint | Method | Request | Description |
|----------|--------|---------|-------------|
| /fire | POST | `{"text": "안녕"}` | Fire + response |
| /teach | POST | `{"input": "안녕", "target": "반가워!"}` | Target-based learning |
| /feedback | POST | `{"fire_id": N, "positive": bool, "strength": 0.8}` | Feedback |
| /feedback_partial | POST | `{"fire_id": N, "token_scores": [["word", 0.5]]}` | Partial feedback |
| /status | GET | - | Status (lock-free atomic) |
| /save | POST | - | Save to DB |

## Training Scripts

### Autonomous Exploration (ai_train.py)

Ollama (gemma3:4b) generates inputs, SNN explores responses autonomously.
The LLM only judges O/X (correct/incorrect) — it does not provide answers.

```
Input → SNN fires "시켜 먹을까?" → Ollama "X" → weaken + re-fire
      → "무슨 일이야?" → Ollama "X" → weaken + re-fire
      → "맞아 나도 배고파" → Ollama "O" → strengthen
      (After 5 failures, Ollama suggests an answer → teach)
```

```bash
ollama pull gemma3:4b
python3 scripts/ai_train.py --topic "음식,여행,감정" --duration 1800
```

### Fast Training (fast_train.py)

Iterative training with local 1:N conversation data. Works without LLM.

```bash
python3 scripts/fast_train.py --rounds 5 --no-llm    # Token matching only
python3 scripts/fast_train.py --rounds 10             # With Ollama semantic eval
```

## Data

- `data/conversation_multi.json` — 1:N conversation pairs (59 inputs, ~4-5 responses each)
- `data/network.redb` — Synapse + network state persistent storage (auto-generated)

## Dependencies

| Crate | Purpose |
|-------|---------|
| actix-web 4 | HTTP API server |
| tokio 1 | Async runtime |
| rayon 1.10 | Parallel neuron computation |
| redb 3.1 | Embedded KV DB (synapse + state persistence) |
| serde + serde_json | Serialization |
| signal-hook 0.3 | Graceful shutdown |
| uuid 1 | Neuron/synapse ID generation |
