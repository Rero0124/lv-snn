# LV-SNN

A brain-inspired Spiking Neural Network (SNN) for Korean conversational AI.

Text is decomposed into multi-level tokens (original, word, n-gram, character, jamo) stored as synapses. Signals propagate tick-by-tick across brain-like regions, with STDP and hippocampal pattern consolidation enabling autonomous learning. A lock-free parallel fire engine (AtomicU64 + CAS) updates neuron activations concurrently without mutexes.

## Architecture

```
Input text → Tokenize (original/word/n-gram/char/jamo)
           → Activate Input region neurons
           → Build atomic array + parallel tick propagation:
             decay → rayon par_iter compute_fires + CAS add → SegQueue drain
           → Output gate → assemble_output (4-layer assembly)
           → Deferred post-processing (hippocampus/STDP/pruning, runs when queue empty)
```

### Regions (region.rs)

| Region | Neurons | Role | Connects to |
|--------|---------|------|-------------|
| Input | 256 | Token hash → neuron mapping, signal origin | Emotion, Reason, Storage |
| Emotion | 128 | Low threshold (0.5), fast intuitive response | Reason, Storage, Output |
| Reason | 128 | High threshold (0.7), evidence-based processing | Emotion, Storage, Output |
| Storage | 512 | Token synapse storage, hippocampal pattern store | Emotion, Reason, Output |
| Output | 128 | Response token collection, dissipates when gate closes | (no outgoing connections) |

### Key Mechanisms

- **Parallel Fire Engine** (network.rs): Lock-free parallel activation updates via AtomicU64 + CAS loop. Results collected with crossbeam SegQueue. ~3x faster than sequential
- **Probabilistic Neuron Firing** (neuron.rs): Sigmoid-based stochastic firing `p = sigmoid(activation - threshold, 0.15)`. Top-K (10) synapses per neuron
- **STDP** (network.rs): Spike-timing-dependent plasticity. Pre→post causal = LTP (strengthen), anti-causal = LTD (weaken)
- **Axonal Sprouting** (network.rs): Active neurons spontaneously form synapses with nearby neurons based on 2D grid distance (Gaussian decay)
- **Hippocampus** (hippocampus.rs): Tracks neuron-level path patterns (length 3) + co-firing neuron pairs → consolidates to Storage
- **Output Gate** (network.rs): Output neuron threshold set high (1.0) → gate opens for 5 ticks after sufficient signal accumulation
- **Cooldown**: History-based (last 10 fires) synapse signal reduction for response diversity
- **Pattern Merging** (consolidate_patterns): Merges consecutively output jamo/char tokens into single synapses
- **Jamo Reassembly** (tokenizer.rs): Reassembles jamo-level output into Hangul syllables (compose_jamo)
- **Multi-layer Response Assembly** (assemble_output): Original (L1) → Word combination (L2) → Char assembly (L3) → Jamo reassembly (L4) → Fallback

### Feedback (modifier system)

Feedback adjusts synapse **modifier** (learning signal, -1.0~1.0) instead of weight.
Weight is structural connection strength, only changed by teach/STDP. Synapses are never deleted by negative modifier, preserving paths.

- **positive**: Output synapse modifier + strength x 0.1
- **negative**: Output synapse modifier - strength x 0.1 x 1.5 (strong) + path synapse modifier - strength x 0.1 x 0.5
- **partial**: Per-token scoring for fine-grained modifier adjustment
- **Firing**: `forward = activation × weight × discount + modifier` (skipped if ≤ 0)

## Server Architecture

```
HTTP client → actix-web handler → crossbeam queue(256) → worker thread (Network sole ownership)
                                                          ├── deferred post-processing (when queue empty)
                                                          └── auto-save (5 min interval)
```

- Network owned exclusively by worker thread — no Mutex/Arc needed
- oneshot channels for request-response mapping
- Saves on channel disconnect before exit

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
| /fire | POST | `{"text": "안녕"}` | Parallel fire + response |
| /fire_sequential | POST | `{"text": "안녕"}` | Sequential fire + response (backup) |
| /fire_debug | POST | `{"text": "안녕"}` | Debug fire |
| /teach | POST | `{"input": "안녕", "target": "반가워!"}` | Target-based learning |
| /feedback | POST | `{"fire_id": N, "positive": bool, "strength": 0.8}` | Feedback |
| /feedback_partial | POST | `{"fire_id": N, "token_scores": [["word", 0.5]]}` | Partial feedback |
| /status | GET | - | Status (lock-free atomic) |
| /save | POST | - | Save to DB |

## Training Scripts

### Autonomous Exploration (ai_train.py)

Ollama (exaone3.5:7.8b) generates inputs, SNN explores responses autonomously.
The LLM only judges O/X (correct/incorrect) — it does not provide answers.

```
Input → SNN fires "시켜 먹을까?" → Ollama "X" → weaken + re-fire
      → "무슨 일이야?" → Ollama "X" → weaken + re-fire
      → "맞아 나도 배고파" → Ollama "O" → strengthen
      (After 5 failures, Ollama suggests an answer → teach)
```

```bash
ollama pull exaone3.5:7.8b
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
| rayon 1.10 | Parallel neuron computation (par_iter) |
| crossbeam 0.8 | Lock-free queue (SegQueue), channels (bounded channel) |
| redb 3.1 | Embedded KV DB (synapse + state persistence) |
| serde + serde_json | Serialization |
| signal-hook 0.3 | Graceful shutdown |
| uuid 1 | Neuron/synapse ID generation |
| rand 0.10 | Probabilistic firing |
