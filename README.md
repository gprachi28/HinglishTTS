# HinglishTTS-Bench

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

> **Work in Progress** — This project is actively under development.

The first systematic evaluation benchmark for Hindi-English code-switched speech synthesis. We evaluate state-of-the-art TTS systems on linguistically controlled Hinglish sentences spanning 7 switching patterns, measuring prosodic continuity at language boundaries, Hindi phoneme fidelity under transliteration, and naturalness via human evaluation with native speakers.

---

## Background: Why a Benchmark, Not a Model

This project began as a VITS fine-tuning pipeline for Hinglish TTS — training on LJSpeech (English) + IndicVoices-R (Hindi) with synthetic code-switched sentences generated via Equivalence Constraint theory.

Three things became clear:

1. **E2E models already handle codeswitching.** Models like Qwen3-TTS (built on Qwen-Omni) produce reasonable Hinglish output zero-shot. Training a VITS pipeline on 2021-era architecture no longer addresses a real gap.

2. **The real problem is nobody has measured _how well_ they handle it.** There are no benchmarks that evaluate TTS specifically on linguistically diverse code-switching patterns in Hindi-English, with controlled variables for script, switch type, and mixing intensity.

3. **Existing Hinglish text corpora are unsuitable for TTS.** GLUECoS depends on a deprecated Twitter API. L3Cube-HingLID is noisy Twitter data — informal and unnatural for speech synthesis. Synthetic generation with linguistic constraints is the only viable path for a clean, controlled test set.

The sentence generation infrastructure built in phase one (`data/codeswitching.py`) now serves as the **controlled test set generator** — the hard part of any evaluation framework.

### Positioning Against Existing Work

| Benchmark | Year | Languages | Focus | Gap |
|---|---|---|---|---|
| MANGO (AI4Bharat) | 2025 | Hindi, Tamil | General naturalness (246k ratings) | No code-switching analysis |
| CS3-Bench | 2025 | Mandarin-English | Code-switched TTS | No Indic languages |
| DISPLACE-M | 2026 | Hindi-English | Medical domain conversations | Domain-specific, not general |
| **HinglishTTS-Bench** | **2026** | **Hindi-English** | **Code-switching + script-agnosticism** | **This work** |

---

## The Three-Tier Framework

### Tier 1: Input Diversity

The same sentences are fed in three script variants, isolating the impact of transliteration:

| Set | Script | Purpose |
|---|---|---|
| **Set A** (Baseline) | Pure Devanagari (human-verified) | Model's maximum potential |
| **Set B** (Experimental) | Romanized Hinglish | Real-world user input |
| **Set C** (Mixed) | Devanagari for Hindi, Roman for English | Script-normalized middle ground |

### Tier 2: Models Under Evaluation

| Model | Architecture | Params | Multilingual | Ecosystem |
|---|---|---|---|---|
| Qwen3-TTS | Autoregressive codec (Qwen-Omni backbone) | 1.7B | Yes (incl. Hindi) | Alibaba |
| CosyVoice 3 | Flow-matching + LLM text encoder | — | Yes (incl. Hindi) | Alibaba |
| XTTS v2 | Autoregressive | — | Yes | Coqui (independent) |
| Fish Audio S2 Pro | Codec-based | — | Yes | Independent |

#### Model Ecosystem Grouping

CosyVoice 3 and Qwen3-TTS share the same BPE tokenizer (Qwen's tokenizer) because CosyVoice 3 uses a Qwen LLM as its text encoder backbone. This has implications for fair comparison:

- **Tokenization is identical**: both models split Hinglish input (Roman, Devanagari, mixed) into the same subwords. OOV handling at switch boundaries, Devanagari coverage, and intraword splitting (`unfriend`, `डेडलाइन`) are identical.
- **Training data overlap**: both are Alibaba-trained models, likely pretrained on overlapping multilingual corpora with Hindi/Devanagari coverage. Differences in Hindi phonetic fidelity scores between them reflect architecture and fine-tuning, not data access.
- **Cross-ecosystem comparison is the headline**: when Alibaba models outperform XTTS v2 or Fish Audio on Hindi metrics, it may reflect shared training advantage rather than architectural superiority.

Results are therefore reported in two clusters:

| Cluster | Models | Shared components |
|---|---|---|
| **Alibaba ecosystem** | Qwen3-TTS, CosyVoice 3 | Tokenizer, pretraining corpus, Qwen LLM |
| **Independent** | XTTS v2, Fish Audio S2 Pro | Nothing shared |

Within-cluster comparisons (Qwen3-TTS vs CosyVoice 3) isolate **architectural** differences. Cross-cluster comparisons measure **ecosystem advantage** as a finding in its own right.

### Tier 3: Evaluation Metrics

#### A. Objective Acoustic Metrics

| Metric | What it measures |
|---|---|
| **PIER** (Point-of-Interest Error Rate) | WER calculated _only_ at switch-point words, not the full sentence |
| **Boundary F0 RMSE** | Pitch continuity at the exact HI↔EN junction — spikes or breaks indicate poor prosody |
| **MCD** (Mel-Cepstral Distortion) | Overall spectral quality |

#### B. Linguistic & Transliteration Metrics

| Metric | What it measures |
|---|---|
| **Phonetic Fidelity (H-Index)** | Do Hindi words in Roman script get Hindi phonemes? ("Samajh" → /sʌmʌdʒʰ/ not /smæʃt/) |
| **LID Confidence at Boundary** | Audio-based language ID at switch points — low confidence = natural blending |
| **TPI** (Transliteration Penalty Index) | `(MOS_Devanagari - MOS_Roman) / MOS_Devanagari * 100` — measures script-agnosticism |

TPI interpretation:
- **0–5%** — Script Agnostic (model doesn't care about input script)
- **20–40%** — Script Biased (understands the language, loses prosody in transliteration)
- **>50%** — Transliteration Blind (fails to trigger correct phonetic engine)

#### C. Human Evaluation

| Metric | What it measures |
|---|---|
| **MOS** | Overall naturalness (native Hinglish speakers, Prolific/MTurk) |
| **Switch-Point Naturalness (SPN)** | Listeners rate _only_ the transition on a 1–5 scale |
| **Transliteration Robustness Score** | MOS_Roman / MOS_Devanagari for the same sentence |

---

## Test Set: 7 Code-Switching Patterns

5,000 sentences stratified by pattern, CMI range, and register:

| ID | Pattern | Example |
|---|---|---|
| CS-01 | Noun Insertion | "Main **office** ja raha hoon" |
| CS-02 | Verb Grafting | "File **download** kar lo" |
| CS-03 | Tag Switching | "Baarish ho rahi hai, **isn't it?**" |
| CS-04 | Clause Boundary | "I was worried ki **train miss ho jayegi**" |
| CS-05 | Technical / Slang | "Ye content kaafi **cringe** hai" |
| CS-06 | Numerical / Entity | "Meeting **Thursday** ko **3 PM** par hai" |
| CS-07 | Intraword | "Usne mujhe **unfriend** kar-diya" |

Quality controlled by **Code Mixing Index (CMI)**: 0.1 ≤ CMI ≤ 0.7

### Sentence Length Design

The full 5,000-sentence corpus is unconstrained in length (minimum 4 tokens). The **golden set** (300 sentences used for human evaluation) is filtered to **6–12 tokens** to keep evaluation consistent and reduce cognitive load for raters.

CS-07 (Intraword) sentences are structurally short (~5 tokens) by nature — e.g. "Usne mujhe unfriend kar diya" — and are allowed a relaxed lower bound of 5 tokens in the golden set.

#### Length as a Metric Confounder

A **confounder** is a variable that affects your results but isn't what you're trying to study — it gets in the way of the measurement you actually care about. Here, sentence length is a confounder: if CS-07 sentences are always short and CS-04 sentences are always long, a difference in scores could be caused by length, not by the switching pattern.

Sentence length affects some metrics and must be controlled:

| Metric | Length effect | How to handle |
|---|---|---|
| **PIER** | Multi-switch sentences have more evaluation points | Normalize to *per switch point*, not per sentence |
| **MCD** | Longer audio → higher absolute distortion | Always compute *per frame* (standard practice) |
| **H-Index** | Short CS-07 sentences (~1 Hindi token) give unstable estimates | Weight by Hindi token count; report confidence intervals |
| **MOS / TPI** | Long sentences introduce variation unrelated to switch quality | Constrain golden set to 6–12 tokens |
| **Boundary F0 RMSE** | Local ±100ms window, unaffected by total length | No adjustment needed |
| **SPN** | 2-second clip centered on boundary, unaffected by total length | No adjustment needed |

#### Length as an Analysis Dimension

Beyond controlling for length as a confounder, results are also reported **stratified by length** as a deliberate analysis axis:

| Stratum | Token range | Characteristic |
|---|---|---|
| Short | 5–8 tokens | Isolated switch, minimal prosodic context |
| Long | 9–12 tokens | Multiple switch points, richer context |

This separates two distinct failure modes: a model that fails on *short* sentences but not long ones has a **prosodic context dependency** — it needs surrounding words to correctly map phonemes. A model that fails on both has a fundamental **switch-boundary failure**.

It also disentangles pattern effects from length effects — CS-07 intraword sentences are always short, CS-04 clause boundary sentences tend to be long. Without length stratification, per-pattern rankings could reflect length differences rather than switching difficulty.

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| **Phase 0** | Test set generation (7 patterns, 5000 sentences) | ✅ Complete |
| **Phase 1** | Golden set selection (300 sentences) | ✅ Complete |
| **Phase 1** | Golden set Devanagari verification | 🔄 In progress |
| **Phase 1.5** | Model compatibility testing (test harness) | ✅ Complete |
| **Phase 1.5** | XTTS-v2 adapter (Phase 1.5 ✓ 90% pass) | ✅ Complete |
| **Phase 1.5** | Qwen3-TTS adapter | ✅ Complete |
| **Phase 1.5** | CosyVoice 3 adapter | 🔄 In progress |
| **Phase 1.5** | Fish Audio S2 adapter | 📌 Pending |
| **Phase 2** | Audio synthesis pipeline | 📌 Pending |
| **Phase 3** | Objective acoustic metrics (PIER, F0, MCD) | 📌 Pending |
| **Phase 4** | Human evaluation (MOS, SPN) | 📌 Pending |
| **Phase 5** | Analysis & write-up | 📌 Pending |

**Latest (Phase 1.5):** XTTS-v2 successfully tested with 90% pass rate across Roman, Devanagari, and mixed-script inputs. Known limitation: numerical/entity code-switching (CS-06).

## Project Structure

```
HinglishTTS/
├── data/
│   ├── codeswitching.py            # Controlled test set generator (7 patterns)
│   ├── preprocess_streaming.py     # Audio preprocessing
│   ├── codeswitched/               # Generated benchmark sentences (CSV)
│   ├── golden/                     # Human-verified Devanagari + recordings
│   └── devanagari_map.py           # Hand-curated Roman→Devanagari dictionary
├── normalization/
│   └── g2p.py                      # G2P pipeline for reference transcriptions
├── evaluation/
│   ├── compatibility/
│   │   ├── run_tests.py            # Phase 1.5 test harness
│   │   ├── test_set.csv            # 20 test sentences (all 7 patterns)
│   │   └── adapters/               # Model wrappers
│   │       ├── base.py             # Adapter interface
│   │       ├── qwen3_tts.py        # ✅ Tested
│   │       ├── xtts_v2.py          # ✅ Tested (90%)
│   │       ├── cosyvoice3.py       # 🔄 In progress
│   │       └── fish_audio_s2.py    # 📌 Pending
│   ├── synthesize.py               # Phase 2: Run models on benchmark set
│   ├── acoustic_metrics.py         # Phase 3: PIER, F0 RMSE, MCD
│   ├── tpi.py                      # Phase 3: Transliteration Penalty Index
│   └── human_eval/                 # Phase 4: MOS + SPN collection setup
├── tests/
├── CAPABILITY_REPORT.md            # Phase 1.5 results (auto-generated)
└── results/                        # Phase 2+: Per-model outputs and analysis
```

---

## Quickstart

```bash
git clone https://github.com/gprachi28/HinglishTTS.git
cd HinglishTTS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate benchmark test set
python data/codeswitching.py --num_sentences 5000

# Run tests
make test
```

---

## Results

### Phase 1.5: Model Capability Testing (✅ Complete)

Script variant compatibility across 20 test sentences (all 7 code-switching patterns):

**Alibaba Ecosystem**

| Model | Roman | Devanagari | Mixed | Latency | Status |
|---|---|---|---|---|---|
| Qwen3-TTS | ✅ Tested | ✅ Tested | ✅ Tested | — | ✅ Adapter ready |
| CosyVoice 3 | ? | ? | ? | — | 🔄 Testing in progress |

**Independent**

| Model | Roman | Devanagari | Mixed | Latency | Status |
|---|---|---|---|---|---|
| XTTS v2 | ~ 18/20 (90%) | ~ 18/20 (90%) | ~ 18/20 (90%) | 2.1s/sent | ✅ Adapter ready |
| Fish Audio S2 Pro | ? | ? | ? | — | 📌 Testing pending |

**XTTS-v2 Detailed Results:**
- **Roman:** 18/20 pass (90%), 2.67s per sentence
- **Devanagari:** 18/20 pass (90%), 2.10s per sentence
- **Mixed:** 18/20 pass (90%), 2.02s per sentence
- **Known limitation:** Fails on CS-06 (numerical/entity code-switching) patterns

### Phase 2+: Objective Metrics (⏳ In Progress)

*Full benchmark results (PIER, TPI, MCD, H-Index, MOS, SPN) will be published after Phase 2 synthesis and Phase 3 metric computation.*

| Model | MOS ↑ | SPN ↑ | PIER ↓ | TPI ↓ | H-Index ↑ |
|---|---|---|---|---|---|
| Qwen3-TTS | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| CosyVoice 3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| XTTS v2 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Fish Audio S2 Pro | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

---

## Development

```bash
make install    # create venv + install dependencies
make test       # run test suite with coverage
make lint       # flake8
make format     # black + isort
make data       # generate 5000 benchmark sentences
```

---

## Citation

```bibtex
@misc{govalkar2026hinglishbench,
  author = {Govalkar, Prachi},
  title  = {HinglishTTS-Bench: Evaluating Code-Switched Speech Synthesis Across Scripts and Switch Patterns},
  year   = {2026},
  url    = {https://github.com/gprachi28/HinglishTTS}
}
```

