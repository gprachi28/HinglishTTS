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

### Models: Qwen3-TTS vs Fish Audio S2

| Model | Architecture | Strength | Challenge |
|---|---|---|---|
| **Qwen3-TTS** | Autoregressive codec (Qwen-Omni) | Reliable, 100% success rate, balanced language handling | Moderate English phoneme accuracy |
| **Fish Audio S2 Pro** | Codec-based (VQ-VAE) | Exceptional on Hindi-dominant patterns, excellent inter-sentential code-switching | ~30% silent failures, unreliable on pure Hindi input |

Both models receive voice cloning references (code-switched speech) to prime the synthesizer for code-switching behavior.

### Evaluation Metrics: Custom Framework for Code-Switching

> **Critical Insight**: Traditional metrics (WER, BLEU, MCD, F0) do **NOT** capture code-switching quality.
> A model can have low WER but fail on naturalness; high F0 continuity but wrong phonemes.
> Custom metrics are necessary.

#### Core Metrics (Code-Switching Phonetic Index — CSPI)

| Metric | What it measures | Why traditional metrics fail |
|---|---|---|---|
| **H-Index** | % Hindi tokens correctly recognized (via ASR) | WER treats all tokens equally; doesn't isolate language-specific failures |
| **E-Index** | % English tokens correctly recognized (via ASR) | A model failing on "meeting"→"making" passes WER if recognized as English word |
| **H-Phoneme Accuracy** | % Hindi phonemes pronounced correctly (char-level similarity) | MCD (spectral distortion) doesn't capture mispronunciation; "meeting"→"making" has identical spectrum  |
| **E-Phoneme Accuracy** | % English phonemes pronounced correctly | Standard ASR+phoneme metrics assume monolingual input |

**CSPI = 0.25 × (H-Index + E-Index + H-Phoneme + E-Phoneme)** OR language-aware weighted version

**Language-Aware Refinement**: Weight metrics by Hindi/English token ratio in each sentence
- Hindi-dominant sentences (70% HI) → H-Index/H-Phoneme weighted 70%, E-Index/E-Phoneme weighted 30%
- Reflects linguistic reality: errors in matrix language more noticeable

#### Supporting Metrics

| Metric | What it measures |
|---|---|
| **VQS** (Vector Quantization Stability) | Do VQ codes remain stable across code-switching boundaries? |
| **Reliability** | % of test cases producing usable output (captures silent failures) |

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
| **Phase 0** | Test set generation (7 patterns, 20 test sentences) | ✅ Complete |
| **Phase 1** | Golden set selection (300 sentences, 6–12 tokens) | ✅ Complete |
| **Phase 1.5** | Model compatibility testing (Qwen3-TTS, Fish Audio S2) | ✅ Complete |
| **Phase 1.5** | Custom metrics framework (CSPI: H/E-Index + Phoneme-Acc) | ✅ Complete |
| **Phase 1.5** | Language-aware CSPI refinement | ✅ Complete |
| **Phase 1.5** | VQ stability analysis | ✅ Complete |
| **Phase 2** | Golden set synthesis (300 sentences, both models) | 📌 Pending |
| **Phase 3** | Human evaluation (MOS, linguist annotation) | 📌 Pending |
| **Phase 4** | Analysis & write-up | 📌 Pending |

**Latest (2026-03-25):** CSPI framework complete. **Qwen3-TTS**: Equal-weight CSPI 0.6228, 100% reliability, balanced language handling. **Fish Audio S2**: Language-aware CSPI 0.6433 (hidden strength on Hindi-dominant patterns), but 65% effective reliability (30% silent failures on pure Hindi inputs). Ready for Phase 2 golden set synthesis.

## Project Structure

```
HinglishTTS/
├── data/
│   ├── codeswitching.py            # Controlled test set generator (7 patterns)
│   ├── codeswitched/               # Generated benchmark sentences (CSV)
│   ├── golden/                     # Golden set (300 sentences, 6–12 tokens)
│   ├── script_variants.py          # Roman/Devanagari/Mixed conversions
│   └── devanagari_map.py           # Hand-curated Roman→Devanagari dictionary
├── evaluation/
│   ├── compatibility/
│   │   ├── run_tests.py            # Model compatibility test harness
│   │   ├── test_set.csv            # 20 test sentences (all 7 patterns)
│   │   ├── adapters/               # Model synthesis wrappers
│   │   │   ├── base.py             # Adapter interface
│   │   │   ├── qwen3_tts.py        # ✅ Qwen3-TTS voice cloning
│   │   │   └── fish_audio_s2.py    # ✅ Fish Audio S2 voice cloning
│   │   ├── run_metrics.py          # Execute all CSPI metrics
│   │   ├── compute_hindex.py       # Hindi token recognition
│   │   ├── compute_eindex.py       # English token recognition
│   │   ├── compute_phoneme_accuracy.py    # Language-specific phoneme accuracy
│   │   ├── compute_cspi.py         # Equal-weight CSPI
│   │   ├── compute_cspi_refined.py # Language-aware CSPI
│   │   ├── compute_vqs.py          # VQ stability analysis
│   │   └── results/                # Metric outputs (per-model)
│   │       ├── qwen3_tts/          # Qwen3-TTS results
│   │       └── fish_audio_s2/      # Fish Audio S2 results
│   └── human_eval/                 # Phase 3: MOS collection setup
├── models/
│   ├── qwen3_tts/                  # Qwen3-TTS installation
│   └── fish_audio_s2/              # Fish Audio S2 installation
├── README.md                       # This file
├── PROJECT_PLAN.md                 # Detailed phased roadmap
└── tests/                          # Unit tests
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

### Phase 1.5: CSPI Framework & Model Evaluation (✅ Complete)

#### Code-Switching Phonetic Index (CSPI) Rankings

**Equal-Weight CSPI** (all metrics weighted 25% each):
1. **Qwen3-TTS: 0.6228** — Balanced performance across all dimensions
2. **Fish Audio S2: 0.6157** — Strong Hindi, weaker English

**Language-Aware CSPI** (weighted by Hindi/English token ratio):
1. **Fish Audio S2: 0.6433** — Exceptional on Hindi-dominant patterns (CS-02, CS-07)
2. **Qwen3-TTS: 0.6327** — Consistent across pattern types

#### Per-Model Summary

| Model | H-Index | E-Index | H-Phoneme | E-Phoneme | Reliability | CSPI |
|---|---|---|---|---|---|---|
| **Qwen3-TTS** | 67.95% | 50.00% | 0.7929 | 0.5188 | 100% | 0.6228 |
| **Fish Audio S2** | 73.13% | 48.39% | 0.7806 | 0.4839 | 65% | 0.6157* |

*Fish Audio effective reliability: 65% (30% of test files fail silently; effective CSPI ≈ 0.45 including failures)

#### Key Findings

- **Traditional metrics fail for code-switching**: WER, F0, and standard prosody metrics don't capture the "meeting"→"making" error visible in E-Phoneme accuracy.
- **Custom metrics are necessary**: CSPI reveals that both models have distinct failure modes (Qwen3 weak on English insertions, Fish Audio weak on pure-Hindi sequences).
- **Language-aware weighting matters**: Fish Audio's strength on Hindi-dominant patterns (70% of real conversations) becomes visible only with linguistic weighting.
- **Devanagari:** 18/20 pass (90%), 2.10s per sentence
- **Mixed:** 18/20 pass (90%), 2.02s per sentence
- **Known limitation:** Fails on CS-06 (numerical/entity code-switching) patterns

**CosyVoice 3 Detailed Results:**
- **Roman:** 20/20 pass (100%), avg 8.63s per sentence
- **Devanagari:** 6/20 valid audio (30%) — 14/20 near-silent ❌
- **Mixed:** 17/20 valid audio (85%), avg 6.95s per sentence
- **Fundamental limitation — Devanagari tokenizer gap:**
  CosyVoice 3 uses a Qwen LLM as its text encoder, sharing the Qwen BPE tokenizer with Qwen3-TTS. However, the specific CosyVoice 3 checkpoint (`Fun-CosyVoice3-0.5B-2512`) was fine-tuned on predominantly Latin/Chinese data, leaving Devanagari Unicode (U+0900–U+097F) outside its effective token coverage. When fed Devanagari text, the tokenizer maps most characters to unknown/rare tokens, producing a near-empty speech token stream. The vocoder then outputs a few milliseconds of near-silence (typically 0.04–0.16s, amplitude < 0.01). No exception is raised — the model completes without error — making this a silent failure that requires audio validation to detect.

  Importantly, **this is not an architectural flaw** — it is a fine-tuning data gap specific to this checkpoint. Qwen3-TTS succeeds on Devanagari because its fine-tuning included Hindi/Devanagari speech data; CosyVoice 3's did not.

- **Benchmark implication:** CosyVoice 3 is evaluated on **Roman and mixed-script inputs only**. It is excluded from Set A (Devanagari) synthesis and from TPI comparison, which requires all three script variants.

### Phase 1.5: Linguistic Quality Evaluation (✅ Complete, 2026-03-25)

**All four models evaluated on canonical metrics (20-sentence test set, all 7 code-switching patterns):**

| Model | H-Index ↑ | WER Mixed | PIER ↓ | F0 Mixed (Hz) | LID ↑ | Reliability |
|---|---|---|---|---|---|---|
| **Qwen3-TTS** | 67.95% | **0.440** | **0.500** | 33.00 | **0.600** | ✅ 100% |
| **Fish Audio S2** | **71.43%** ✅ | 0.605 | 0.643 | **25.27** ✅ | 0.585 | ⚠️ 70% (30% silent) |
| **XTTS-v2** | 42.25% | 1.890 | 0.804 | 29.91 | 0.455 | ✅ 100% |
| **CosyVoice 3** | 0.0% ❌ | 0.763 | 0.679 | 24.25 | 0.304 | ⚠️ Roman/Mixed only |

**Key findings:**

1. **Qwen3-TTS is the safest production choice** — 67.95% H-Index, functional across all three script variants, zero silent failures, lowest PIER (0.500).

2. **Fish Audio S2 Pro is competitive but unreliable** — **Highest H-Index (71.43%)**, **best-in-class F0 prosody (25.27 Hz)**, but ~30% of test files produce no audio. Silent failures are a critical reliability issue; root cause requires debugging.

3. **XTTS-v2 is moderate** — H-Index 42.25%, better prosody than Qwen3-TTS on individual files but worse overall phonetic fidelity. Reliable (no silent failures).

4. **CosyVoice 3 is weak** — H-Index 0% (zero Hindi tokens recognized), Devanagari synthesis fundamentally broken (tokenizer gap). Only viable for Roman/Mixed evaluation.

**Recommendation:**
- **Primary:** Qwen3-TTS (proven reliability + competitive performance)
- **Secondary:** Fish Audio S2 Pro IF silent failures are resolved
- **Baseline:** XTTS-v2 for acoustic comparison
- **Exclude from Phase 2:** CosyVoice 3 Devanagari (non-viable)

**Full analysis:** See `LINGUISTIC_QUALITY_REPORT.md` with detailed four-model comparison, per-pattern breakdowns, and silent failure diagnosis.

### Phase 2+: Full Benchmark (⏳ In Progress)

*MOS and SPN from human evaluation will be collected after Phase 2 audio synthesis.*

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

