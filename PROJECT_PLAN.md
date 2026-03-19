# HinglishTTS-Bench: Project Plan

Detailed implementation plan, phase dependencies, and critical risk analysis.

---

## Phase 0: Test Set Completion (Weeks 1–2)

### What exists
- `data/codeswitching.py` generates 5,000 sentences across 7 builder patterns
- CMI filtering (0.1–0.7), language tagging, CSV export all working
- Tests in `tests/test_codeswitching.py` cover all builders

### Vocabulary Banks (current state)

#### Hindi Banks
| Bank | Items | Purpose |
|---|---|---|
| `HI_SUBJECTS` | 9 | General subjects incl. dative forms |
| `HI_SUBJECTS_SOCIAL` | 8 | Social/family subjects |
| `HI_SUBJECTS_FIRST_PERSON` | 2 | main, hum |
| `HI_VERBS_GENERAL` | 17 | General Hindi verb phrases |
| `HI_VERBS_FUTURE` | 6 | Future tense verb phrases |
| `HI_INTENSIFIERS` | 9 | bahut, thoda, ekdum, kaafi… |
| `HI_ADJECTIVES_PURE` | 8 | Pure Hindi adjectives only |
| `HI_TIME_SHORT` | 6 | kal, aaj, abhi, pehle… |
| `HI_TIME_LONG` | 9 | is hafte, kal subah, raat ko… |
| `HI_DISCOURSE` | 14 | yaar, arre, matlab, theek hai… |
| `HI_CONNECTORS` | 8 | ke baare mein, ki wajah se… |
| `HI_EMOTIONS` | 7 | mujhe achha laga, thak gaya… |

#### English Banks
| Bank | Items | Domain | Purpose |
|---|---|---|---|
| `EN_NP_WORK` | 20 | Professional | meeting, deadline, sprint, pull request… |
| `EN_NP_EVERYDAY` | 20 | General | phone, coffee, weekend, movie, gym… |
| `EN_NP_TECH` | 12 | Technology | laptop, wifi, app, API, deployment… |
| `EN_NP_EDUCATION` | 15 | Education | class, exam, semester, internship, scholarship… |
| `EN_NP_ENTERTAINMENT` | 15 | Entertainment | Netflix, episode, concert, reel, podcast… |
| `EN_NP_SOCIAL` | 15 | Social media | post, story, meme, caption, vibe, selfie… |
| `EN_NP_FOOD_LIFESTYLE` | 15 | Food & lifestyle | burger, cafe, brunch, cheat day, detox… |
| `EN_VERBS_WORK` | 18 | Professional | cancel, reschedule, escalate, prioritize… |
| `EN_VERBS_EVERYDAY` | 11 | General | book, order, download, install, reset… |
| `EN_VERBS_SOCIAL` | 12 | Social media | post, like, follow, stream, subscribe… |
| `EN_VERBS_QUICK_TASK` | 14 | Physical tasks | check, fix, reset, send, call, charge… |
| `EN_ADJECTIVES` | 20 | General | boring, interesting, urgent, amazing… |

#### Composite Pools
| Pool | Contents |
|---|---|
| `EN_NP_ALL` | All 7 NP banks combined |
| `EN_NP_GENERAL` | All NP banks except `EN_NP_WORK` |
| `EN_NP_COUNTABLE` | Subset of countable NPs for "I have a…" templates |
| `EN_VERBS_ALL` | All 3 verb banks combined |

### What needs to happen

**0.1 Align patterns with benchmark taxonomy.**
The current 7 builders (hindi_matrix_english_np, hindi_matrix_english_verb, etc.) do not map 1:1 to the benchmark's 7 categories (CS-01 through CS-07). Two patterns are missing entirely:

| Benchmark ID | Status | Action |
|---|---|---|
| CS-01 Noun Insertion | Covered by `build_hindi_matrix_english_np` | Map existing |
| CS-02 Verb Grafting | Covered by `build_hindi_matrix_english_verb` | Map existing |
| CS-03 Tag Switching | Covered by `build_tag_switching` | Map existing |
| CS-04 Clause Boundary | Partially covered by `build_inter_sentential` | Review and extend |
| CS-05 Technical/Slang | Partially covered by `build_everyday_conversation` | Needs dedicated builder |
| CS-06 Numerical/Entity | **Not implemented** | New builder needed |
| CS-07 Intraword | **Not implemented** | New builder needed |

CS-06 (Numerical/Entity) requires date, time, and number mixing — e.g. "Meeting Thursday ko 3 PM par hai". CS-07 (Intraword) involves English morphemes fused with Hindi grammar — e.g. "unfriend kar-diya". This is linguistically harder and the vocabulary bank will be smaller.

**0.2 Stratify output.**
Current generation picks builders randomly. Add stratification so the 5,000 sentences are balanced across:
- 7 pattern categories (~700 each)
- 3 CMI buckets: low (0.1–0.3), mid (0.3–0.5), high (0.5–0.7)

**0.3 Generate script variants (Tier 1 inputs).**
Build a `data/script_variants.py` that takes the Romanized CSV and produces:
- **Set A:** Devanagari (via `indic-transliteration`, then human-verified for the golden subset)
- **Set B:** Romanized (already exists — this is the base output)
- **Set C:** Mixed script (Hindi tokens → Devanagari, English tokens → Roman, using the language tags)

### Deliverable
`data/codeswitched/benchmark_v1.csv` with columns: `sentence_id, pattern_id, cmi_bucket, text_roman, text_devanagari, text_mixed, language_tags`

---

## Phase 1: Golden Set (Weeks 2–4)

This is the hardest dependency in the project. Everything downstream depends on it.

**1.1 Select 300 sentences from the 5,000.**
Stratified sample: ~43 per pattern category, balanced across CMI buckets.

**1.2 Human verification of Devanagari.**
The `indic-transliteration` library will produce errors on Hinglish-specific words ("unfriend" → ???). A native speaker must verify and correct every Devanagari sentence in the golden set.

**1.3 Human audio recordings (optional but high-value).**
If a native Hinglish speaker records the 300 golden sentences, this becomes the ground truth for MCD and F0 reference. Without recordings, you can still run TPI and PIER, but MCD and Boundary F0 RMSE become model-vs-model comparisons rather than model-vs-human.

**Decision point:** Do you record golden audio or not? If not, reframe the paper as a comparative benchmark (models vs each other) rather than an absolute quality measurement.

### Deliverable
`data/golden/` directory with verified Devanagari text + optional audio.

---

## Phase 2: Audio Synthesis Pipeline (Weeks 3–5)

**2.1 Model access and API setup.**
Each model has different access requirements:

| Model | Access | Synthesis method |
|---|---|---|
| Qwen3-TTS | Open weights (HuggingFace) | Local inference, needs GPU |
| CosyVoice 2 | Open weights | Local inference |
| XTTS v2 | Open weights (Coqui) | Local inference |
| Fish-Speech 1.5 | Open weights | Local inference |

All require GPU. Estimate ~2 hours per model for 5,000 sentences on a single A100.

**2.2 Build `evaluation/synthesize.py`.**
Takes `benchmark_v1.csv` + model config, synthesizes all three script variants (A/B/C) for each model. Output: WAV files organized as `results/{model}/{script_set}/{sentence_id}.wav`.

**2.3 Normalize output audio.**
Use the existing `preprocess_streaming.py` loudness normalization (-23 LUFS) to ensure fair comparison across models with different output levels.

### Deliverable
`results/` directory with ~60,000 WAV files (5,000 sentences × 3 scripts × 4 models).

---

## Phase 3: Objective Metrics (Weeks 5–7)

**3.1 PIER (Point-of-Interest Error Rate).**
- Run Whisper Large v3 ASR on each synthesized audio
- Align ASR transcript with reference text
- Calculate WER only at switch-point tokens (using language tags to identify boundaries)
- This reuses the `language_tags` column already in the CSV

**3.2 Boundary F0 RMSE.**
- Extract F0 contour (using CREPE or pYIN via librosa — already a dependency)
- Use forced alignment (MFA or similar) to locate switch-point timestamps
- Calculate F0 RMSE in a ±100ms window around each switch boundary
- Compare against intra-language F0 variance as a baseline

**3.3 MCD (Mel-Cepstral Distortion).**
Only meaningful if golden audio exists. If not, compute MCD between Set A and Set B outputs of the same model (measures how much transliteration degrades the model's own output).

**3.4 H-Index (Phonetic Fidelity).**
- Run Whisper on Hindi-tagged tokens
- If Whisper transcribes "Samajh" as an English word → phonetic failure
- Score: fraction of Hindi tokens correctly recognized as Hindi by ASR

**3.5 TPI (Transliteration Penalty Index).**
Requires human MOS scores from Phase 4 for the full version. But an ASR-proxy TPI can be computed now: `(WER_Roman - WER_Devanagari) / WER_Devanagari * 100` as an objective stand-in.

### Deliverable
`evaluation/acoustic_metrics.py`, `evaluation/tpi.py` producing per-model, per-pattern, per-CMI results tables.

---

## Phase 4: Human Evaluation (Weeks 7–10)

**4.1 Platform setup.**
Prolific or Amazon Mechanical Turk. Require native Hinglish speakers (grew up speaking both Hindi and English in India). Screen with a short Hinglish comprehension task.

**4.2 MOS collection.**
- Use the 300 golden sentences (not all 5,000)
- Each listener rates 30–40 samples (mixed across models and script variants)
- Target: 10+ ratings per sample → ~900 samples × 10 ratings = 9,000 judgments
- Estimated cost: $500–1,000 on Prolific at $12/hr

**4.3 Switch-Point Naturalness (SPN).**
Separate task: play a 2-second clip centered on the switch boundary. Listener rates the transition only (1–5 scale). Reduces confounds from overall sentence quality.

**4.4 Transliteration Robustness Score.**
Computed directly from MOS: `MOS_SetB / MOS_SetA` for each model. No additional data collection needed.

### Deliverable
`evaluation/human_eval/` with anonymized ratings, inter-annotator agreement stats, per-model MOS/SPN tables.

---

## Phase 5: Analysis and Write-Up (Weeks 10–12)

**5.1 Per-pattern breakdown.**
Where do models fail? Hypothesis: CS-06 (numerical/entity) and CS-07 (intraword) will be hardest. CS-01 (noun insertion) will be easiest.

**5.2 TPI leaderboard.**
Rank models by script-agnosticism. This is the headline result.

**5.3 CMI-stratified analysis.**
Do models degrade gracefully as code-mixing intensity increases, or is there a cliff?

**5.4 Publish results to README and a preprint.**

---

## Critical Breakpoints

### 1. Transliteration quality poisons TPI

**Risk:** If `indic-transliteration` (Roman → Devanagari) produces bad Devanagari, Set A inputs are degraded. Models sound bad on Set A not because they can't handle Devanagari, but because the Devanagari is wrong. This makes TPI artificially low (model looks more script-agnostic than it is).

**Mitigation:** The golden set (300 sentences) must be human-verified Devanagari. Report TPI on the golden set separately from TPI on the full machine-transliterated set. If the two numbers diverge significantly, the transliteration library is the bottleneck, not the model.

### 2. English tokens in the Devanagari set

**Risk:** Words like "meeting", "download", "email" appear in both Hindi and English contexts. When generating Set A (full Devanagari), should "meeting" become "मीटिंग"? If yes, you're testing the model's Devanagari reading, not its Hinglish capability. If no, Set A and Set C become identical for high-English sentences.

**Mitigation:** Define a clear rule: only Hindi-tagged tokens get transliterated. English-tagged tokens stay in Roman script even in Set A. This means Set A is really "Hindi-in-Devanagari + English-in-Roman" (which is actually how real Devanagari Hinglish is written). Document this explicitly.

### 3. Forced alignment quality at switch boundaries

**Risk:** F0 RMSE depends on knowing exactly where the switch happens in the audio. Forced alignment tools (MFA, Whisper timestamps) may be unreliable at exactly the points you care about — the switch boundaries — because these are where pronunciation is most ambiguous.

**Mitigation:** Use two independent alignment methods and report agreement. For the golden set, manual alignment of switch boundaries (time-consuming but gold-standard). For the full set, accept that alignment noise adds variance and report confidence intervals.

### 4. No golden audio recordings

**Risk:** Without human-recorded reference audio, MCD becomes model-vs-model (meaningless for absolute quality) and F0 RMSE has no ground truth. The benchmark becomes a relative comparison rather than an absolute quality measurement.

**Mitigation:** This is actually okay for a first publication. Frame it as: "We compare models against each other and against their own Devanagari performance." TPI, PIER, SPN, and MOS all work without reference audio. MCD and F0 can be added in a v2 if recordings are obtained later.

### 5. Human evaluation sample size and cost

**Risk:** 300 golden sentences × 3 scripts × 4 models = 3,600 samples. At 10 ratings each = 36,000 judgments. At $12/hr and ~120 judgments/hour, that's ~$3,600. This may exceed a personal project budget.

**Mitigation:** Reduce scope. Options:
- Evaluate 2 models instead of 4 (pick the most interesting pair, e.g. Qwen3-TTS vs XTTS v2)
- Evaluate only Set A vs Set B (drop Set C — the Devanagari vs Roman comparison is the headline anyway)
- Use 150 golden sentences instead of 300
- Use fewer raters (5 instead of 10) and report wider confidence intervals
- A reduced design: 150 sentences × 2 scripts × 2 models × 5 raters = 3,000 judgments ≈ $300

### 6. CS-07 (Intraword) is linguistically hard to generate

**Risk:** Intraword code-switching ("unfriend kar-diya", "recharge karna") involves English morphemes fused with Hindi verb conjugation. The vocabulary is limited and patterns are irregular. Generating 700 diverse intraword sentences may produce repetitive or unnatural output.

**Mitigation:** Accept a smaller set for CS-07 (~200 instead of 700). This pattern is rare in natural Hinglish too, so a smaller sample is defensible. Over-index on quality over quantity for this category.

### 7. Model versioning and reproducibility

**Risk:** E2E models update frequently. Qwen3-TTS today may not be Qwen3-TTS in 3 months. Results become stale.

**Mitigation:** Pin exact model versions (HuggingFace commit hashes). Store synthesis configs. Publish the benchmark set and evaluation code so others can rerun on newer models.

---

## Dependency Graph

```
Phase 0 (Test Set)
    │
    ├──→ Phase 1 (Golden Set) ──→ Phase 4 (Human Eval)
    │                                      │
    └──→ Phase 2 (Synthesis) ──→ Phase 3 (Objective Metrics)
                                           │
                                           └──→ Phase 5 (Analysis)
                                                     ↑
                                           Phase 4 ──┘
```

Phase 0 and Phase 2 can run in parallel once the benchmark CSV exists.
Phase 3 and Phase 4 can run in parallel.
Phase 5 requires both Phase 3 and Phase 4 to complete.

---

## What's Already Built vs What's Needed

| Component | Status | Location |
|---|---|---|
| Sentence generation (5 of 7 patterns) | Done | `data/codeswitching.py` |
| CS-06 Numerical/Entity builder | **Needed** | `data/codeswitching.py` |
| CS-07 Intraword builder | **Needed** | `data/codeswitching.py` |
| Script variant generator | **Needed** | `data/script_variants.py` |
| Stratified sampling | **Needed** | `data/codeswitching.py` |
| G2P pipeline | Done | `normalization/g2p.py` |
| Audio preprocessing | Done | `data/preprocess_streaming.py` |
| Tests for codeswitching | Done | `tests/test_codeswitching.py` |
| Synthesis pipeline | **Needed** | `evaluation/synthesize.py` |
| PIER metric | **Needed** | `evaluation/acoustic_metrics.py` |
| F0 boundary analysis | **Needed** | `evaluation/acoustic_metrics.py` |
| TPI calculation | **Needed** | `evaluation/tpi.py` |
| Human eval setup | **Needed** | `evaluation/human_eval/` |
| Golden set curation | **Needed** | `data/golden/` |
