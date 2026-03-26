# XTTS-v2 Linguistic Quality & Phonetic Fidelity Report
## Phase 1.5 Evaluation Results (Corrected Metrics)

---

## Executive Summary

XTTS-v2 is a **moderate performer** among the three evaluated models. It produces functional Hinglish audio but with significant gaps in Hindi phonetic recognition and code-switch handling:

1. **Phonetic Fidelity (H-Index): 42.25%** — Moderate Hindi recognition; better than CosyVoice 3 (0%) but far below Qwen3-TTS (67.95%).
2. **WER: 1.47–1.89** — High overall word error rate; mixed-script is worst (1.89).
3. **PIER: 0.804** — Poor code-switch boundary handling; worst performance in cs01_noun_insertion and cs06_numerical_entity (both 1.0 error).
4. **Language Boundary Clarity: 0.455** — Moderate; far below XTTS-v2's historical reputation for sharp language boundaries (old metrics showed 1.0 — those were incorrect).

---

## Detailed Findings

### 1. Phonetic Fidelity (H-Index): 42.25%

**What it measures:** Fraction of Hindi tokens in the mixed-script variant correctly recognized by Whisper ASR — a proxy for phonetic accuracy.

| Category | H-Index | Bar |
|----------|---------|-----|
| cs02_verb_grafting | **0.70** ✅ | ██████████████░░░░░░ |
| cs05_technical_slang | **0.80** ✅ | ████████████████░░░░ |
| devanagari_pure | 0.455 ⚠️ | █████████░░░░░░░░░░░ |
| cs04_clause_boundary | 0.667 ⚠️ | █████████████░░░░░░░ |
| roman_pure | 0.545 ⚠️ | ██████████░░░░░░░░░░░ |
| mixed_script | 0.286 ❌ | █████░░░░░░░░░░░░░░░ |
| cs01_noun_insertion | **0.0** ❌ | ░░░░░░░░░░░░░░░░░░░░ |
| cs03_tag_switching | **0.0** ❌ | ░░░░░░░░░░░░░░░░░░░░ |
| cs07_intraword | **0.0** ❌ | ░░░░░░░░░░░░░░░░░░░░ |
| **Overall weighted** | **0.4225** | |

**Interpretation:**
- 42.25% H-Index is moderate — better than CosyVoice 3 (0%) but significantly below Qwen3-TTS (67.95%).
- XTTS-v2 completely fails at noun insertion (cs01), tag-switching (cs03), and intraword code-switching (cs07), all scoring 0% H-Index.
- Strongest categories: cs05_technical_slang (80%) and cs02_verb_grafting (70%).
- Pure Devanagari (45.5%) slightly outperforms pure Roman (54.5%), contradicting the old report's claim of parity.

---

### 2. Word Error Rate & TPI

| Script | WER | Δ vs Devanagari (TPI) |
|--------|-----|----------------------|
| Roman | 1.468 | -4.7% (Roman slightly easier) |
| Devanagari | 1.540 | — |
| Mixed | 1.890 | +22.7% (Mixed much harder) |

**Key findings:**
- All three scripts have **high WER (>1.4)** — worse performance than both CosyVoice 3 (Roman 0.84, Mixed 0.76) and Qwen3-TTS (Roman 0.895, Mixed 0.44).
- Roman performs slightly better than Devanagari (-4.7% TPI).
- Mixed-script is significantly harder (+22.7% TPI), reversing the expected trend where mixed should combine strengths.

---

### 3. Boundary F0 RMSE (Prosody Continuity)

**What it measures:** F0 std-dev across voiced frames as a proxy for prosodic discontinuity. Lower = smoother.

| Script | Mean F0 std-dev | Median | N valid |
|--------|-----------------|--------|---------|
| Roman | 26.17 Hz | 24.58 Hz | 18 / 20 |
| Devanagari | 28.24 Hz | 28.75 Hz | 18 / 20 |
| Mixed | 29.91 Hz | 27.09 Hz | 18 / 20 |

**Interpretation:**
- Roman has best prosody (26.17 Hz), followed by Devanagari (28.24 Hz), then Mixed (29.91 Hz).
- All three are in the 20–30 Hz range (moderate discontinuity) — comparable to Qwen3-TTS (29–33 Hz) and better than CosyVoice 3 in mixed-script context.
- **No file shows silence** — all 18 files per variant produced valid audio (unlike CosyVoice 3 which had 14 silent Devanagari files).

---

### 4. PIER (Code-Switch Boundary Error Rate): 0.804

**What it measures:** Fraction of tokens at language switch-points incorrectly transcribed.

| Category | PIER |
|----------|------|
| cs01_noun_insertion | **1.000** ❌ |
| cs04_clause_boundary | **1.000** ❌ |
| cs06_numerical_entity | **1.000** ❌ |
| cs03_tag_switching | 0.750 ❌ |
| mixed_script | 0.750 ❌ |
| cs02_verb_grafting | 0.571 ⚠️ |
| cs05_technical_slang | 0.500 ⚠️ |

**Interpretation:**
- PIER 0.804 is **significantly worse than Qwen3-TTS (0.500)** — 30 percentage points higher error rate.
- Three patterns achieve 100% error: noun insertion, clause boundaries, and numerical entities. XTTS-v2 completely fails at these boundary types.
- Only cs05_technical_slang has acceptable PIER (0.5), suggesting the model handles technical terms better than structured code-switching patterns.

---

### 5. Language Boundary Confidence (LID): 0.455

**What it measures:** At code-switch boundary positions, whether ASR output uses the expected Unicode script (Devanagari for HI, Latin for EN).

| Metric | Value |
|--------|-------|
| Total boundary tokens | 44 |
| Correctly script-typed | 20 |
| Overall LID | **0.455** |

**Interpretation:**
- Only 45.5% of boundary tokens are in the expected script — moderate, not the "perfect 1.0" claimed in the old report.
- This **contradicts the earlier claim** that XTTS-v2 has "perfect language boundary clarity."
- **Why the old report was wrong:** It used manual annotation, not ASR-based detection. The new method is more rigorous: it checks if Whisper actually perceives the expected language at each boundary.
- LID 0.455 indicates **significant language blending** — Whisper struggles to identify which language is being spoken at code-switch boundaries. This is moderately good for naturalness but worse than both Qwen3-TTS (0.60) and significantly worse than the false "1.0" baseline.

---

## Per-Switching-Pattern Analysis

### Patterns with Best Phonetic Fidelity:

| Pattern | Example Sentence | H-Index |
|---------|-----|---------|
| T04 (Pure Hindi colloquial) | "Yaar main bahut thak gaya hoon" | 83.3% ✅ |
| T02 (Pure Hindi verb-final) | "Mujhe kal subah jaldi uthna padega" | 66.7% ⚠️ |
| T08 (CS-01, Noun insertion) | "Mera code review pending hai" | 60.0% ⚠️ |

### Patterns with Worst Phonetic Fidelity:

| Pattern | Example Sentence | H-Index |
|---------|-----|---------|
| T14 (CS-04, EN clause + Hindi response) | "I finished the report. Ekdum acha laga" | 0.0% ❌ |
| T12 (CS-03, EN sentence + Hindi tag) | "The project got cancelled. Main khush hoon" | 28.6% ❌ |
| T11 (CS-03, Hindi tag + EN clause) | "Yaar the deadline is really urgent" | 33.3% ❌ |

**Pattern:** Sentences with large English segments (CS-03, CS-04) have worse H-Index. This suggests XTTS-v2's Hindi phoneme coverage collapses when there's heavy code-switching.

---

## Prosody Patterns by Code-Switching Type

### Intraword Code-Switching (CS-07) Shows Worst Prosody:

| Category | F0 RMSE Examples |
|----------|---------|
| CS-07 (Intraword) | T19_mixed: **91.06 Hz** ⚠️ |
| CS-02 (Verb Grafting) | T09_mixed: **84.28 Hz** |
| CS-06 (Numerical/Entity) | T17, T18: ~50-70 Hz range |

**T19_mixed (worst case):** "Usne mujhe unfriend kar diya"
- English verb stem "unfriend" + Hindi inflection "kar diya"
- F0 RMSE = 91.06 Hz (extreme pitch discontinuity)
- This is the **hardest code-switching pattern** and XTTS-v2 fails dramatically.

---

## Limitations of Current Metrics

The H-Index computation in this report is **simplified** — it only checks if words appear in a hand-curated dictionary. A more rigorous H-Index would require:

1. **Forced alignment** — Align audio frames to text tokens to identify where each word is pronounced.
2. **Phonetic recognition** — Extract actual phonemes from audio using a phonetic ASR or acoustic model.
3. **G2P prediction** — Predict expected phonemes for each word using script-specific (Roman vs Devanagari) grapheme-to-phoneme rules.
4. **Comparison** — Match extracted vs expected and score phonetic correctness.

The current simplified version only indicates "vocabulary coverage," not true phonetic accuracy. A full H-Index would likely show **even worse** results because even words in vocabulary may be mispronounced.

---

## Recommendations for Phase 2+

### For XTTS-v2:

1. **Phonetic Fidelity is the primary blocker.** The model needs better Hindi phoneme coverage or a multilingual phoneme space that handles code-switching.
   - **Action:** Fine-tune XTTS-v2 on Hindi-English code-switched speech data with explicit phoneme-level annotations.

2. **Prosody must be improved at boundaries.** F0 RMSE of 30-41 Hz indicates unnatural switching.
   - **Action:** Add prosody modeling for code-switch boundaries (e.g., continuity loss during training).

3. **Mixed-script has worst prosody.** If supporting all three script variants, prioritize Roman/Devanagari and deprioritize Mixed for now.

### For Benchmark Evaluation:

1. **Report both acoustic AND linguistic metrics.** The acoustic metrics (RMS, spectral) showed all three scripts as "good," but linguistic metrics reveal serious problems.

2. **Use per-pattern analysis.** Separate results by code-switching pattern (CS-01 through CS-07) to identify which patterns are hardest.

3. **Compute full H-Index with forced alignment** (requires phonetic ASR like Phoneme Recognition or wav2vec + forced aligner).

4. **Compare against baseline:** XTTS-v2 should be compared against:
   - Qwen3-TTS (expected to be better due to Qwen tokenizer coverage)
   - CosyVoice 3 (expected to be similar to Qwen3-TTS, except for Devanagari)
   - Fish Audio S2 Pro (not yet tested)

---

## Conclusion

**XTTS-v2's perceived audio quality (high RMS, clean spectrum) masks severe linguistic problems:**

- **36.3% H-Index** → Weak Hindi phonetic grounding
- **30-41 Hz F0 RMSE** → Unnatural prosody at code-switches
- **No script advantage for Devanagari** → Contradicts user observation

The model is suitable for **non-linguistic evaluation** (naturalness ratings, speaker similarity) but problematic for **linguistic evaluation** (phonetic fidelity, prosodic naturalness at boundaries).

This benchmark will reveal whether other models (Qwen3-TTS, CosyVoice 3, Fish Audio S2) handle these linguistic challenges better.

---

---

# CosyVoice 3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512) Linguistic Quality Report
## Phase 1.5 Evaluation Results

---

## Executive Summary

CosyVoice 3 produces functional audio for Roman and Mixed-script Hinglish but suffers from **fundamental Devanagari failure** and **complete Hindi phonetic non-recognition**:

1. **Devanagari: 100% WER, 14/20 files silent** — CosyVoice 3's Qwen LLM was fine-tuned on Mandarin+English only; Devanagari chars map to rare/unknown tokens → near-empty speech token stream → near-silent audio.
2. **H-Index: 0.0%** — Zero Hindi tokens correctly transcribed in the mixed-script variant. ASR output is too garbled or language-confused to match any Hindi reference tokens.
3. **LID: 0.3036** — Only 30% of language-boundary tokens use the expected script. CosyVoice 3 does not produce acoustically clear Hindi vs English distinctions (contrast: XTTS-v2 achieves 1.0).
4. **Prosody: 24–29 Hz F0** (Roman/Mixed) — Moderate, slightly better than XTTS-v2, but Devanagari data is too sparse (only 5/20 files voiced) to be meaningful.

---

## Metric Summary

| Metric | Roman | Devanagari | Mixed |
|--------|-------|-----------|-------|
| WER | 0.842 | **1.000** | 0.763 |
| TPI vs Devanagari | -15.8% | — | -23.7% |
| PIER (mixed) | — | — | **0.679** |
| H-Index | — | — | **0.0%** |
| F0 std-dev (Hz) | 29.12 (N=20) | 40.29 (N=5) | 24.25 (N=17) |
| LID confidence | — | — | **0.304** |

> TPI negative values mean Roman/Mixed have **lower** WER than Devanagari (Devanagari is the worst performer).

---

## Detailed Findings

### 1. Phonetic Fidelity (H-Index): 0.0%

**What it measures:** Fraction of Hindi tokens in the mixed-script variant that are correctly transcribed by Whisper ASR — a proxy for how well the model produces recognisable Hindi phonemes.

| Category | H-Index |
|----------|---------|
| All categories | 0.0% ❌ |

**Interpretation:**
- Not a single Hindi token across all 20 test sentences was correctly transcribed.
- This is **worse than XTTS-v2 (36.3%)**, despite CosyVoice 3 producing intelligible Roman/Mixed audio.
- The likely cause: CosyVoice 3 uses a Qwen-based text encoder fine-tuned on Mandarin+English. Hindi phoneme mappings are unreliable — the model may be producing acoustically plausible but phonetically incorrect realisations that Whisper cannot match to Hindi reference tokens.
- **Expected behavior (good fidelity):** > 70% H-Index.

---

### 2. Word Error Rate & TPI

| Script | WER | Δ vs Devanagari (TPI) |
|--------|-----|----------------------|
| Roman | 0.842 | -15.8% (Roman easier) |
| Devanagari | **1.000** | — |
| Mixed | 0.763 | -23.7% (Mixed easier) |

**Key findings:**
- **Devanagari WER = 1.0**: Every Devanagari sentence is wrong. 14/20 files are near-silent; the remaining 6 produce garbled output in an unrelated language (Telugu, Korean, or silence bursts observed in transcripts).
- **Roman WER = 0.842**: Quite high. Some sentences transcribe partially (T08 roman: 0.40 WER; T11/T12 roman: 0.17/0.43 WER). Hindi-dominant sentences fail almost completely.
- **Mixed WER = 0.763**: Slightly better than Roman. The English portions transcribe correctly; the Hindi portions drag WER up.

**Best-performing sentences (lowest WER, mixed):**
| Test ID | Sentence | WER |
|---------|---------|-----|
| T08 | Mera code review pending hai | 0.40 |
| T11 | Yaar the deadline is really urgent | 0.17 |
| T18 | Google Meet 5 PM par milte hain | 0.50 |

**Root cause for Devanagari failure:**
CosyVoice 3's Qwen LLM backbone was fine-tuned exclusively on Mandarin and English speech data. When given Devanagari input, the tokenizer produces valid but extremely rare token sequences that the speech synthesis model has never seen in training — resulting in near-zero speech token output and silent audio. This is a **fine-tuning data gap**, not a vocabulary gap (the underlying Qwen BPE tokenizer does have Devanagari entries).

---

### 3. PIER (Code-Switch Boundary Error Rate): 0.679

**What it measures:** Fraction of tokens at language switch-points that are incorrectly transcribed.

| Category | PIER |
|----------|------|
| cs07_intraword | **1.000** ❌ |
| mixed_script | 0.875 ❌ |
| cs02_verb_grafting | 0.714 ❌ |
| cs05_technical_slang | 0.700 ❌ |
| cs01_noun_insertion | 0.571 ⚠️ |
| cs03_tag_switching | 0.500 ⚠️ |
| cs04_clause_boundary | 0.500 ⚠️ |
| cs06_numerical_entity | 0.500 ⚠️ |
| **Overall** | **0.679** ❌ |

**Interpretation:**
- ~68% of all tokens at code-switch boundaries are transcribed incorrectly — the model fails precisely at the points where language identity matters most.
- **Intraword code-switching (cs07) is 100% error** — fusing Hindi morphology with English stems (e.g., "unfriend kar diya") is completely beyond CosyVoice 3's capability.
- Clause-boundary and tag-switching patterns (cs03, cs04) have the lowest PIER at 0.5 — the model handles fully-English segments reasonably well even when surrounded by Hindi context.

---

### 4. Language Boundary Confidence (LID): 0.304

**What it measures:** At code-switch boundary positions in the mixed-script variant, whether ASR output uses the expected Unicode script (Devanagari for HI tokens, Latin for EN tokens).

| Metric | Value |
|--------|-------|
| Total boundary tokens | 56 |
| Correctly script-typed | 17 |
| Overall LID | **0.304** |

**Interpretation:**
- Only 30% of boundary tokens are in the expected script — far below XTTS-v2 (1.0).
- This means CosyVoice 3 **does not produce acoustically distinct Hindi vs English signals** at boundaries. The language blur is measurable: Whisper frequently romanises what should be Devanagari output, or picks up the wrong language entirely.
- **Counterintuitively, low LID can indicate better naturalness** — the model is blending languages acoustically rather than hard-switching. But given the concurrent 0% H-Index, in CosyVoice 3's case the low LID reflects linguistic confusion rather than smooth blending.

**Per-sentence highlights:**
| Test ID | Category | LID |
|---------|----------|-----|
| T11 | cs03_tag_switching | 0.50 |
| T08 | cs01_noun_insertion | 0.50 |
| T05 | mixed_script | 0.00 |
| T09 | cs02_verb_grafting | 0.00 |
| T19, T20 | cs07_intraword | 0.00 |

---

### 5. Boundary F0 RMSE (Prosody): 24–40 Hz

| Script | Mean F0 std-dev | Median | N valid |
|--------|----------------|--------|---------|
| Roman | 29.12 Hz | 27.34 Hz | 20 / 20 |
| Devanagari | 40.29 Hz | 35.50 Hz | **5 / 20** |
| Mixed | **24.25 Hz** | 24.26 Hz | 17 / 20 |

> ⚠️ Devanagari F0 is from only 5 voiced files — not representative. The 14 silent files are excluded from computation.

**Interpretation:**
- Roman (29.12 Hz) is comparable to XTTS-v2 (30.14 Hz) — both exhibit moderate prosodic discontinuity.
- **Mixed (24.25 Hz) is the best result across both models** — 17 Hz better than XTTS-v2's mixed (41.17 Hz). CosyVoice 3 handles prosodic blending in mixed utterances somewhat better.
- Devanagari data is too sparse to draw conclusions.

**Worst-case files:**
| File | F0 std-dev | Note |
|------|-----------|------|
| T07_mixed | 55.73 Hz | cs01 noun insertion — large pitch spikes |
| T14_devanagari | 59.06 Hz | one of 5 voiced Devanagari files |
| T20_devanagari | 49.94 Hz | high variation in Devanagari |
| T05_roman | 40.64 Hz | mixed-script test in roman transcription |
| T14_roman | 43.03 Hz | clause boundary pattern |

---

## Limitations of XTTS-v2 Report

**Note:** This report was corrected from an earlier version that used inferior methodology:
- Old H-Index: 36.3% (hand-coded 21-word phoneme dictionary)
- New H-Index: 42.25% (ASR-based, ASR transcription matching)
- Old LID: 1.000 (manual annotation of script expected)
- New LID: 0.455 (ASR perception of actual script used)

The new metrics are more rigorous and comparable to other models.

---

---

# Qwen3-TTS Linguistic Quality Report
## Phase 1.5 Evaluation Results

---

## Executive Summary

Qwen3-TTS is the strongest performer evaluated so far for Hinglish TTS. It is the **only model with functional Devanagari support** and achieves near-target Hindi phonetic fidelity:

1. **Devanagari WER: 0.473** — All 20 Devanagari files produce valid audio. This is the only model where Devanagari is viable.
2. **H-Index: 67.95%** — Close to the >70% target; vastly better than XTTS-v2 (36.3%) and CosyVoice 3 (0%).
3. **Roman WER: 0.895** — Counterintuitively, Roman script is *harder* for this model than Devanagari (+89.3% TPI). The model was optimized for Devanagari Hinglish.
4. **LID: 0.600** — Moderate language boundary clarity. Better than CosyVoice 3 (0.304), reflecting genuine Hindi phoneme output but incomplete boundary sharpness.
5. **PIER: 0.500** — 50% error rate at code-switch boundaries; better than CosyVoice 3 (67.9%) but still significant.

---

## Metric Summary

| Metric | Roman | Devanagari | Mixed |
|--------|-------|-----------|-------|
| WER | 0.895 | **0.473** | 0.440 |
| TPI vs Devanagari | **+89.3%** | — | -7.0% |
| PIER (mixed) | — | — | **0.500** |
| H-Index | — | — | **67.95%** |
| F0 std-dev (Hz) | 29.18 (N=20) | 33.00 (N=20) | 33.00 (N=20) |
| LID confidence | — | — | **0.600** |

> TPI +89.3% means Roman WER is 89% *higher* than Devanagari — the model strongly prefers Devanagari input.

---

## Detailed Findings

### 1. Phonetic Fidelity (H-Index): 67.95%

**What it measures:** Fraction of Hindi tokens in the mixed-script variant correctly transcribed by Whisper ASR.

| Category | H-Index | Bar |
|----------|---------|-----|
| cs06_numerical_entity | **1.000** ✅ | ████████████████████ |
| cs09_verb_grafting | 0.800 ✅ | ████████████████░░░░ |
| cs05_technical_slang | 0.833 ✅ | █████████████████░░░ |
| mixed_script | 0.857 ✅ | █████████████████░░░ |
| roman_pure | 0.818 ✅ | ████████████████░░░░ |
| cs01_noun_insertion | 0.714 ⚠️ | ██████████████░░░░░░ |
| cs07_intraword | 0.750 ⚠️ | ███████████████░░░░░ |
| devanagari_pure | 0.455 ⚠️ | █████████░░░░░░░░░░░ |
| cs04_clause_boundary | 0.286 ❌ | █████░░░░░░░░░░░░░░░ |
| cs03_tag_switching | **0.000** ❌ | ░░░░░░░░░░░░░░░░░░░░ |
| **Overall weighted** | **0.6795** | |

**Key observations:**
- Numerical entities (T17, T18 — "Meeting Monday 3 PM", "Google Meet 5 PM par milte hain") achieve 100% H-Index. Qwen3-TTS handles mixed numeric/Hindi utterances cleanly.
- cs03_tag_switching (0.0%): T11 "Yaar the deadline is really urgent" and T12 "The project got cancelled. Main khush hoon" both score 0 — the Hindi tag/response at end of English sentence is not produced with recognizable phonemes.
- Devanagari-pure category (0.455): The model handles Devanagari better than Roman, but pure Hindi sentences in Devanagari still have moderate fidelity gaps.

---

### 2. Word Error Rate & TPI

| Script | WER | Δ vs Devanagari (TPI) |
|--------|-----|----------------------|
| Roman | 0.895 | **+89.3%** (Roman much harder) |
| Devanagari | **0.473** | — |
| Mixed | 0.440 | -7.0% (Mixed slightly easier) |

**Key findings:**
- **Devanagari WER = 0.473**: Lowest of all three scripts — and the only model where Devanagari produces meaningful audio. 8/20 sentences transcribe nearly perfectly (WER ≤ 0.17): T01, T03, T05, T07, T09, T15.
- **Roman WER = 0.895**: The model struggles heavily with Romanized Hindi input. Many sentences produce mostly-English output, ignoring the Hindi tokens entirely.
- **Mixed WER = 0.440**: Best overall. The combination of Hindi and English in one sentence plays to the model's strengths.
- **TPI +89.3%**: Devanagari is the preferred input script for this model — an inverse of the expected Roman-script TTS behavior.

**Best-performing sentences (Devanagari, WER = 0):**

| Test ID | Sentence | WER Devanagari |
|---------|---------|---------------|
| T01 | आज बहुत काम करना है | 0.000 |
| T03 | बहुत काम करना है | 0.000 |
| T05 | Hope the team meeting mein jaana... | 0.000 |
| T09 | Hale se karo phir baat karte hain | 0.143 |

---

### 3. PIER (Code-Switch Boundary Error Rate): 0.500

| Category | PIER |
|----------|------|
| cs05_technical_slang | **1.000** ❌ (T16 only — all 6 boundary tokens wrong) |
| cs02_verb_grafting | 0.571 ❌ |
| cs03_tag_switching | 0.500 ⚠️ |
| cs04_clause_boundary | 0.500 ⚠️ |
| mixed_script | 0.500 ⚠️ |
| cs01_noun_insertion | 0.429 ⚠️ |
| cs07_intraword | **0.333** ✅ |
| cs06_numerical_entity | **0.300** ✅ |
| **Overall** | **0.500** |

**Key findings:**
- PIER is substantially better than CosyVoice 3 (0.679) — 17.9 percentage points lower.
- cs06 (numerical entity) and cs07 (intraword) have lowest PIER — the model handles number-entity boundaries and morphological code-switching better than sentence-level switches.
- T16 (cs05 technical slang: "Laptop slow ho gayi hai, kya karo yaar") scores PIER = 1.0 — every boundary token is wrong. This sentence has 6 switch points and the model fails at all of them.

---

### 4. Language Boundary Confidence (LID): 0.600

| Metric | Value |
|--------|-------|
| Total boundary tokens | 50 |
| Correctly script-typed | 30 |
| Overall LID | **0.600** |

> Note: T16 has 0 boundary tokens (T16 transcript was skipped in LID — the mixed output was empty/unrecognizable), so total is 50 not 56.

**Interpretation:**
- 60% of boundary tokens are in the expected script — a meaningful improvement over CosyVoice 3 (30.4%), but well below XTTS-v2 (100%).
- The model produces genuine Devanagari ASR output (not romanized) for many Hindi segments, unlike CosyVoice 3.
- Perfect boundary score (LID = 1.0) achieved on T14 (cs04 clause boundary) and T18 (cs06 numerical entity).
- Weakest: T06 mixed_script (0.40), T17 numerical (0.40) — the model blends languages at these switch points.

**LID vs XTTS-v2 interpretation:**
XTTS-v2 scores 1.0 (hard switches) while Qwen3-TTS scores 0.6 (partial blending). The lower Qwen3-TTS LID may actually indicate more **natural** code-switching, where the acoustic boundary is less abrupt — consistent with its better mixed F0 in some files.

---

### 5. Boundary F0 RMSE (Prosody): 29–33 Hz

| Script | Mean F0 std-dev | Median | N valid |
|--------|----------------|--------|---------|
| Roman | 29.18 Hz | 30.05 Hz | 20 / 20 |
| Devanagari | 33.00 Hz | 32.55 Hz | **20 / 20** |
| Mixed | 33.00 Hz | 28.24 Hz | **20 / 20** |

**Key observation: All 20 files are voiced for every variant** — no silent outputs. This is the first model with complete audio coverage.

**Interpretation:**
- Roman prosody (29.18 Hz) is comparable across all three models.
- Devanagari and Mixed (33 Hz mean) show moderate discontinuity — in the same range as XTTS-v2's Roman (30 Hz).
- The mixed **median** (28.24 Hz) is close to CosyVoice 3's mixed (24.26 Hz), suggesting the majority of sentences have acceptable prosody, with a few high-F0 outliers pulling the mean up.

**Notable outlier:**
| File | F0 std-dev | Note |
|------|-----------|------|
| T02_mixed | **140.27 Hz** | Extreme outlier — "Mujhe kal subah jaldi uthna padega" in mixed script triggered severe pitch discontinuity |
| T05_roman | 47.12 Hz | mixed-script sentence in Roman |
| T14_devanagari | 44.52 Hz | clause-boundary pattern |

T02_mixed (140.27 Hz) is the highest F0 outlier across all models. This single sentence inflates the mixed mean — without it, mixed mean would be ~27 Hz.

---

## Cross-Model Comparison: All Three Models (Corrected)

| Metric | XTTS-v2 | CosyVoice 3 | **Qwen3-TTS** | Best |
|--------|---------|------------|--------------|------|
| WER Roman | 1.468 | 0.842 | 0.895 | CosyVoice 3 |
| WER Devanagari | 1.540 | **1.000** | **0.473** | Qwen3-TTS |
| WER Mixed | 1.890 | 0.763 | **0.440** | Qwen3-TTS |
| H-Index (mixed) | **42.25%** | 0.0% | **67.95%** | Qwen3-TTS |
| PIER (mixed) | **0.804** | 0.679 | **0.500** | Qwen3-TTS |
| F0 Roman (Hz) | 26.17 | 29.12 | 29.18 | XTTS-v2 |
| F0 Devanagari (Hz) | 28.24 | 40.29 (N=5) | 33.00 (N=20) | XTTS-v2 |
| F0 Mixed (Hz) | 29.91 | **24.25** | 33.00 | CosyVoice 3 |
| LID (mixed) | 0.455 | 0.304 | **0.600** | Qwen3-TTS |
| Valid Devanagari files | 18 / 18 | 6 / 20 | **20 / 20** | Qwen3-TTS |

**Summary:**
- **Qwen3-TTS wins decisively:** H-Index (67.95%), PIER (0.500), WER on Devanagari (0.473) and Mixed (0.440), LID (0.600), Devanagari viability.
- **XTTS-v2 is competitive on prosody:** Best F0 Roman (26.17) and Devanagari (28.24), moderate H-Index (42.25%).
- **CosyVoice 3 is weakest overall:** H-Index 0%, WER 1.0+ on Devanagari, only advantage is mixed-script F0 (24.25).
- **All models have high WER:** 0.44–1.89 range suggests ASR transcription is challenging for Hinglish audio, not a model-specific problem.

**Ranking:** Qwen3-TTS >> XTTS-v2 > CosyVoice 3

---

## Conclusion

**Qwen3-TTS is the best current option for Hinglish TTS, but still has significant gaps:**

- **Strengths:** Only model with working Devanagari (WER 0.473), near-target H-Index (67.95%), lowest code-switch boundary error (PIER 0.5).
- **Weaknesses:** Roman script surprisingly poor (WER 0.895); cs03/tag-switching H-Index collapses to 0%; one extreme F0 outlier (T02_mixed: 140 Hz).
- **Optimal input:** Devanagari or Mixed-script Hinglish. Roman input should be avoided or pre-converted.

For production use, Qwen3-TTS with Devanagari or Mixed input is the recommended choice. Fish Audio S2 Pro is a competitive alternative pending investigation of silent failure root cause.

---

---

# Fish Audio S2 Pro Linguistic Quality Report
## Phase 1.5 Evaluation Results

---

## Executive Summary

**Fish Audio S2 Pro emerges as a competitive second choice**, with highest H-Index (71.43%) and best prosody (25.27 Hz F0) among all four models, but with significant reliability concerns due to silent failures on ~30% of test cases:

1. **H-Index: 71.43%** ✅ — Highest H-Index of all four models (better than Qwen3-TTS's 67.95%).
2. **WER Devanagari: 0.645** — Similar to Qwen3-TTS (0.473) but with more failures.
3. **TPI +43.69%** — Strongest Devanagari preference; Roman script penalty is severe.
4. **Silent failures: ~30% of test files** — Produces no/minimal audio (7/20 files per variant), limiting reliability.
5. **F0 RMSE: 25.27 Hz (mixed)** — **Best prosody among all models**.
6. **LID: 0.585** — Moderate; better than CosyVoice 3 (0.304), close to Qwen3-TTS (0.600).

---

## Metric Summary

| Metric | Roman | Devanagari | Mixed |
|--------|-------|-----------|-------|
| WER | 0.926 | **0.645** | **0.605** |
| TPI vs Devanagari | **+43.69%** | — | -6.16% |
| PIER (mixed) | — | — | **0.643** |
| H-Index | — | — | **71.43%** ✅ |
| F0 std-dev (Hz) | 28.60 (N=13) | 24.62 (N=13) | **25.27 (N=13)** |
| LID confidence | — | — | **0.585** |
| Silent files | Many | Many | Some |

---

## Detailed Findings

### 1. Phonetic Fidelity (H-Index): 71.43%

| Category | H-Index |
|----------|---------|
| **cs05_technical_slang** | **100.0%** ✅ |
| **cs07_intraword** | **100.0%** ✅ |
| **mixed_script** | **100.0%** ✅ |
| cs02_verb_grafting | 83.3% ✅ |
| cs04_clause_boundary | 66.7% ⚠️ |
| roman_pure | 66.7% ⚠️ |
| cs01_noun_insertion | 57.1% ⚠️ |
| cs06_numerical_entity | 57.1% ⚠️ |
| cs03_tag_switching | **0.0%** ❌ |
| **Overall weighted** | **71.43%** ✅ |

**Interpretation:**
- **Perfect on three categories:** cs05_technical_slang, cs07_intraword, and mixed_script all achieve 100% H-Index — pattern-level mastery.
- **Weakness on tag-switching:** H-Index 0%, same as other models — Hindi tags after English clauses are not phonetically reconstructed.
- **Higher average than Qwen3-TTS:** 71.43% vs 67.95%; fewer complete category failures.

---

### 2. Word Error Rate & TPI

| Script | WER | Δ vs Devanagari (TPI) |
|--------|-----|----------------------|
| Roman | 0.926 | **+43.69%** |
| Devanagari | **0.645** | — |
| Mixed | **0.605** | -6.16% |

**Key findings:**
- **Roman WER = 0.926:** Worst among all four models. Model strongly rejects Roman script.
- **Devanagari WER = 0.645:** Reasonable. Silent failures inflat the number; if counted as 0 (skipped), real WER would be lower.
- **Mixed WER = 0.605:** Excellent — model exploits mixed input (explicit language markers) to minimize errors.
- **TPI +43.69%:** Highest Devanagari preference; Roman is nearly unusable.

**Silent files impact:** T01, T02, T03, T06, T10, T12, T13, T15 all have WER = 1.0 due to no audio produced.

---

### 3. PIER (Code-Switch Boundary Error Rate): 0.643

| Category | PIER |
|----------|------|
| cs02_verb_grafting | 0.857 ❌ |
| mixed_script | 0.750 ❌ |
| cs06_numerical_entity | 0.700 ⚠️ |
| cs05_technical_slang | 0.600 ⚠️ |
| cs01_noun_insertion | 0.571 ⚠️ |
| cs04_clause_boundary | 0.500 ⚠️ |
| cs03_tag_switching | 0.500 ⚠️ |
| cs07_intraword | **0.333** ✅ |
| **Overall** | **0.643** |

**Interpretation:**
- PIER 0.643 is **worse than Qwen3-TTS (0.500)** — model struggles at boundaries despite good overall phonetics.
- **Strong on intraword (0.333):** English morphemes with Hindi inflection are handled well.
- **Weak on verb-grafting (0.857):** English verb + Hindi inflection transition is where the model fails most.

---

### 4. Language Boundary Confidence (LID): 0.585

| Metric | Value |
|--------|-------|
| Total boundary tokens | 41 |
| Correctly script-typed | 24 |
| Overall LID | **0.585** |

**Interpretation:**
- 58.5% of boundary tokens use the expected script — moderate but not sharp.
- **Pattern:** English words often get Devanagari transcription ("meeting" → "मीटिंग") despite being expected in Roman.
- Slightly below Qwen3-TTS (0.600), well above CosyVoice 3 (0.304).

---

### 5. Boundary F0 RMSE (Prosody): 25.27 Hz

| Script | Mean F0 (Hz) | Median | N valid / 20 |
|--------|--------------|--------|--------------|
| Roman | 28.60 | 26.15 | 13 |
| Devanagari | 24.62 | 27.36 | 13 |
| Mixed | **25.27** | 25.59 | 13 |

**Interpretation:**
- **Best F0 among all four models:** 25.27 Hz is the lowest/smoothest prosodic discontinuity.
- Only 13/20 files produce sufficient voiced audio per variant due to silent failures.
- **Implicit message:** When the model does produce audio, it has excellent prosody. The problem is reliability/coverage.

---

## Cross-Model Ranking (All Four Models)

| Rank | Model | H-Index | WER Mixed | PIER | F0 Mixed | LID | Reliability |
|------|-------|---------|-----------|------|----------|-----|-------------|
| 1 | **Qwen3-TTS** | 67.95% | **0.440** | **0.500** | 33.00 | **0.600** | ✅ All files voiced |
| 2 | **Fish Audio S2** | **71.43%** ✅ | 0.605 | 0.643 | **25.27** ✅ | 0.585 | ⚠️ 30% silent |
| 3 | **XTTS-v2** | 42.25% | 1.890 | 0.804 | 29.91 | 0.455 | ✅ All files voiced |
| 4 | **CosyVoice 3** | 0.0% ❌ | 0.763 | 0.679 | 24.25 | 0.304 | ⚠️ Devanagari broken |

---

## Critical Issues

1. **Silent failures:** 7/20 files per variant (T01, T02, T03, T06, T10, T12, T13, T15) produce empty/near-silent audio. Cause unknown — could be:
   - Adapter implementation bug (wrong parameters, timeout)
   - Model limitation with certain sentence patterns
   - Transcription artifact (Whisper treats silence as empty)

2. **Reliability vs performance trade-off:** H-Index 71.43% looks good, but ~30% of outputs are unusable. Real-world reliability is ~70%, not 100%.

3. **Roman script unusable:** WER 0.926 with +43.69% TPI means Roman input should not be used in production.

---

---

# Subjective Evaluation: Listening Assessment (2026-03-25)

To validate whether objective metrics reflect naturalness, we conducted informal listening on two representative code-switching patterns using the 5-point scale (naturalness, prosodic blending, phonetic accuracy).

## Test Case 1: T07 — Noun Insertion ("Aaj ka meeting bahut lamba tha")

| Model | Naturalness | Prosodic Blending | Phonetic Accuracy | Subjective Feedback |
|---|---|---|---|---|
| **Fish Audio S2** | 4.0 | 3.5 | 4.0 | Best Hindi quality, but "meeting" pronounced as "making" (/t/ → /k/ substitution) |
| **Qwen3-TTS** | 3.0 | 3.0 | 3.5 | Second choice; accented Hindi but English word correctly pronounced |
| **CosyVoice3** | 2.5 | 2.0 | 2.5 | Sounds like South Indian speaker; unnatural pace and prosody |
| **XTTS-v2** | — | — | — | Audio much longer (7s vs 1-2s); temporal anomaly made direct scoring impractical |

## Test Case 2: T09 — Verb Grafting ("Pehle send karo phir baat karte hain")

| Model | Naturalness | Prosodic Blending | Phonetic Accuracy | Subjective Feedback |
|---|---|---|---|---|
| **Fish Audio S2** | 3.0 | 2.0 | 2.0 | Phonetic accuracy degrades on verb stems |
| **Qwen3-TTS** | 2.5 | 2.0 | 2.0 | Consistent with noun insertion pattern |
| **CosyVoice3** | 1.0 | 1.0 | 1.5 | Barely intelligible |

## Critical Misalignments: Objective vs Subjective

### 1. Fish Audio's Phoneme Substitution (H-Index Blind Spot)

**Objective:** H-Index 71.43% (highest)
**Subjective:** "meeting" → "making" (/t/ → /k/ substitution)

**Why the disconnect?** H-Index measures if Whisper ASR recognizes Hindi tokens as Hindi. It doesn't measure **English phoneme accuracy**. The substituted "making" is still recognized as an English word by ASR, so H-Index passes. But for code-switching naturalness, English pronunciation errors are critical.

**Implication:** A model can score highest on H-Index while having unusable English phoneme errors.

### 2. XTTS-v2 Temporal Anomaly (All Metrics Blind Spot)

**Objective:** 42.25% H-Index, 0.804 PIER, valid acoustic metrics
**Subjective:** Speech rate 5–7× slower than expected (7 seconds for a 1-second sentence)

**Why the disconnect?** None of the 5 metrics measure speech rate. XTTS takes 5–7 seconds to synthesize what Qwen3-TTS does in 1–2 seconds. This makes it unusable despite having acceptable accuracy metrics.

**Implication:** A model with good accuracy can be practically unusable due to temporal anomalies invisible to the metrics.

### 3. Prosodic Blending vs F0 RMSE

**Objective F0 RMSE ranking:** CosyVoice3 (24.25 Hz) > Fish Audio (25.27 Hz) > Qwen3-TTS (33.00 Hz)
**Subjective blending ranking:** Fish Audio > Qwen3-TTS >> CosyVoice3

**Why the disconnect?** F0 RMSE measures pitch variance (pitch continuity). CosyVoice3 has the lowest variance, suggesting the best pitch stability. But the listener perceives it as "South Indian accent with unnatural pace" — issues related to formant quality, voice quality consistency, and speech rate, not pitch.

**Implication:** Excellent F0 continuity can coexist with poor prosodic blending when other factors (accent, voice quality, pace) aren't controlled.

## Evaluation Framework Gaps

The 5-metric pipeline (TPI, PIER, H-Index, F0, LID) measures **speech quality proxies**, not **code-switching naturalness**:

| Dimension | Current Metric | Gap | Why It Matters |
|---|---|---|---|
| English phoneme accuracy | None (H-Index only) | English phoneme errors slip through | "meeting"→"making" breaks intelligibility |
| Speech rate coherence | None | Models can be 5–7× slower than expected | XTTS unusable despite decent metrics |
| Voice quality consistency | None | Formant instability, accent shifts invisible | CosyVoice3 sounds non-native despite low F0 RMSE |
| Naturalness overall | None | All metrics are indirect proxies | No direct measure of "native Hinglish speaker quality" |

## Recommendations for Phase 4+

### Immediate (add to Phase 3):
1. **EN-Phoneme Accuracy Metric** — Apply H-Index logic to English tokens (phoneme-level ASR comparison)
2. **Speech Rate Consistency** — Ratio of actual_duration / expected_duration (target: 0.95–1.05)

### Medium-term (Phase 4):
3. **Formal MOS Study** — 10–15 native Hinglish speakers rating naturalness on 15–20 sentences per model
4. **Phoneme Error Analysis** — Per-token phoneme accuracy for both Hindi and English (finer than H-Index)

### Outcome:
These additions will shift evaluation from "technical sound quality" to **"code-switching usability"** — answering the real question: *"Would a native Hinglish speaker naturally use this for daily communication?"*

---

## Conclusion

**Fish Audio S2 Pro is promising but unreliable for production:**

- **When it works:** Superior H-Index (71.43%), best-in-class prosody (25.27 Hz F0).
- **When it fails:** ~30% of test cases produce no audio; cause requires debugging.
- **Critical caveat:** Subjective evaluation reveals phoneme substitution errors ("meeting" → "making") that objective metrics miss.
- **Safe alternative:** Qwen3-TTS is still the recommended choice due to 100% reliability, consistent pronunciation, and no temporal anomalies.

**Broader insight:** The benchmark's 5-metric framework successfully identifies phonetic/linguistic differences but misses practical naturalness factors (temporal coherence, voice quality, accent stability). Phase 4+ work should add these dimensions for a complete evaluation.
- **Path forward:** Investigate silent failures; if resolved, Fish Audio S2 Pro could become a strong co-leader with Qwen3-TTS.
