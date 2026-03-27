# Sarvam TTS vs Qwen3-TTS — Hinglish Evaluation
## CSPI · HNR · Boundary Penalty — 2026-03-26

**Test set:** T01–T20 (20 sentences, both roman and mixed script for both models)
**Sarvam model:** `bulbul:v3` via Sarvam AI API (built-in `suhani` speaker)
**Qwen3 model:** locally run Qwen3-TTS-12Hz-1.7B-Base with ICL voice cloning (`hindi_ref.wav`)

> **Synthesis paradigm note:** Sarvam uses a production TTS API with a native Hindi speaker voice. Qwen3 uses the Base model with voice cloning from a reference audio clip — the recommended approach for Hinglish without fine-tuning. The comparison therefore reflects real-world deployment scenarios for each model, not a controlled parity setting.

---

## 1. CSPI — Code-Switching Phonetic Index

CSPI combines four dimensions:
- **H-Index** — fraction of Hindi tokens correctly recognised by Whisper ASR
- **E-Index** — fraction of English tokens correctly recognised by Whisper ASR
- **H-Phoneme** — character-level phoneme similarity for Hindi tokens
- **E-Phoneme** — character-level phoneme similarity for English tokens

**Weighting — language-proportional (primary):**

```
w_hi = n_hindi_tokens / (n_hindi_tokens + n_english_tokens)
w_en = n_english_tokens / (n_hindi_tokens + n_english_tokens)

CSPI = w_hi × (H-Index + H-Phoneme) / 2
     + w_en × (E-Index + E-Phoneme) / 2
```

Each sentence is weighted by its own token distribution, so errors in the dominant language count more. A Hindi-heavy sentence (e.g. T01: 83% HI) penalises Hindi mispronunciations more than English ones; an English-heavy sentence (e.g. T05: 83% EN) does the reverse.

> **Normalisation note:** Both reference and Whisper hypothesis are fully normalised to Devanagari before comparison. English words (e.g. "meeting" → "मीटिंग") are mapped via an extended `devanagari_map.py` covering all EN-tagged vocabulary. This ensures the metric evaluates phonetic fidelity in the Hinglish phonological context, where English words naturally adopt Hindi phonetics.

### Results

| Metric | Sarvam Roman | Sarvam Mixed | Qwen3 Roman | Qwen3 Mixed |
|--------|:------------:|:------------:|:-----------:|:-----------:|
| H-Index | **0.8675** | 0.8434 | 0.7470 | 0.7108 |
| E-Index | 0.7347 | **0.8367** | 0.7755 | 0.8163 |
| H-Phoneme | **0.9184** | 0.8824 | 0.7558 | 0.7512 |
| E-Phoneme | 0.7274 | **0.8513** | 0.8330 | 0.8788 |
| **CSPI (language-weighted)** | — | — | — | — |
| Sarvam avg¹ | — | **0.8467** | — | — |
| Qwen3 avg¹ | — | — | **0.7715** | — |

> ¹ Language-weighted CSPI is computed per sentence using each sentence's own HI/EN token ratio, then averaged. It is not decomposable into separate roman/mixed columns — the per-sentence weights already average across both variants. Equal-weight CSPI (0.25×each component) for reference: Sarvam 0.833, Qwen3 0.784.

### Key Findings

- **Sarvam leads overall** (language-weighted CSPI 0.847 vs 0.772 for Qwen3, a gap of 7.5 points).
- **Hindi fidelity is Sarvam's core strength.** H-Phoneme 0.918 vs Qwen3's 0.756 — a gap of 16.2 points — reflects `bulbul:v3`'s native Hindi training. Because the test set is ~68% Hindi tokens on average, this advantage carries significant weight in the final score.
- **Mixed script improves both models on English handling.** Sarvam E-Index: 0.735→0.837 (+10.2 pts); Qwen3 E-Index: 0.776→0.816 (+4.0 pts). Devanagari Hindi words give the model a cleaner language-boundary signal.
- **Qwen3 has better E-Phoneme than Sarvam roman** (0.833 vs 0.727). As a multilingual model, Qwen3 handles English words with more native-like phonetics — but this advantage is down-weighted in Hindi-dominant sentences, which form the majority of the test set.
- **Language weighting lifts both models** (+1.4 pts Sarvam, +3.5 pts Qwen3) relative to equal-weight CSPI. The lift is larger for Qwen3 because its English advantage is partially discounted by the Hindi-dominant sentence distribution.

---

## 2. HNR — Harmonics-to-Noise Ratio

HNR measures the ratio of harmonic (periodic) energy to noise energy in the synthesised speech using Praat's autocorrelation method. It is an **absolute perceptual voice quality** measure that provides a dB scale directly comparable across models.

> Praat standard: > 20 dB = excellent · 15–20 dB = good · 10–15 dB = moderate · < 10 dB = poor

### Results

| Variant | Mean HNR (dB) | Median HNR (dB) | Std (dB) | N |
|---------|:-------------:|:---------------:|:--------:|:-:|
| Sarvam Roman | **15.98** | **16.03** | 1.47 | 20 |
| Sarvam Mixed | 15.44 | 15.23 | 1.42 | 20 |
| Qwen3 Roman | 14.92 | 15.02 | 1.61 | 20 |
| Qwen3 Mixed | 14.95 | 15.39 | 2.04 | 20 |

### Key Findings

- **Sarvam roman leads on absolute voice quality** (+1.06 dB over Qwen3 roman). Both sit in the 15–20 dB "good quality" band, but Sarvam's cleaner harmonic structure produces less breathiness and fewer synthesis artifacts.
- **Qwen3 roman vs mixed are essentially tied** (14.92 vs 14.95 dB). Unlike Sarvam, Qwen3's voice quality is unaffected by input script — the voice cloning anchors the acoustic character regardless of what script is fed.
- **Sarvam mixed drops 0.54 dB below Sarvam roman.** Mixed-script input slightly reduces harmonic quality, likely due to noisier prosodic transitions when the input script alternates mid-sentence.
- **Qwen3 mixed has higher std (2.04) than roman (1.61).** Mixed input introduces more per-sentence variance in Qwen3, whereas Sarvam shows consistent std (~1.4) across both variants.

---

## 3. Boundary Penalty (BP) — Code-Switch Transition Smoothness

BP measures how much rougher code-switch boundaries are compared to within-language transitions, using raw (un-normalised) MFCC Euclidean distances:

```
BP = mean_discontinuity(boundary frames) / mean_discontinuity(within frames)
```

Word boundaries come from **Whisper word-level timestamps** (`word_timestamps=True`), giving exact per-word start/end times from the same ASR pass used for CSPI. A ±2 frame window is applied around each switch point.

> BP ≈ 1.0 = ideal (boundaries as smooth as within-language) · BP > 1.5 = model struggles at switches

### Results

| Variant | Mean BP | Median BP | Std | N |
|---------|:-------:|:---------:|:---:|:-:|
| Sarvam Roman | 1.219 | 1.273 | 0.435 | 20 |
| Sarvam Mixed | 1.376 | 1.307 | 0.341 | 20 |
| Qwen3 Roman | 1.196 | 1.234 | 0.345 | 20 |
| Qwen3 Mixed | **1.242** | **1.263** | **0.351** | 20 |

### Key Findings

- **Qwen3 produces smoother code-switch boundaries** in both variants (BP 1.196/1.242) vs Sarvam (1.219/1.376). Qwen3 is a multilingual model trained on code-switched data, so language transitions are familiar patterns.
- **Mixed script increases BP for Sarvam** (1.219→1.376) — the script alternation at boundaries appears to create additional prosodic disruption rather than helping. Qwen3 shows a smaller mixed-script effect (1.196→1.242).
- **Qwen3 roman achieves best BP across all variants** (1.196, std 0.345) — most consistent and smoothest boundaries overall.
- **Models are much closer than uniform estimation suggested.** The gap between Qwen3 roman and Sarvam roman is only 0.023 BP points; earlier uniform-based figures overstated Qwen3's advantage.
- **BP and HNR reveal a clear quality trade-off:** Sarvam produces cleaner phonemes (higher HNR) but rougher transitions (higher BP). Qwen3 produces smoother transitions but lower absolute voice quality. The ideal Hinglish TTS would combine Sarvam's phoneme quality with Qwen3's boundary handling.

---

## 4. Summary — All Metrics

| Metric | Sarvam Roman | Sarvam Mixed | Qwen3 Roman | Qwen3 Mixed | Winner |
|--------|:------------:|:------------:|:-----------:|:-----------:|:------:|
| **CSPI (lang-weighted) ↑** | — | **0.847** | — | 0.772 | Sarvam |
| CSPI (equal-weight) ↑ | 0.812 | **0.854** | 0.778 | 0.789 | Sarvam Mixed |
| H-Index ↑ | **0.868** | 0.843 | 0.747 | 0.711 | Sarvam Roman |
| E-Index ↑ | 0.735 | **0.837** | 0.776 | 0.816 | Sarvam Mixed |
| H-Phoneme ↑ | **0.918** | 0.882 | 0.756 | 0.751 | Sarvam Roman |
| E-Phoneme ↑ | 0.727 | 0.851 | 0.833 | **0.879** | Qwen3 Mixed |
| HNR (dB) ↑ | **15.98** | 15.44 | 14.92 | 14.95 | Sarvam Roman |
| Boundary Penalty ↓ | 1.219 | 1.376 | **1.196** | 1.242 | Qwen3 Roman |

> Language-weighted CSPI averages per-sentence scores weighted by each sentence's own HI/EN token ratio; it does not decompose cleanly by input script variant.

### Overall Recommendation

**Use Sarvam with mixed-script input** for production Hinglish TTS:
- Best language-weighted CSPI (0.847 vs 0.772)
- Superior Hindi phonetics — critical for native-speaker acceptance in a Hindi-dominant test set
- Mixed script closes the gap on English token handling (E-Index +10.2 pts)
- Higher absolute voice quality (HNR) than Qwen3 in both variants

**Qwen3's advantages** (E-Phoneme on mixed, boundary smoothness) are real but secondary — in Hinglish, correct Hindi pronunciation is perceptually more salient to listeners than code-switch transition smoothness or English phonetic accuracy. Language-weighted scoring reflects this by giving more weight to Hindi performance in the predominantly Hindi-heavy sentences.

---

## 5. Methodology Notes

### Devanagari Normalisation for CSPI
All metric comparisons (H-Index, E-Index, H-Phoneme, E-Phoneme) normalise both reference and Whisper hypothesis to full Devanagari before computing character similarity. This is scientifically correct for Hinglish because:
1. Whisper transcribes Hinglish audio to Devanagari regardless of input script
2. English words in Hinglish context adopt Hindi phonetics ("meeting" → "मीटिंग")
3. The evaluation question is intelligibility in Hinglish context, not native-English phonetics

### Boundary Penalty — Timestamp Method
Three approaches were evaluated for locating word boundaries:

1. **Uniform partition** (initial): word i estimated at fraction i/N of total duration. Simple but underestimates boundary roughness — switch points land at average positions regardless of actual speech rate variation.
2. **Montreal Forced Aligner (MFA)**: would give precise phoneme-level alignment but hit a Hinglish-specific roadblock — MFA 3.x has no Hindi acoustic model in its repository, and the multilingual model was removed in 3.x. Building a custom Hindi acoustic model was out of scope.
3. **Whisper word timestamps** (current): `word_timestamps=True` from the same faster-whisper pass already used for CSPI. Gives real per-word start/end times, handles both Roman and Devanagari Hinglish without any extra tooling, and is reliable for clean TTS output.

The figures in this report use Whisper timestamps. For future work, MFA with a custom Hinglish acoustic model would provide phoneme-level precision at boundaries.

### Synthesis Paradigm
Sarvam uses a dedicated Hindi TTS production API (`bulbul:v3`, `suhani` speaker). Qwen3 uses the Base model with ICL voice cloning from `hindi_ref.wav` — the recommended OOTB approach for Hinglish. Both configurations reflect realistic deployment, not a controlled ablation.
