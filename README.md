# HinglishTTS-Bench: Code-Switching Evaluation Framework

> **A benchmark for evaluating TTS quality on Hindi-English code-switched speech.**

---

## 🎧 [Live Audio Examples & Metric Illustrations →](https://gprachi28.github.io/HinglishTTS/)

*Listen to synthesised Hinglish sentences side-by-side with per-sentence metric breakdowns.*

---

## Why This Benchmark Exists

**Hinglish** — the fluid mixing of Hindi and English — is the everyday language of over 500 million urban Indians. It appears in casual conversation, social media, cinema, and professional settings. Sentences like *"Aaj meeting kaafi productive thi"* or *"Mujhe seriously help chahiye yaar"* are not exceptions; they are the norm.

Despite this, no dedicated evaluation framework exists for Hindi-English code-switched TTS. Models are assessed on monolingual benchmarks and deployed on code-switched input they were never evaluated against. The result: synthesised speech that handles each language individually but fails at the seams — unnatural prosody at language switches, phoneme confusion for borrowed words, and inconsistent voice quality across a single sentence.

**Code-switching is not a dialect or a style choice — it is a distinct phonological and syntactic phenomenon.** The matrix language provides the grammatical frame; the embedded language inserts lexical items with their own phoneme inventories, stress patterns, and intonation. A TTS model must handle both simultaneously and transition between them without acoustic disruption.

### Why Standard Metrics Fail

| Metric | What It Misses |
|--------|---------------|
| WER | Treats Hindi and English errors equally; a model that mangles every Hindi token but nails English scores the same as the reverse |
| BLEU | Text similarity, not phonetic fidelity |
| MCD / F0 | Measures overall acoustic distance; cannot distinguish boundary roughness from general style |
| MOS | Requires costly human annotation; not reproducible at scale |

This framework proposes **language-aware, automatic metrics** that isolate the code-switching challenge.

---

## Proposed Metrics

### CSPI — Code-Switching Phonetic Index

Standard WER treats all tokens equally. CSPI decomposes accuracy by language and weights each sentence by its own Hindi/English token ratio — so errors in the dominant language matter more.

| Component | What It Measures |
|-----------|-----------------|
| **H-Index** | Fraction of Hindi tokens correctly recognised by Whisper ASR |
| **E-Index** | Fraction of English tokens correctly recognised by Whisper ASR |
| **H-Phoneme** | Character-level phoneme similarity for Hindi tokens (Devanagari-normalised) |
| **E-Phoneme** | Character-level phoneme similarity for English tokens (mapped to Devanagari) |

```
w_hi = n_hindi_tokens / total_tokens
w_en = n_english_tokens / total_tokens

CSPI = w_hi × (H-Index + H-Phoneme) / 2
     + w_en × (E-Index + E-Phoneme) / 2
```

A Hindi-heavy sentence (e.g. 83% HI tokens) penalises Hindi mispronunciations more; an English-heavy sentence does the reverse.

### HNR — Harmonics-to-Noise Ratio

Absolute voice quality in dB via Praat autocorrelation. Unlike MFCC-based distances, HNR is directly comparable across models and independent of speaking rate.

| Range | Quality |
|-------|---------|
| > 20 dB | Excellent |
| 15–20 dB | Good |
| 10–15 dB | Moderate |
| < 10 dB | Poor |

### Boundary Penalty (BP)

Measures acoustic roughness specifically at code-switch points, relative to within-language frames.

```
BP = mean_discontinuity(boundary frames) / mean_discontinuity(within-language frames)
```

Word boundaries are derived from **Whisper word-level timestamps** (same ASR pass as CSPI). A ±2 MFCC frame window is applied around each switch point. BP = 1.0 is ideal; BP > 1.5 indicates the model struggles at language boundaries.

> MFA forced alignment was evaluated but is not currently viable for Hinglish — MFA 3.x has no Hindi acoustic model.

---

## Models Evaluated

Full CSPI, HNR, and Boundary Penalty benchmarks were run on **two models**:

| Model | Type | Notes |
|-------|------|-------|
| **Sarvam TTS** (`bulbul:v3`) | Production API | Native Hindi speaker voice; Sarvam AI |
| **Qwen3-TTS** (1.7B Base) | Open-weight LLM-TTS | Multilingual; run locally with ICL voice cloning |

> **Synthesis paradigm note:** Sarvam uses a production API with a native Hindi speaker. Qwen3 uses the Base model with voice cloning from a reference audio clip — the recommended zero-shot approach for Hinglish without fine-tuning. The comparison reflects real-world deployment scenarios for each model, not a controlled parity setting.

> **Three additional models were attempted but excluded from full benchmarking due to reliability issues:**
> - **Fish Audio S2 Pro** — strong phoneme fidelity when it worked, but ~30% of test files produced silence on Apple M4 (MPS memory constraints suspected). Root cause unresolved.
> - **XTTS-v2** — stable inference, but WER on Hinglish was too high (1.890) to yield meaningful CSPI scores; treated as a lower-bound baseline only.
> - **CosyVoice 3** — Devanagari tokenizer gap: the underlying Qwen LLM was fine-tuned on Mandarin and English only, so Hindi script characters map to unknown tokens, producing silence on 70% of Devanagari inputs.

### Key Results (Sarvam vs Qwen3, 20-sentence test set)

Each model was evaluated on two input variants — **Roman script** (how Hinglish is naturally written) and **Mixed script** (Devanagari for Hindi tokens, Roman for English) — to test whether input script is a confounder. Both variants use the same spoken content; only the orthographic representation changes.

| Metric | Sarvam Roman | Sarvam Mixed | Qwen3 Roman | Qwen3 Mixed | What the numbers tell you |
|--------|:------------:|:------------:|:-----------:|:-----------:|--------------------------|
| CSPI (lang-weighted) ↑ | — | **0.847** | — | 0.772 | Sarvam gets ~85% of phonemes right weighted by language dominance; Qwen3 ~77%. A 7.5-point gap is perceptible in a Hindi-heavy conversation. |
| H-Index ↑ | **0.868** | 0.843 | 0.747 | 0.711 | Sarvam correctly reproduces ~87% of Hindi words; Qwen3 ~75%. A native Hindi listener would notice roughly 1 in 4 Hindi words sounding off in Qwen3. |
| E-Index ↑ | 0.735 | **0.837** | 0.776 | 0.816 | Both models handle English better with mixed-script input. Sarvam benefits more (+10 pts) — Devanagari Hindi words provide a clearer language-switch signal. |
| H-Phoneme ↑ | **0.918** | 0.882 | 0.756 | 0.751 | Sarvam's Hindi phoneme accuracy is markedly higher (~92% vs ~76%). This reflects native Hindi training data — aspirated consonants and vowel length contrasts are reproduced more faithfully. |
| E-Phoneme ↑ | 0.727 | 0.851 | 0.833 | **0.879** | Qwen3 produces more native-sounding English phonemes. Sarvam roman scores lowest here — English words are pronounced with a stronger Hindi accent. |
| HNR (dB) ↑ | **15.98** | 15.44 | 14.92 | 14.95 | Both models fall in the "good quality" band (15–20 dB). Sarvam is ~1 dB cleaner — a small but consistent difference in breathiness and synthesis artifacts. |
| Boundary Penalty ↓ | 1.219 | 1.376 | **1.196** | 1.242 | Values close to 1.0 mean language switches sound as smooth as the rest of the sentence. Both models are in the mild range; Qwen3 roman transitions are slightly less disruptive at switch points. |

> CSPI (lang-weighted) is computed per sentence and averaged across both variants — it does not decompose by script column. Mixed script closes the English-handling gap for Sarvam (+10.2 pts E-Index) but increases boundary roughness (BP 1.219→1.376). Script is a meaningful confounder for Sarvam; Qwen3 is largely script-invariant.

---

## Test Set

**20 linguistically controlled sentences** in Roman script — how Hinglish is actually written in India, not a transliteration exercise. Covering 7 code-switching patterns:

| Pattern | Roman | Mixed |
|---------|-------|-------|
| **CS-01** Noun Insertion | *"Main office ja raha hoon"* | *"मैं office जा रहा हूँ"* |
| **CS-02** Verb Grafting | *"File download kar lo"* | *"File download कर लो"* |
| **CS-03** Tag Switching | *"Baarish ho rahi hai, isn't it?"* | *"बारिश हो रही है, isn't it?"* |
| **CS-04** Clause Boundary | *"I was worried ki train miss ho jayegi"* | *"I was worried कि train miss हो जाएगी"* |
| **CS-05** Technical/Slang | *"Ye content kaafi cringe hai"* | *"ये content काफ़ी cringe है"* |
| **CS-06** Numerical/Entity | *"Meeting Thursday ko 3 PM par hai"* | *"Meeting Thursday को 3 PM पर है"* |
| **CS-07** Intraword | *"Usne mujhe unfriend kar-diya"* | *"उसने मुझे unfriend कर-दिया"* |

**Golden Set**: 300 sentences (6–12 tokens), stratified by pattern, Code-Mixing Index (CMI 0.1–0.7), and register.

---


## Structure

```
HinglishTTS/
├── data/
│   ├── codeswitching.py              # Test set generator
│   ├── codeswitched/                 # Generated sentences (CSV)
│   └── golden/                       # Golden set (300 sentences)
├── evaluation/
│   ├── compatibility/
│   │   ├── test_set.csv              # 20 test sentences
│   │   ├── compute_hindex.py         # Hindi token recognition
│   │   ├── compute_eindex.py         # English token recognition
│   │   ├── compute_phoneme_accuracy.py # Phoneme-level accuracy
│   │   ├── compute_cspi.py           # CSPI (equal-weight)
│   │   ├── compute_cspi_refined.py   # CSPI (language-aware)
│   │   ├── compute_hnr.py            # HNR voice quality
│   │   ├── compute_boundary_penalty.py # BP transition smoothness
│   │   ├── compute_word_timestamps.py  # Whisper word-level timestamps
│   │   ├── run_metrics.py            # Pipeline orchestrator
│   │   └── results/                  # Metric outputs per model
└── docs/
    └── index.html                    # Audio examples (GitHub Pages)
```

---

## Citation

If you use this framework, please cite:
```bibtex
@misc{hinglish_tts_bench_2026,
  title={HinglishTTS-Bench: A Code-Switching Evaluation Framework for Speech Synthesis},
  author={Prachi Govalkar},
  year={2026},
  url={https://github.com/gprachi28/HinglishTTS}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
