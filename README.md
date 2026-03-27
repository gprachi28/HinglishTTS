# HinglishTTS-Bench: Code-Switching Evaluation Framework

> **A code-switching TTS evaluation framework, demonstrated on Hinglish.**

---

## 🎧 [Live Audio Examples & Metric Illustrations →](https://gprachi28.github.io/HinglishTTS/)

*Listen to synthesised Hinglish sentences side-by-side with per-sentence metric breakdowns.*

---

**Hinglish** — the fluid mixing of Hindi and English — is the everyday language of over 500 million urban Indians. Sentences like *"Aaj meeting kaafi productive thi"* or *"Mujhe seriously help chahiye yaar"* are not exceptions; they are the norm. Yet no dedicated evaluation framework exists for code-switched TTS. This project builds one.

---

## 🌍 Generalises Beyond Hinglish

The framework is language-pair agnostic. CSPI's matrix-language weighting means the dominant language in each sentence automatically gets more influence on the score — no hardcoded assumptions about Hindi or English. To adapt it for another language pair, provide a tagged test set and the appropriate ASR/phoneme normalisation:

- 🇪🇸 **Spanglish** (Spanish–English)
- 🇰🇷 **Konglish** (Korean–English)
- 🇵🇭 **Taglish** (Tagalog–English)
- Any other bilingual mixing pattern

---

## ❌ Why Standard Metrics Fail

Code-switching is not a dialect or a style choice — it is a distinct phonological and syntactic phenomenon. The matrix language provides the grammatical frame; the embedded language inserts lexical items with their own phoneme inventories and stress patterns. Standard metrics ignore this entirely:

| Metric | What It Misses |
|--------|---------------|
| WER | Treats Hindi and English errors equally — a model that mangles every Hindi token but nails English scores the same as the reverse |
| BLEU | Text similarity, not phonetic fidelity |
| MCD / F0 | Overall acoustic distance; cannot distinguish boundary roughness from general style |
| MOS | Requires costly human annotation; not reproducible at scale |

This framework proposes **language-aware, automatic metrics** that isolate the code-switching challenge.

---

## 📐 Metrics

| Metric | What It Captures | Scale |
|--------|-----------------|-------|
| **CSPI** — Code-Switching Phonetic Index | Phonetic fidelity across both languages, weighted by how dominant each language is in each sentence | 0–1 ↑ |
| **L1-Index / L2-Index** | Fraction of matrix / embedded language tokens correctly reproduced via Whisper ASR | 0–1 ↑ |
| **L1-Phoneme / L2-Phoneme** | Character-level phoneme similarity for matrix / embedded language tokens | 0–1 ↑ |
| **HNR** — Harmonics-to-Noise Ratio | Absolute voice quality in dB via Praat autocorrelation | >20 dB excellent · 15–20 good · 10–15 moderate |
| **Boundary Penalty (BP)** | Acoustic roughness at code-switch points relative to within-language frames | 1.0 = ideal · >1.5 = struggles |

**L1 = matrix language** (the language providing the grammatical frame) · **L2 = embedded language** (the language inserting lexical items). For Hinglish: L1 = Hindi, L2 = English. For Spanglish: L1 = Spanish, L2 = English.

CSPI combines L1-Index, L2-Index, L1-Phoneme, and L2-Phoneme, weighted per sentence by each sentence's own L1/L2 token ratio. A L1-heavy sentence penalises L1 mispronunciations more; a L2-heavy sentence does the reverse. BP uses Whisper word-level timestamps to localise switch points in the audio.

---

## 🗂️ Test Set

20 linguistically controlled sentences covering 7 code-switching patterns, evaluated in two orthographic variants:

| Pattern | Roman | Mixed |
|---------|-------|-------|
| **CS-01** Noun Insertion | *"Main office ja raha hoon"* | *"मैं office जा रहा हूँ"* |
| **CS-02** Verb Grafting | *"File download kar lo"* | *"File download कर लो"* |
| **CS-03** Tag Switching | *"Baarish ho rahi hai, isn't it?"* | *"बारिश हो रही है, isn't it?"* |
| **CS-04** Clause Boundary | *"I was worried ki train miss ho jayegi"* | *"I was worried कि train miss हो जाएगी"* |
| **CS-05** Technical/Slang | *"Ye content kaafi cringe hai"* | *"ये content काफ़ी cringe है"* |
| **CS-06** Numerical/Entity | *"Meeting Thursday ko 3 PM par hai"* | *"Meeting Thursday को 3 PM पर है"* |
| **CS-07** Intraword | *"Usne mujhe unfriend kar-diya"* | *"उसने मुझे unfriend कर-दिया"* |

Roman script reflects how Hinglish is actually written in India. Mixed script (Devanagari for Hindi tokens, Roman for English) was included to test whether input orthography is a confounder — it is, for some models.

---

## 🤖 Models & Results

Full benchmarks were run on two models:

| Model | Type |
|-------|------|
| **Sarvam TTS** (`bulbul:v3`) | Production API — native Hindi speaker voice |
| **Qwen3-TTS** (1.7B Base) | Open-weight LLM-TTS — run locally with ICL voice cloning |

> **Synthesis paradigm:** Sarvam uses a production API with a native Hindi speaker. Qwen3 uses the Base model with voice cloning from a reference audio clip — the recommended zero-shot approach for Hinglish without fine-tuning. The comparison reflects real-world deployment scenarios, not a controlled parity setting.

Three additional models were attempted but could not be fully benchmarked: **Fish Audio S2 Pro** (~30% silent failures on Apple M4), **XTTS-v2** (WER too high for meaningful CSPI scores), and **CosyVoice 3** (Devanagari tokenizer gap — silent failures on Hindi script input).

### 📊 Results

| Metric | Sarvam Roman | Sarvam Mixed | Qwen3 Roman | Qwen3 Mixed |
|--------|:------------:|:------------:|:-----------:|:-----------:|
| CSPI (lang-weighted) ↑ | — | **0.847** | — | 0.772 |
| L1-Index ↑ | **0.868** | 0.843 | 0.747 | 0.711 |
| L2-Index ↑ | 0.735 | **0.837** | 0.776 | 0.816 |
| L1-Phoneme ↑ | **0.918** | 0.882 | 0.756 | 0.751 |
| L2-Phoneme ↑ | 0.727 | 0.851 | 0.833 | **0.879** |
| HNR (dB) ↑ | **15.98** | 15.44 | 14.92 | 14.95 |
| Boundary Penalty ↓ | 1.219 | 1.376 | **1.196** | 1.242 |

**💡 What the numbers show:**
- Sarvam's L1-Phoneme (Hindi) accuracy is substantially higher (~92% vs ~76%), reflecting native Hindi training. In a L1-dominant test set, this drives the overall CSPI gap.
- Qwen3 produces more native-sounding L2 (English) phonemes (L2-Phoneme 0.879 vs 0.727 for Sarvam Roman) — expected from a multilingual model.
- Both models sit in the "good quality" HNR band (15–20 dB); Sarvam is ~1 dB cleaner.
- Boundary Penalty is close across variants (1.19–1.38) — neither model struggles severely at switch points.
- Mixed-script input helps Sarvam on L2 handling (+10 pts L2-Index) but increases boundary roughness. Qwen3 is largely script-invariant.

---

## 🗃️ Structure

```
HinglishTTS/
├── data/
│   ├── codeswitching.py              # Test set generator
│   ├── codeswitched/                 # Generated sentences (CSV)
│   └── golden/                       # Golden set (300 sentences)
├── evaluation/
│   └── compatibility/
│       ├── test_set.csv              # 20 test sentences
│       ├── compute_hindex.py         # Hindi token recognition
│       ├── compute_eindex.py         # English token recognition
│       ├── compute_phoneme_accuracy.py # Phoneme-level accuracy
│       ├── compute_cspi.py           # CSPI (equal-weight)
│       ├── compute_cspi_refined.py   # CSPI (language-aware)
│       ├── compute_hnr.py            # HNR voice quality
│       ├── compute_boundary_penalty.py # BP transition smoothness
│       ├── compute_word_timestamps.py  # Whisper word-level timestamps
│       ├── run_metrics.py            # Pipeline orchestrator
│       └── results/                  # Metric outputs per model
└── docs/
    └── index.html                    # Audio examples (GitHub Pages)
```

---

## 📎 Citation

```bibtex
@misc{hinglish_tts_bench_2026,
  title={HinglishTTS-Bench: A Code-Switching Evaluation Framework for Speech Synthesis},
  author={Prachi Govalkar},
  year={2026},
  url={https://github.com/gprachi28/HinglishTTS}
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with [Claude Code](https://claude.ai/code).*
