# HinglishTTS-Bench: Code-Switching Evaluation Framework

A systematic evaluation framework for Hindi-English code-switched speech synthesis. Provides linguistically controlled test sets, custom metrics, and evaluation methodology for assessing TTS quality on code-mixing patterns.

## Why This Framework

Traditional metrics (WER, BLEU, MCD, F0) **do not capture code-switching quality**. A model can have low WER but fail on naturalness; high F0 continuity but wrong phonemes. This framework demonstrates the necessity of **custom language-aware metrics** for code-switched speech.

## Core Components

### 1. Test Set (7 Code-Switching Patterns)

20 linguistically controlled test sentences in Roman script (how Hinglish is actually used in India — not testing script agnosticism). Covering:
- **CS-01** Noun Insertion: "Main office ja raha hoon"
- **CS-02** Verb Grafting: "File download kar lo"
- **CS-03** Tag Switching: "Baarish ho rahi hai, isn't it?"
- **CS-04** Clause Boundary: "I was worried ki train miss ho jayegi"
- **CS-05** Technical/Slang: "Ye content kaafi cringe hai"
- **CS-06** Numerical/Entity: "Meeting Thursday ko 3 PM par hai"
- **CS-07** Intraword: "Usne mujhe unfriend kar-diya"

**Golden Set**: 300 sentences (6–12 tokens) in Roman script, stratified by pattern, CMI (0.1–0.7), and register.

### 2. Custom Evaluation Metrics (CSPI Framework)

**Problem**: Standard metrics treat all tokens equally and ignore language-specific pronunciation errors.

**Solution**: Code-Switching Phonetic Index (CSPI) combines four complementary dimensions:

| Metric | Measurement |
|--------|------------|
| **H-Index** | % Hindi tokens correctly recognized (via ASR) |
| **E-Index** | % English tokens correctly recognized (via ASR) |
| **H-Phoneme Acc** | % Hindi phonemes pronounced correctly |
| **E-Phoneme Acc** | % English phonemes pronounced correctly |

**Formula**: `CSPI = 0.25 × (H-Index + E-Index + H-Phoneme + E-Phoneme)`

**Language-Aware Refinement**: Weight metrics by Hindi/English token ratio per sentence—errors in dominant language are more noticeable.

### 3. Supporting Metrics

- **HNR** (Harmonics-to-Noise Ratio): Absolute voice quality in dB via Praat autocorrelation. Scale: >20 dB excellent, 15–20 dB good, 10–15 dB moderate.
- **Boundary Penalty (BP)**: MFCC frame discontinuity at code-switch points relative to within-language frames. BP = 1.0 is ideal; BP > 1.5 indicates the model struggles at language boundaries. Word boundaries are derived from Whisper word-level timestamps (same ASR pass as CSPI). MFA forced alignment was evaluated but is not currently viable for Hinglish — MFA 3.x has no Hindi acoustic model.

- **Reliability**: % of test cases producing usable output (accounts for silent failures)

## Project Status

| Phase | Status |
|-------|--------|
| Test set generation (7 patterns, 5000 sentences) | ✅ Complete |
| Golden set (300 sentences) | ✅ Complete |
| CSPI framework & metrics | ✅ Complete |
| Language-aware weighting | ✅ Complete |
| Golden set synthesis | 📌 Pending |

## Structure

```
HinglishTTS/
├── data/
│   ├── codeswitching.py              # Test set generator
│   ├── codeswitched/                 # Generated sentences (CSV)
│   ├── golden/                       # Golden set (300 sentences)
│   └── script_variants.py            # Roman/Devanagari/Mixed conversion
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
│   │   └── results/                  # Metric outputs
└── README.md                         # This file
```


## Citation

If you use this framework, please cite:
```
@inproceedings{hinglish_tts_bench_2026,
  title={HinglishTTS-Bench: A Code-Switching Evaluation Framework for Speech Synthesis},
  author={[Your Name]},
  year={2026}
}
```

## License

[Your license here]
