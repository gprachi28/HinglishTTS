# 🗣️ Hinglish TTS — Code-Switched Hindi-English Speech Synthesis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> ⚠️ **Work in Progress** — This project is actively under development. APIs, data formats, and model interfaces may change without notice.

```
Neural Text-to-Speech synthesis for Hinglish — the natural code-switched variety of Hindi and English spoken by 350M+ people.
```


## 🎧 Demo
[**Try it live on Hugging Face Spaces →**](https://huggingface.co/spaces/YOUR_USERNAME/hinglish-tts)

Sample outputs:

| Input | Audio |
|---|---|
| "Aaj ka meeting bahut boring tha" | [▶ Play] |
| "Please send me the report kal tak" | [▶ Play] |
| "Main yeh project finish kar dunga" | [▶ Play] |

---

## 🔍 Motivation

Hinglish is spoken by an estimated 350 million people across India and the diaspora, yet almost no TTS research addresses it. Most systems either:
- Fail on Hindi words when using English TTS
- Fail on English words when using Hindi TTS
- Produce unnatural prosody at language switch points

This project addresses that gap by fine-tuning a neural TTS system on code-switched data with linguistically motivated switching patterns.

---

## 🏗️ Architecture
```
Raw Text (Hinglish)
        │
        ▼
┌─────────────────┐
│ Language Tagger │  ← word-level HI/EN detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Normalizer│  ← numbers, dates, abbreviations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   G2P Pipeline  │  ← eSpeak-NG (EN) + transliteration + eSpeak-NG (HI)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VITS Model    │  ← fine-tuned on LJSpeech + IndicVoices-R
└────────┬────────┘
         │
         ▼
    Audio Output
```

---

## 📦 Datasets

| Dataset | Language | Size | Quality | Source |
|---|---|---|---|---|
| LJSpeech | English | ~24h | Studio, 22kHz | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) |
| IndicVoices-R | Hindi | ~46h (filtered) | LJSpeech-quality | [HuggingFace](https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi) |

IndicVoices-R was accepted at **NeurIPS 2024** and matches LJSpeech recording quality.

### Data filtering criteria
- SNR ≥ 20dB
- Duration: 1–15 seconds
- Speaking rate: 2–8 syllables/second

### Why not existing Hinglish corpora?

Two natural candidates were evaluated and ruled out:

- **GLUECoS** — integration was explored but abandoned due to deprecated Twitter API dependencies, a known issue affecting the broader research community.
- **L3Cube-HingLID** — scraped from Twitter, making it inherently noisy, informal, and unsuitable for TTS where naturalness is critical.

Due to the absence of clean, naturally occurring Hinglish speech corpora suitable for TTS, this project adopts a **linguistically motivated synthetic sentence generation** approach based on Equivalence Constraint theory, validated by native Hinglish speakers.

---

## 🔄 Code-Switching Approach

Sentences are generated using 5 linguistically motivated switching patterns
based on the **Equivalence Constraint** theory:

| Pattern | Example |
|---|---|
| Hindi matrix + English NP | "Aaj ka **meeting** bahut lamba tha" |
| Hindi matrix + English verb | "Please yeh **cancel** kar do" |
| English matrix + Hindi NP | "I will send it **kal**" |
| Inter-sentential | "Yeh important hai. **Please review it today**" |
| Tag switching | "**Yaar**, the deadline is tomorrow" |

Quality filtered by **Code Mixing Index (CMI)**: 0.1 ≤ CMI ≤ 0.7

---

## 🚀 Quickstart
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/hinglish-tts.git
cd hinglish-tts

# Install dependencies
pip install -r requirements.txt

# Generate code-switched sentences
python data/codeswitching.py --num_sentences 5000

# Test G2P pipeline
python normalization/g2p.py

# Run preprocessing (streaming — no full download needed)
python data/preprocess_streaming.py \
    --output_dir data/processed \
    --max_samples 1000
```

---

## 📊 Results

| System | MOS ↑ | MCD ↓ | RTF (GPU) ↑ |
|---|---|---|---|
| English-only baseline | - | - | - |
| Hindi-only baseline | - | - | - |
| **Hinglish-TTS (ours)** | **TBD** | **TBD** | **TBD** |

*Results will be updated upon completion of evaluation.*

---

## 🗂️ Project Structure
```
hinglish-tts/
├── data/               # Download and preprocessing scripts
├── normalization/      # Text normalization and G2P pipeline
├── training/           # Model training configuration
├── evaluation/         # Objective and subjective evaluation
├── serving/            # Gradio demo and Docker deployment
└── tests/              # Unit tests

---

## 🔧 Development
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest tests/ -v --cov=. --cov-report=term-missing

# Format code
black . && isort .
```

---

## 📝 Citation

If you use this work, please cite:
```bibtex
@misc{govalkar2025hinglish,
  author = {Govalkar, Prachi},
  title  = {Hinglish TTS: Code-Switched Hindi-English Speech Synthesis},
  year   = {2025},
  url    = {https://github.com/gprachi28/hinglishtts}
}
```
