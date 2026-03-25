# Subjective Audio Quality Evaluation

**Date**: 2026-03-25  
**Evaluator**: User  
**Methodology**: Informal MOS-style scoring on 1-5 scale for code-switching quality  
**Focus dimensions**: Naturalness, Prosodic Blending, Phonetic Accuracy

---

## Test Case T07: Noun Insertion
**Text**: "Aaj ka meeting bahut lamba tha"  
**English**: "Today's meeting was very long"  
**Challenge**: English noun (meeting) inserted into Hindi sentence; needs smooth prosodic transition

### Scoring Template

| Model | Naturalness | Prosodic Blending | Phonetic Accuracy | Notes |
|-------|---|---|---|---|
| **Qwen3-TTS** | 3 | 3 | 3.5| |
| **Fish Audio S2** | 4 | 3.5 | 4 | |
| **XTTS-v2** | — | — | — | _Note: Audio is much longer (7s) than others_ |
| **CosyVoice3** | 2.5 | 2 | 2.5 | |

---

## Test Case T09: Verb Grafting  
**Text**: "Pehle send karo phir baat karte hain"  
**English**: "First send, then let's talk"  
**Challenge**: English verb stem (send) with Hindi grammatical suffix; tests phoneme mapping

### Scoring Template

| Model | Naturalness | Prosodic Blending | Phonetic Accuracy | Notes |
|-------|---|---|---|---|
| **Qwen3-TTS** | 2.5 | 2 | 2| |
| **Fish Audio S2** | 3 | 2 | 2 | |
| **XTTS-v2** | — | — | — | _Note: Audio is much longer (5s) than others_ |
| **CosyVoice3** | 1 | 1| 1.5 | |

---

## Key Observations to Document

After listening, please note:

1. **Naturalness** — Which model sounds most like a native Hinglish speaker?
2. **Prosodic Blending** — Where do you hear the biggest jarring switches? Seamless blends?
3. **Phonetic Accuracy** — Any mispronunciations of Hindi or English words?
4. **Artifacts** — Do you notice stutters, pauses, audio artifacts, voice quality inconsistencies?
5. **Cold-start issues** — Does Fish Audio have stuttering at the beginning (as you mentioned)?

---

## Observations to Add

(You can add any additional findings here about which metrics aligned with subjective quality, where objective and subjective diverged, etc.)

**T07 (Noun Insertion) Observations:**
- Fish Audio sounds best for Hindi, however "meeting" sounds like "making"
- Qwen3-TTS comes second — sounds like accented Hindi, English word is correct
- XTTS is complete garbage
- CosyVoice sounds like a South Indian speaker Hinglish with slow pace and unnatural prosody

---

## Step 1 Validation: E-Index (English Token Recognition)

After initial evaluation, we computed **E-Index** to measure English token recognition (parallel to H-Index for Hindi). This validates your subjective observations:

| Model | E-Index | H-Index | Findings |
|---|---|---|---|
| CosyVoice3 | 0.7619 | 0.0% | Excellent English, zero Hindi — explains "South Indian speaker" perception |
| Qwen3-TTS | 0.5000 | 67.95% | Balanced moderate performance on both |
| Fish Audio S2 | 0.4839 | 71.43% | Better Hindi than English, silent failures complicate picture |
| XTTS-v2 | 0.0690 | 42.25% | **Catastrophic English failure** — only 2/29 tokens correct |

**Key Validation:**
- Your "XTTS is garbage" judgment is confirmed: 6.9% E-Index means it's getting almost all English words wrong
- Fish Audio's "sounds good but meeting→making" aligns with 48.39% E-Index (hits ~half the English tokens, but misses critical phonemes)
- CosyVoice's inverted behavior (good English, no Hindi) explains the "non-native accent" perception

**Critical Discovery:** H-Index alone masked XTTS-v2's English catastrophe (42.25% H-Index sounds acceptable but E-Index 6.9% proves it's unusable for code-switching)



