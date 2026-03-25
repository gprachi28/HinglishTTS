# Phase 1.5 Evaluation Summary
## All Four Models Benchmarked — 2026-03-25

---

## Executive Summary

**All four models have been evaluated on a canonical 5-metric pipeline.** Qwen3-TTS is the clear winner for production; Fish Audio S2 Pro shows promise but requires critical reliability fixes.

### Final Rankings

| Rank | Model | H-Index | WER Mixed | PIER | F0 Mixed (Hz) | LID | Reliability | Recommendation |
|------|-------|---------|-----------|------|---------------|-----|-------------|-----------------|
| 1 | **Qwen3-TTS** | 67.95% | 0.440 | 0.500 | 33.00 | 0.600 | ✅ 100% | **PRODUCTION** |
| 2 | **Fish Audio S2** | **71.43%** | 0.605 | 0.643 | **25.27** | 0.585 | ⚠️ 70% | **Debug → Try** |
| 3 | XTTS-v2 | 42.25% | 1.890 | 0.804 | 29.91 | 0.455 | ✅ 100% | Baseline |
| 4 | CosyVoice 3 | 0.0% | 0.763 | 0.679 | 24.25 | 0.304 | ⚠️ Devanagari broken | Limited |

---

## Detailed Insights by Model

### 1. Qwen3-TTS — **RECOMMENDED FOR PRODUCTION**

**Strengths:**
- ✅ Balanced H-Index (67.95%) — near 70% target for phonetic fidelity
- ✅ Lowest WER Mixed (0.440) — best ASR-detectable accuracy
- ✅ Best PIER (0.500) — lowest error rate at code-switch boundaries
- ✅ 100% file coverage — no silent failures, all audio usable
- ✅ Functional Devanagari (WER 0.473) — only model with reliable Devanagari
- ✅ Good LID (0.600) — moderate language boundary clarity

**Weaknesses:**
- Roman script is surprisingly poor (WER 0.895) — +89.3% TPI penalty
- One extreme F0 outlier (T02_mixed: 140.27 Hz)
- Tag-switching collapses to 0% H-Index (consistent with all models)

**Optimal Use:**
- **Input:** Devanagari or Mixed-script (Roman not recommended)
- **Output:** ~68% of Hindi tokens correctly synthesized

---

### 2. Fish Audio S2 Pro — **PROMISING BUT UNRELIABLE**

**Strengths:**
- ✅ **Highest H-Index (71.43%)** — best phonetic fidelity when working
- ✅ **Best-in-class F0 (25.27 Hz)** — smoothest prosody of all models
- ✅ Perfect on technical/slang/intraword patterns (100% H-Index each)
- ✅ LID 0.585 — reasonable language boundary clarity

**Critical Weaknesses:**
- ⚠️ **~30% silent failures** — 7/20 files per variant produce no/minimal audio (T01, T02, T03, T06, T10, T12, T13, T15)
- ⚠️ **Unreliable reliability:** Only ~70% of test cases produce usable output
- ⚠️ **Severe Roman script penalty:** WER 0.926 with +43.69% TPI (unusable)
- ⚠️ **Weak PIER (0.643)** — 14+ percentage points worse than Qwen3-TTS

**Root Cause Unknown:**
Silent failures could be:
- Adapter implementation bug (wrong parameters, timeout, path issues)
- Model limitation on specific sentence patterns
- Transcription artifact (Whisper treating silence as empty)
- Memory/resource issue on M4

**Status:** **Pending debug.** If silent failures are fixed, could become co-leader with Qwen3-TTS.

---

### 3. XTTS-v2 — **USABLE BASELINE**

**Strengths:**
- ✅ 100% file coverage — all inputs produce audio
- ✅ Reasonable F0 (29.91 Hz mixed)
- ✅ Mid-tier H-Index (42.25%) — not great, but usable

**Weaknesses:**
- ❌ High WER Mixed (1.890) — ASR can barely recognize the output
- ❌ Worst PIER (0.804) — 30% worse than Qwen3-TTS on code-switch boundaries
- ❌ Lowest LID (0.455) — language boundaries blur in ASR output
- ❌ Collapses on noun insertion & tag-switching (H-Index 0%)

**Best For:**
- Acoustic baseline comparisons
- Models that prioritize sound quality over linguistic accuracy

---

### 4. CosyVoice 3 — **LIMITED USE CASE**

**Strengths:**
- Roman script usable (though not great)
- Mixed-script functional
- Quickest inference among viable models

**Weaknesses:**
- ❌ H-Index 0.0% — zero Hindi tokens recognized as Hindi
- ❌ Devanagari fundamentally broken (100% WER, 14/20 silent files)
- ❌ Lowest LID (0.304) — heavy language blending
- ❌ High PIER (0.679)

**Root Cause:**
Devanagari tokenizer gap — Qwen LLM fine-tuned on Mandarin+English only; Devanagari chars map to rare/unknown tokens → empty speech stream → silence.

**Status:** Exclude Devanagari from Phase 2. Roman/Mixed only.

---

## Critical Issues to Resolve

### Issue 1: Fish Audio S2 Pro Silent Failures

**Impact:** Highest H-Index (71.43%) but ~30% of files unusable. Real-world reliability ~70%.

**Action Items:**
1. Run single test case (e.g., T01_devanagari) with verbose logging
2. Check if output WAV file is actually created (check `/tmp/` for leftover files)
3. Verify adapter parameters match fish-speech CLI expectations
4. Test on GPU (current runs on M4 MPS; MPS memory constraints possible)
5. Try with smaller subset (e.g., T04_roman only) to isolate pattern

**Success Criteria:** All 60 files (20 sentences × 3 variants) produce audible output.

### Issue 2: CosyVoice 3 Devanagari

**Impact:** Model claims to support Devanagari but produces 100% silent failures.

**Status:** Understood root cause (tokenizer gap). **No fix planned** — this is a fine-tuning data limitation in upstream Qwen LLM.

**Workaround:** Use Roman or Mixed input only; exclude Set A (Devanagari) from Phase 2 analysis.

---

## Recommendations for Phase 2

**Proceed with Qwen3-TTS as primary evaluation model:**
- All three script variants (A/B/C)
- Full 300-sentence golden set
- Expected synthesis cost: ~2 hours on A100 GPU

**Conditional inclusion of Fish Audio S2 Pro:**
- Debug silent failures first
- If resolved, run on GPU (M4 inference too slow for 300+ sentences)
- If not resolved, skip for now; can add in Phase 2b

**Include XTTS-v2 for acoustic/linguistic baseline:**
- All three variants
- For comparative analysis only

**Exclude CosyVoice 3 from Set A (Devanagari):**
- Roman + Mixed only
- Reduces Phase 2 workload

---

## Deliverables Complete

✅ `LINGUISTIC_QUALITY_REPORT.md` — Full 4-model analysis with per-pattern breakdowns  
✅ `evaluation/compatibility/compute_*.py` — 5 canonical metrics (TPI, PIER, H-Index, F0, LID)  
✅ `evaluation/compatibility/run_metrics.py` — Master orchestrator  
✅ `evaluation/compatibility/adapters/*.py` — All four model adapters  
✅ `evaluation/compatibility/results/{model}/*.json` — Computed metrics per model  

---

## Next Steps

1. **Phase 2 Planning:** Finalize synthesis scope (golden set only? full 5000?)
2. **Debug Fish Audio S2 Pro** (parallel with Phase 2 planning)
3. **Synthesize 300 golden sentences** with Qwen3-TTS (primary)
4. **Run Phase 3 metrics** on synthesized audio
5. **Human evaluation** (Phase 4) on golden set

---

**Report Generated:** 2026-03-25  
**Models Evaluated:** 4/4  
**Metrics Computed:** 5/5  
**Status:** Phase 1.5 ✅ Complete
