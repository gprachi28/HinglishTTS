# Fish Audio S2 Pro: Silent Failure Debug Report

**Date**: 2026-03-25
**Issue**: 35% of test files fail to produce audio (21/60 files missing)
**Status**: Pattern identified, root cause investigation needed

---

## Failure Summary

Out of 60 test files (20 tests × 3 script variants):
- **39 files produced** (65%)
- **21 files missing** (35%)

---

## Failure Patterns Identified

### Pattern 1: Complete Failures (0/3 variants) — 4 Tests Affected

| Test ID | Category | Text | Why Failed? |
|---------|----------|------|------------|
| **T01** | devanagari_pure | "Aaj bahut kaam karna hai" | Pure Hindi in Devanagari |
| **T02** | devanagari_pure | "Mujhe kal subah jaldi uthna padega" | Pure Hindi in Devanagari |
| **T03** | roman_pure | "Aaj bahut kaam karna hai" | Pure Hindi in Roman script |
| **T13** | cs04_clause_boundary | "Main soch raha tha. The meeting..." | Multi-sentence input with period |

**Hypothesis 1: Pure Hindi Sentences Fail**
- Both T01 and T02 are pure Hindi (100% HI tokens, 0% EN)
- T03 is also pure Hindi (just in Roman script)
- This could indicate a prompt/reference mismatch issue (reference transcript is Hindi code-switched, not pure Hindi)

**Hypothesis 2: Multi-Sentence Inputs Fail**
- T13 contains a period: "Main soch raha tha. The meeting is really long"
- Fish Audio model may not handle sentence boundaries well
- Could cause tokenization or generation issues

### Pattern 2: Partial Failures (1-2/3 variants) — 6 Tests Affected

| Test ID | Category | Working Variants | Missing | Note |
|---------|----------|---|---|---|
| T06 | mixed_script | roman, devanagari | **mixed** | Only mixed script fails |
| T10 | cs02_verb_grafting | roman, devanagari | **mixed** | Only mixed script fails |
| T15 | cs05_technical_slang | roman | devanagari, **mixed** | Roman works, others fail |
| T12 | cs03_tag_switching | **mixed** | roman, devanagari | Only mixed works (inverted) |
| T14 | cs04_clause_boundary | **mixed** | roman, devanagari | Only mixed works (inverted) |
| T19 | cs07_intraword | devanagari, **mixed** | roman | Mostly working, only roman fails |

**Hypothesis 3: 'Mixed' Script Variant is Unreliable**
- T06, T10, T15 fail on 'mixed' variant (3 cases)
- T12, T14 work ONLY on 'mixed' (inverted failure)
- T19 works on 2/3 but not roman

**Mixed script definition**: "Devanagari for Hindi, Roman for English"
- Example: "मुझे meeting में जाना है" (Mujhe meeting mein jaana hai)
- This is linguistically correct but possibly creates adapter/tokenization issues

---

## Root Cause Hypotheses

### Hypothesis A: Reference Transcript Mismatch
**Evidence:**
- Reference transcript: "Gas connection band hone ki khabaron sun kar bahut chinta ho gayi mujhe" (code-switched)
- T01/T02/T03: Pure Hindi (no English)
- The model trained with code-switched prompt may fail on pure Hindi input

**Solution:** Test with code-switched reference instead of pure Hindi

### Hypothesis B: Multi-Sentence Handling
**Evidence:**
- T13: "Main soch raha tha. The meeting is really long"
- Only test with a period (sentence boundary)
- Model may tokenize/generate incorrectly across sentences

**Solution:** Test if model can handle sentences with periods

### Hypothesis C: Mixed Script Encoding Issue
**Evidence:**
- Multiple failures on 'mixed' variant specifically
- Mixed = Devanagari Hindi + Roman English
- Could be character encoding, byte-pair encoding (BPE) tokenization, or device-specific issue

**Solution:** Debug subprocess call with verbose logging to see tokenization/generation errors

### Hypothesis D: Device/Memory Issue (MPS on macOS)
**Evidence:**
- Running on M4 Pro with MPS backend (Apple Silicon)
- Some test cases trigger OOM or numerical instability
- Model may not cleanly handle state reset between calls

**Solution:** Check if pure CUDA or different device reduces failures

---

## Investigation Steps Needed

### Step 1: Enable Verbose Logging
Modify Fish Audio adapter to capture subprocess stderr:
```python
result = subprocess.run(cmd, capture_output=True, text=True, ...)
if result.stderr:
    print(f"Fish Audio stderr: {result.stderr}")  # Capture tokenization errors
if result.returncode != 0:
    print(f"Return code: {result.returncode}, stdout: {result.stdout[:500]}")
```

### Step 2: Test Individual Hypotheses
- **Hypothesis A:** Replace reference with pure Hindi
- **Hypothesis B:** Test with single-sentence version of T13
- **Hypothesis C:** Force character encoding UTF-8 explicitly in subprocess
- **Hypothesis D:** Test on GPU if available (CUDA)

### Step 3: Check Adapter Parameters
Verify adapter is calling Fish Audio correctly:
- Correct checkpoint path (`WEIGHTS_DIR`)
- Correct model arguments (`--prompt-text`, `--prompt-audio`)
- Environment variables properly passed (`PYTHONPATH`, `CUDA_VISIBLE_DEVICES`)

---

## Critical Questions

1. **Why does pure Hindi fail but code-switched succeed?**
   - Is the reference transcript incompatible?
   - Does the model require English in the prompt?

2. **Why does 'mixed' script have inconsistent behavior?**
   - Is it a UTF-8 encoding issue?
   - Is it a BPE tokenization issue?
   - Is it device-specific (MPS vs CUDA)?

3. **Can Fish Audio even handle pure Hindi?**
   - If not, this is a fundamental model limitation (not adapter bug)
   - Fish Audio may be optimized for code-switching, not monolingual input

---

## Quick Validation Check

To determine if this is an adapter bug vs model limitation:

1. Test Fish Audio directly on command line (skip adapter):
   ```bash
   cd models/fish_audio_s2/repo
   python -m fish_speech.models.text2semantic.inference \
     --text "Aaj bahut kaam karna hai" \
     --prompt-text "Gas connection band..." \
     --prompt-audio path/to/ref.wav \
     --checkpoint-path models/fish_audio_s2/weights \
     --device mps \
     --output /tmp/test.wav
   ```

2. Check if output file is created and has valid audio

3. Try with alternative reference (code-switched vs pure Hindi)

---

## Expected Next Steps (When GPU Available)

### Option 1: Quick Fix (If Adapter Bug)
- Add verbose logging to identify exact error
- Fix subprocess parameters or encoding
- Re-run tests

### Option 2: Workaround (If Model Limitation)
- Add error handling to gracefully skip failing cases
- Document limitations: "Fish Audio works for code-switched, fails for pure Hindi"
- Revise CSPI to account for reliability (already done: effective score ~0.45)

### Option 3: Deep Debug (If Uncertain)
- Run on GPU (A100 or V100) to rule out MPS memory issues
- Test different reference transcripts
- Check Fish Audio's GitHub issues/documentation for known limitations

