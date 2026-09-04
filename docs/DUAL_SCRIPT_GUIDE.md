# Dual-Script Ground Truth Implementation Guide

## Current Status

✅ **Implemented:**
- Ground truth loader supports dual-script format
- Script detection utilities available
- Transliteration still works as fallback

❌ **Not Yet Implemented:**
- Automatic script-based ground truth selection
- Evaluator doesn't pass model output script to loader

## How It Works (After Full Implementation)

### Step 1: Create Dual-Script Ground Truth

Run the script to convert Roman to Devanagari:
```bash
python scripts/create_devanagari_ground_truth.py
```

This creates `assets/grounds_truth_dual.txt` with format:
```
audio.wav | Roman text | देवनागरी text | Diagnosis
```

### Step 2: Automatic Script Selection

**For models that output Devanagari** (Vaani, AI4Bharat, Shunya):
- Loader automatically selects Devanagari ground truth
- Direct comparison without transliteration
- More accurate WER/CER

**For models that output Roman** (Oriserve, Whisper base):
- Loader automatically selects Roman ground truth
- Direct comparison without transliteration
- Fair evaluation

**For single-script ground truth (current):**
- Falls back to transliteration (existing behavior)
- Still works with old format

### Step 3: Comparison Results

**Before (with transliteration):**
```
Predicted (Devanagari): मेरा नाम अभिषेक है
Transliterated: mera nama abhisheka hai
Ground Truth (Roman): mera naam abhishek hai
WER: High due to transliteration differences
```

**After (with dual-script GT):**
```
Predicted (Devanagari): मेरा नाम अभिषेक है
Ground Truth (Devanagari): मेरा नाम अभिषेक है
WER: Low - exact script match!
```

## Implementation Status

### ✅ What's Working Now:

1. **Script Detection** - Can detect if text is Devanagari or Roman
2. **Transliteration** - Can convert between scripts as fallback
3. **Dual-Format Parsing** - Loader can read both formats:
   - Old: `audio.wav | text | diagnosis`
   - New: `audio.wav | roman | देवनागरी | diagnosis`

### ⚠️ What Needs Manual Setup:

1. **Generate Devanagari Ground Truth**:
   ```bash
   python scripts/create_devanagari_ground_truth.py
   ```
   This uses OpenAI API (costs ~$0.01-0.05 for 400+ lines)

2. **Use Dual-Script File**:
   Update `run_comparison.py` to use dual-script file:
   ```python
   # Change line 148:
   ground_truths = gt_loader.load("assets/grounds_truth_dual.txt")
   ```

3. **Manual Verification**:
   - Check a few Devanagari conversions are correct
   - Fix any conversion errors manually

## Expected Results

### Current Performance (with transliteration):
- Vaani Medium: WER ~96% (due to transliteration artifacts)
- Oriserve Apex: WER ~53% (Roman to Roman, no transliteration)

### Expected Performance (with dual-script):
- Vaani Medium: WER ~30-40% (Devanagari to Devanagari)
- Oriserve Apex: WER ~53% (Roman to Roman, unchanged)
- AI4Bharat: WER ~25-35% (Devanagari to Devanagari)

## Advantages

1. ✅ **Fair Comparison** - Models evaluated in their native output script
2. ✅ **Better Accuracy** - No transliteration artifacts
3. ✅ **Backward Compatible** - Works with old single-script format
4. ✅ **Automatic Detection** - Loader picks right script version
5. ✅ **Fallback** - Transliteration still works if one script missing

## Limitations

1. ⚠️ **Requires OpenAI API** - For initial Devanagari generation
2. ⚠️ **Manual Verification Needed** - Some conversions may need correction
3. ⚠️ **One-Time Setup** - Need to generate Devanagari once

## Next Steps

1. Run `create_devanagari_ground_truth.py` to generate dual-script file
2. Verify a sample of conversions
3. Use the dual-script file for evaluations
4. Compare results with and without dual-script to see improvement

## Alternative: Manual Devanagari Ground Truth

If you don't want to use OpenAI API, you can:
1. Use Google Translate or other tools
2. Manually type Devanagari versions
3. Use indic-transliteration library (less accurate)

Format:
```
0054_M.wav | Mera naam Abhishek hai | मेरा नाम अभिषेक है | Chronic Insomnia
```
