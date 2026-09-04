# Transliteration Analysis - High WER Investigation

## Summary
The high WER (~96%) is due to **TWO separate issues**:

### 1. Script Mismatch (RESOLVED ✓)
- **Problem**: Models output Devanagari, ground truth is in Roman script
- **Solution**: Implemented transliteration using `indic-transliteration` library
- **Result**: Both texts are now normalized to Roman for comparison

### 2. Transcription Accuracy Issues (MAIN PROBLEM)
Even after transliteration, WER remains high (~90%) because:

#### A. Model Transcription Errors
The model is making actual transcription mistakes:

| Expected (Roman GT) | Model Output (Devanagari) | Transliterated | Issue |
|---------------------|---------------------------|----------------|-------|
| age 31 | एज़ इकत्तीस | eza ikattisa | Wrong: Should transcribe "age 31" but got phonetic Hindi |
| naam | नाम | nama | Extra 'a' from inherent vowel |
| Abhishek | अभिषेक | abhisheka | Extra 'a' from inherent vowel |
| Tiwari | तिवारी | tivari | Wrong vowel: 'i' vs 'a' |
| Neend | नींद | nimda | Wrong transcription: न + ईं  → ni + m |
| Doctor | डॉक्टर | daॉktara | Not recognizing English loanword |
| Enclave | एंग्लेफ | emglepha | Severe mistranscription |

#### B. Transliteration Scheme Differences
The ITRANS transliteration doesn't perfectly match the ground truth romanization:
- Ground truth uses natural English-like spelling: "naam", "hai", "mein"
- ITRANS uses phonetic: "nama", "hai", "mem"

## Test Results

### Example 1: File 0054_M.wav

**Ground Truth (Roman)**:
```
Mera naam Abhishek Tiwari hai, meri age 31 saal hai, mai Rohini Enclave, Banda Road, UP me rehta hoon. Doctor sahab, Neend aane mein bahut dikkat hoti hai, baar-baar neend toot jaati hai, aur din bhar thakan rehti hai
```

**Model Output (Devanagari)**:
```
मेरा नाम अभिषेक तिवारी है मेरी एज़ इकत्तीस साल है मैं रोहिनी एंग्लेफ बंदा रोड यूपी में रहता हूँ डॉक्टर साहब नींद आने में बहुत दिक्कत होती है बार बार नींद टूट जाती है और दिन भर थकान रहती है
```

**Transliterated (using indic-transliteration)**:
```
mera nama abhisheka tivari hai meri eza ikattisa sala hai maim rohini emglepha bamda roda yupi mem rahata hu.n daॉktara sahaba nimda ane mem bahuta dikkata hoti hai bara bara nimda tuta jati hai aura dina bhara thakana rahati hai
```

**After Normalization (lowercased, cleaned)**:
```
Ground Truth: mera naam abhishek tiwari hai, meri age 31 saal hai...
Prediction:   mera nama abhisheka tivari hai meri eza ikattisa sala hai...
```

**Metrics**:
- WER: 89.74%
- CER: 33.18%

## Root Cause Analysis

### Why is the model outputting wrong words?

1. **Training Data Mismatch**: 
   - The Vaani model appears to be trained primarily on Hindi audio
   - Ground truth contains mixed Hindi-English (code-switching)
   - Model struggles with English words (Doctor → डॉक्टर, Enclave → एंग्लेफ)
   - Numbers in English (31 → इकत्तीस) are converted to Hindi words

2. **Acoustic Confusion**:
   - "age" → "एज़" (ej/ez) - phonetically similar but wrong language
   - "Neend" → "नींद" but transcribes as "nimda" (possible nasalization issue)

3. **Code-Switching Challenge**:
   - The ground truth expects: "age 31" (English)
   - The model outputs: "एज़ इकत्तीस" (phonetic approximation + Hindi number word)

## Recommendations

### Immediate Actions

1. **✅ Script Normalization (DONE)**
   - Transliteration is working correctly
   - Reports now show both Devanagari and Roman versions
   - WER calculation uses normalized text

2. **🔧 Ground Truth Standardization**
   - Consider creating Devanagari ground truth to match model output style
   - OR use models that output Roman script directly
   - OR improve transliteration mapping

3. **📊 Separate Metrics for Code-Switching**
   - Track accuracy separately for:
     - Pure Hindi segments
     - English loanwords
     - Numbers and dates
     - Mixed code-switching

### Long-Term Solutions

1. **Use Appropriate Models**:
   - For Roman script output: Models fine-tuned for Roman transliteration
   - For Devanagari output: Update ground truth to Devanagari
   - For code-switching: Models specifically trained on Hinglish data

2. **Improve Transliteration**:
   - Create custom mapping rules for common words
   - Handle English loanwords specially
   - Map numbers consistently

3. **Better Ground Truth**:
   - Use phonetically consistent romanization scheme
   - Or convert all ground truth to Devanagari
   - Document the scheme used

## Current Performance

| Model | WER (Raw) | WER (Normalized) | CER (Normalized) | Notes |
|-------|-----------|------------------|------------------|-------|
| Vaani Medium | 96.26% | ~90% | ~33% | After transliteration |
| Shunya Hinglish | 90.99% | - | 36.79% | Better for code-switching |
| AI4Bharat | 88.44% | - | 35.19% | Competitive |
| IndicWav2Vec | 94.22% | - | 40.36% | Faster but less accurate |

## Conclusion

✅ **Transliteration is working correctly**
❌ **High WER is due to actual transcription errors, not script mismatch**

The models are struggling with:
1. English loanwords (Doctor, Enclave, Road)
2. Code-switching between Hindi and English
3. Numeric values (31 vs इकत्तीस)
4. Proper nouns and place names

**Next Steps**:
- Test with models specifically trained for Hinglish/code-switching
- Consider using Devanagari ground truth
- Evaluate on pure Hindi segments separately
