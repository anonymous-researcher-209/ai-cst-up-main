"""Test transliteration quality"""

from src.utils.transliteration import (
    devanagari_to_roman,
    normalize_for_comparison,
    detect_script,
    get_transliteration_info
)

# Test cases from actual predictions
test_cases = [
    ("मेरा नाम अभिषेक तिवारी है", "Mera naam Abhishek Tiwari hai"),
    ("मेरी एज़ इकत्तीस साल है", "meri age 31 saal hai"),
    ("डॉक्टर साहब", "Doctor sahab"),
    ("नींद आने में बहुत दिक्कत होती है", "Neend aane mein bahut dikkat hoti hai"),
]

print("=" * 80)
print("TRANSLITERATION TEST")
print("=" * 80)
print(f"\nLibrary info: {get_transliteration_info()}")
print()

for devanagari, expected_roman in test_cases:
    actual_roman = devanagari_to_roman(devanagari)
    script = detect_script(devanagari)
    
    print(f"Devanagari: {devanagari}")
    print(f"Expected:   {expected_roman}")
    print(f"Got:        {actual_roman}")
    print(f"Script:     {script}")
    print(f"Match:      {'✓' if actual_roman.lower() == expected_roman.lower() else '✗'}")
    print()

# Test the full sentence
print("=" * 80)
print("FULL SENTENCE TEST")
print("=" * 80)

devanagari_full = "मेरा नाम अभिषेक तिवारी है मेरी एज़ इकत्तीस साल है मैं रोहिनी एंग्लेफ बंदा रोड यूपी में रहता हूँ डॉक्टर साहब नींद आने में बहुत दिक्कत होती है बार बार नींद टूट जाती है और दिन भर थकान रहती है"
roman_expected = "Mera naam Abhishek Tiwari hai, meri age 31 saal hai, mai Rohini Enclave, Banda Road, UP me rehta hoon. Doctor sahab, Neend aane mein bahut dikkat hoti hai, baar-baar neend toot jaati hai, aur din bhar thakan rehti hai"

print(f"\nDevanagari:\n{devanagari_full}\n")
print(f"Expected Roman:\n{roman_expected}\n")

actual_roman = devanagari_to_roman(devanagari_full)
print(f"Actual Roman:\n{actual_roman}\n")

# Test normalization
gt_norm, pred_norm = normalize_for_comparison(roman_expected, devanagari_full)
print(f"\nNormalized Ground Truth:\n{gt_norm}\n")
print(f"Normalized Prediction:\n{pred_norm}\n")

from jiwer import wer, cer
print(f"WER (after normalization): {wer(gt_norm, pred_norm):.4f}")
print(f"CER (after normalization): {cer(gt_norm, pred_norm):.4f}")
