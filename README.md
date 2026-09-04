# Hindi Medical Audio ASR - Model Comparison Framework

Automated Speech Recognition (ASR) evaluation system for Hindi medical audio transcription. Compare multiple ASR models including Whisper variants, AI4Bharat models, IndicWav2Vec, Qwen-Audio, and Gemini Audio API.

## 🎯 Features

- **Multiple ASR Models**: Support for Whisper, AI4Bharat, IndicWav2Vec (CTC), Qwen-Audio, and Gemini Audio
- **Flexible Evaluation**: Run individual models or compare multiple models
- **Comprehensive Reports**: Generate reports in Text, CSV, JSON, and Markdown formats
- **YAML Configuration**: Easy model configuration via YAML files
- **Command-Line Interface**: Simple CLI for all operations
- **GPU Acceleration**: Automatic GPU detection and usage when available
- **Detailed Metrics**: WER, CER, inference time, model size, and more

## 📁 Project Structure

```
cst-up-project/
├── src/                   # Source code
│   ├── data/              # Audio processing and ground truth loading
│   │   ├── audio_processor.py
│   │   └── ground_truth_loader.py
│   ├── models/            # ASR model implementations
│   │   ├── base_model.py
│   │   ├── whisper_model.py
│   │   ├── ctc_model.py
│   │   ├── qwen_model.py
│   │   └── gemini_model.py
│   ├── evaluation/        # Evaluation and metrics calculation
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── reporting/         # Report generation (TXT, CSV, JSON, HTML, MD)
│   │   ├── report_generator.py
│   │   ├── html_reporter.py
│   │   └── formatters.py
│   └── utils/             # Utilities (device detection, transliteration)
│       ├── device_utils.py
│       ├── model_utils.py
│       └── transliteration.py
├── configs/               # YAML configuration files (10 models)
│   ├── whisper_medium.yaml
│   ├── whisper_small.yaml
│   ├── ai4bharat_whisper.yaml
│   ├── vaani_medium.yaml
│   ├── oriserve_apex.yaml
│   ├── oriserve_swift.yaml
│   ├── shunya_hinglish.yaml
│   ├── indicwav2vec.yaml
│   ├── qwen.yaml
│   └── gemini.yaml
├── scripts/               # Runner scripts
│   ├── run_comparison.py  # Multi-model comparison (main script)
│   ├── run_whisper_medium.py
│   ├── run_ai4bharat.py
│   ├── run_vaani_medium.py
│   ├── run_oriserve_apex.py
│   ├── run_oriserve_swift.py
│   ├── run_shunya_hinglish.py
│   ├── run_indicwav2vec.py
│   ├── run_qwen.py
│   └── create_devanagari_ground_truth.py
├── assets/                # Audio files and ground truth
│   ├── grounds_truth.txt
│   ├── grounds_truth_dual.txt
│   └── *.wav              # 400+ audio files
├── docs/                  # Documentation
│   ├── DUAL_SCRIPT_GUIDE.md
│   └── TRANSLITERATION_ANALYSIS.md
├── test/                  # Unit tests
│   └── test_transliteration.py
├── results/               # Generated reports (timestamped)
│   └── YYYYMMDD-HHMMSS/   # Timestamped folders
│       ├── report.html
│       ├── report.txt
│       ├── summary.csv
│       ├── detailed.csv
│       ├── results.json
│       ├── comparison.md
│       └── metadata.json
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Setup Assets

Place your audio files and ground truth in the `assets` folder:

```
assets/
├── grounds_truth.txt
├── audio1.wav
├── audio2.wav
└── ...
```

**Ground Truth Format** (`assets/grounds_truth.txt`):
```
audio1.wav | Mareej ki sthiti sthir hai  | मरीज की स्थिति स्थिर है | Diagnosis

0054_M.wav | Mera naam Abhishek Tiwari hai, meri age 31 saal hai, mai Rohini Enclave, Banda Road, UP me rehta hoon. Doctor sahab, Neend aane mein bahut dikkat hoti hai, baar-baar neend toot jaati hai, aur din bhar thakan rehti hai | मेरा नाम अभिषेक तिवारी है, मेरी उम्र 31 साल है, मैं रोहिणी एन्क्लेव, बांदा रोड, यूपी में रहता हूँ। डॉक्टर साहब, नींद आने में बहुत दिक्कत होती है, बार-बार नींद टूट जाती है, और दिन भर थकान रहती है। | Chronic Insomnia

```

### Run Individual Model for quick test

```bash
# Run Whisper Medium
python scripts/run_whisper_medium.py --max-files 2

# Run AI4Bharat Whisper
python scripts/run_ai4bharat.py --max-files 2

# Run Vaani Medium
python scripts/run_vaani_medium.py --max-files 2

# Run Oriserve Apex (Hinglish)
python scripts/run_oriserve_apex.py --max-files 2

# Run Oriserve Swift (Hinglish)
python scripts/run_oriserve_swift.py --max-files 2

# Run Shunya Hinglish
python scripts/run_shunya_hinglish.py --max-files 2

# Run IndicWav2Vec
python scripts/run_indicwav2vec.py

# Run Qwen-Audio
python scripts/run_qwen.py

# Custom assets path and options
python scripts/run_whisper_medium.py --assets-path /path/to/assets --max-files 10
```

### Run Model Comparison

```bash
# Compare default models
python scripts/run_comparison.py

# Compare specific models
python scripts/run_comparison.py --models whisper_medium ai4bharat vaani_medium

# Compare all Whisper variants
python scripts/run_comparison.py --models whisper_medium whisper_small ai4bharat vaani_medium

# Compare Hinglish models
python scripts/run_comparison.py --models oriserve_apex oriserve_swift shunya_hinglish

# Compare with custom configs
python scripts/run_comparison.py --configs configs/whisper_medium.yaml configs/qwen.yaml

# Limit evaluation to the first 50 files and use batching
python scripts/run_comparison.py --max-files 50 --batch-size 8

# Randomly sample 20 files (reproducible with --seed)
python scripts/run_comparison.py --max-files 20 --sampling random --seed 123

# Custom assets path
python scripts/run_comparison.py --assets-path /path/to/custom/assets
```

### Sampling & Batching Controls

The comparison runner now supports rapid experiments by evaluating a subset of files and/or batching audio inputs (for models that support it, currently Whisper and IndicWav2Vec).

- `--max-files N` limits evaluation to the first `N` items (after sorting by filename)
- `--sampling random` picks a random subset instead of the first `N` files (use `--seed` for reproducibility)
- `--batch-size K` enables batched transcription (recommended 8-12 for sub-30s clips on 4 GB GPUs)

Example:

```bash
python scripts/run_comparison.py \
  --max-files 32 \
  --sampling random \
  --seed 42 \
  --batch-size 12
```

## 📊 Generated Reports

Each evaluation generates timestamped reports in `results/YYYYMMDD-HHMMSS/`:

- **report.html**: Human-readable html report with visuals, can be opened in browser
- **report.txt**: Human-readable text report
- **summary.csv**: Model comparison summary
- **detailed.csv**: Per-file predictions and metrics
- **results.json**: Complete results in JSON format
- **comparison.md**: Markdown comparison report (multi-model only)
- **metadata.json**: Run metadata and environment info

## ⚙️ Configuration

Models are configured via YAML files in `configs/`:

```yaml
# configs/whisper_medium.yaml
model:
  name: "Whisper Medium"
  model_id: "openai/whisper-medium"
  type: "whisper"
  language: "hindi"
  
  generation:
    max_new_tokens: 440
    num_beams: 5
    do_sample: false
```

## 🧪 Available Models

| Model | Config File | Type | Notes |
|-------|-------------|------|-------|
| Whisper Medium | `whisper_medium.yaml` | Whisper | OpenAI Whisper |
| Whisper Small | `whisper_small.yaml` | Whisper | OpenAI Whisper |
| AI4Bharat Whisper | `ai4bharat_whisper.yaml` | Whisper | Fine-tuned for Hindi |
| Vaani Medium | `vaani_medium.yaml` | Whisper | Sarvam AI Hindi Whisper |
| Oriserve Apex | `oriserve_apex.yaml` | Whisper | Hindi2Hinglish (romanized) |
| Oriserve Swift | `oriserve_swift.yaml` | Whisper | Hindi2Hinglish (romanized) |
| Shunya Hinglish | `shunya_hinglish.yaml` | Whisper | Code-switching support |
| IndicWav2Vec | `indicwav2vec.yaml` | CTC | AI4Bharat CTC model |
| Qwen-Audio | `qwen.yaml` | Qwen | Multimodal model |
| Gemini Audio | `gemini.yaml` | API | Requires GOOGLE_API_KEY |

## 📈 Metrics

Each evaluation provides:

- **WER (Word Error Rate)**: Lower is better (0.0 = perfect)
- **CER (Character Error Rate)**: Lower is better
- **Inference Time**: Per-file transcription time
- **Model Parameters**: Model size in millions
- **Load Time**: Model loading time
- **Total Time**: Complete evaluation time

## 🛠️ Command-Line Options

### Individual Model Scripts

```bash
python scripts/run_whisper_medium.py \
  --assets-path assets \           # Assets directory
  --config configs/whisper_medium.yaml \  # Config file
  --output-dir results              # Output directory
```

### Comparison Script

```bash
python scripts/run_comparison.py \
  --assets-path assets \            # Assets directory
  --models whisper_medium ai4bharat \  # Predefined models
  --configs config1.yaml config2.yaml \  # OR custom configs
  --max-files 50 \                  # (Optional) number of files to evaluate
  --sampling random \               # (Optional) use random subset (default: first)
  --batch-size 8 \                  # (Optional) batch size for supported models
  --output-dir results              # Output directory
```

## 🔧 Advanced Usage

### Custom Model Configuration

Create a new YAML config:

```yaml
# configs/my_custom_model.yaml
model:
  name: "My Custom Model"
  model_id: "huggingface/model-id"
  type: "whisper"  # or "ctc", "qwen", "gemini"
  language: "hindi"
```

### Environment Variables

For API-based models:

```bash
# Gemini Audio
export GOOGLE_API_KEY=your-key-here

# GPT-4 Translation (optional)
export OPENAI_API_KEY=your-key-here
export ENABLE_GPT4_TRANSLATION=true
```

### GPU Memory Management

Models automatically handle GPU out-of-memory errors by falling back to CPU. For large models:

```python
# Qwen-Audio automatically uses float16 on GPU
# Other models use full precision
```

## 📝 Example Output

```
================================================================================
WHISPER MEDIUM MODEL EVALUATION
================================================================================
🔧 Using device: cuda
   GPU: NVIDIA GeForce RTX 3080
   VRAM: 10.00 GB
Assets path: assets
Config: configs/whisper_medium.yaml

✓ Loaded 5 audio file(s)

================================================================================
Evaluating: Whisper Medium
================================================================================
⚙ Loading Whisper model...
✓ Processor loaded
✓ Model loaded
✓ Model moved to GPU
✓ Model loaded in 12.34s
✓ Parameters: 769.00M

Processing 5 audio file(s)...
  ✓ audio1.wav: WER=0.125, CER=0.045, Time=2.34s
  ✓ audio2.wav: WER=0.089, CER=0.023, Time=1.98s
  ...

✓ Evaluation completed in 15.67s
✓ Average WER: 0.1024
✓ Average CER: 0.0342

📊 Reports generated in: results/20260201-143022
   📄 Text report: report.txt
   📊 CSV summary: summary.csv
   📊 CSV detailed: detailed.csv
   📋 JSON results: results.json
   🌐 HTML report: report.html
   ℹ️  Metadata: metadata.json
```

## 🚧 Challenges

### Current Performance (Latest Benchmark: 2026-02-02, 50 files)

**Best Performing Models:**
- **Best WER:** AI4Bharat Whisper-Medium-Hi - **0.3458** (34.58% word error rate)
- **Best CER:** IndicWav2Vec Hindi (CTC) - **0.1575** (15.75% character error rate)
- **Fastest:** IndicWav2Vec Hindi (CTC) - 1.01s/file

**Overall Performance Range:**
- WER ranges from 0.3458 to 0.6611 across 8 models
- CER ranges from 0.1575 to 0.3497 across 8 models

**Key Challenges:**

1. **High Error Rates**: Even the best model shows ~35% WER, indicating significant room for improvement in medical Hindi transcription accuracy
   
2. **Number Recognition**: Models struggle with digit transcription:
   - Numbers often transcribed in words (e.g., "तीस" instead of "30")
   - Inconsistent formatting of ages, dates, and addresses
   - Poor recognition of alphanumeric codes (flat numbers, room numbers)

3. **Medical Terminology**: Domain-specific medical terms show lower accuracy due to lack of medical corpus training

4. **Model Hosting & Availability**:
   - **Hugging Face Model Availability**: Some models require gated access (need HF_TOKEN)
   - **API Dependencies**: Gemini/Qwen models require API keys and stable internet
   - **Model Download Size**: Large models (700-800M parameters) require significant bandwidth
   - **GPU Memory Requirements**: Some models need 4GB+ VRAM for efficient inference
   - **Version Changes**: Model updates on Hugging Face can break compatibility

5. **Script & Dialect Variations**:
   - Mixed Hindi-English (code-switching) scenarios
   - Regional dialect variations in medical terminology
   - Devanagari vs romanized output inconsistencies

## 🎯 Next Steps

### Planned Improvements

#### 1. **WER/CER Reduction Strategies**

**A. Post-Processing Pipeline**
- **Number Normalization**: Replace word-form digits with numeric digits
  - Pattern matching: "तीस" → "30", "चौतीस" → "34"
  - Age extraction and standardization
  - Address number formatting (flat/room numbers)
  
**B. Fine-Tuning on Medical Corpus**
- Create curated medical Hindi dataset with:
  - Medical terminology (symptoms, diagnoses, medications)
  - Common patient information patterns
  - Number and date formats used in medical contexts
  
**C. Ensemble & Hybrid Approaches**
- Combine predictions from multiple models
- Use IndicWav2Vec (best CER) + AI4Bharat (best WER)
  - CTC model for character-level accuracy
  - Whisper for contextual understanding
  
**D. Model Selection**
- **Primary target for fine-tuning**: AI4Bharat Whisper-Medium-Hi
  - Best WER baseline (0.3458)
  - Already fine-tuned for Hindi
  - Good balance of speed (10s/file) and accuracy

#### 2. **Medical Domain Adaptation**

**A. Dataset Creation**
- Collect 500+ hours of medical Hindi audio
- Annotate with medical entity tags
- Include regional dialect variations

**B. Fine-Tuning Pipeline**
```bash
# Planned workflow
1. Prepare medical corpus (audio + transcriptions)
2. Fine-tune AI4Bharat Whisper on medical data
3. Apply LoRA/QLoRA for efficient training
4. Validate on held-out medical test set
5. Target: WER < 0.20 (20%)
```

**C. Domain-Specific Vocabulary**
- Medical terminology lexicon
- Drug names database
- Symptom/diagnosis keywords

#### 3. **Infrastructure Improvements**

- **Model Caching**: Local model storage to reduce download dependency
- **Fallback Strategies**: Automatic failover between models if primary unavailable
- **Batch Processing Optimization**: Further GPU utilization improvements
- **API Rate Limiting**: Implement retry logic for API-based models

#### 4. **Evaluation Enhancements**

- **Medical-Specific Metrics**:
  - Entity-level accuracy (names, ages, addresses)
  - Number extraction accuracy
  - Critical information retention rate
  
- **Error Analysis Dashboard**:
  - Categorize errors by type (number, medical term, general)
  - Visualize per-category performance
  - Track improvement over time

#### 5. **Future Research Directions**

- **Multilingual Support**: Extend to other Indian languages (Tamil, Telugu)
- **Real-Time Transcription**: Streaming ASR for live consultations
- **Speaker Diarization**: Identify doctor vs patient speech
- **Accent Adaptation**: Regional accent handling (North vs South India)
- **Privacy-Preserving**: On-device models for sensitive medical data

### Immediate Next Actions

1. ✅ Complete baseline evaluation of all 10 models
2. 🔄 Implement number normalization post-processor
3. 📊 Analyze error patterns in medical terminology
4. 🎯 Prepare medical corpus for fine-tuning
5. 🔬 Fine-tune AI4Bharat Whisper on medical data
6. 📈 Re-evaluate and compare post-processing improvements

**Target Goals (Q1 2026)**:
- WER < 0.25 (25%) with post-processing
- WER < 0.20 (20%) with fine-tuned model
- 95%+ accuracy on number/age extraction
- Sub-5s inference time per file

## 🤝 Contributing

Contributions welcome! To add a new model:

1. Create model class in `src/models/` inheriting from `BaseASRModel`
2. Implement `load()` and `transcribe()` methods
3. Create YAML config in `configs/`
4. Create runner script in `scripts/`

