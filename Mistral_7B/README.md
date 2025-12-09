# Mistral 7B Cryptanalysis Experiments

This directory contains experiments evaluating **Mistral-7B-Instruct-v0.3** on cryptanalysis tasks, comparing standard tokenization vs. character-level tokenization.

##  Contents

- **`mistral_experiments.ipynb`** - Main Jupyter notebook containing all experiments
- **`Results/`** - Experimental results and analysis
  - `rq1_experiments_results_1.csv` - Cipher decryption results (RQ1)
  - `rq2_rq3_results.csv` - Cipher type identification results (RQ2 & RQ3)


##  Setup

### Requirements
```bash
pip install datasets pandas tqdm torch transformers python-Levenshtein accelerate
```

### Model
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Tokenizer:** SentencePiece (subword BPE variant)
- **Hardware:** CUDA-enabled GPU (tested on single GPU setup - Colab A100 - 40GB Memory required)



##  Results Summary

### RQ1: Decryption Performance
- **Standard Tokenization:** 0.00% exact match, 0.1671 Levenshtein similarity
- **Character-Level:** 0.00% exact match, 0.1542 Levenshtein similarity
- **Winner:** Standard tokenization (+1.29% similarity)

### RQ2: Cipher Identification (with Plaintext)
- **Standard Tokenization:** 14.28% accuracy
- **Character-Level:** 12.44% accuracy
- **Winner:** Standard tokenization (+1.84%)

### RQ3: Cipher Identification (Ciphertext Only)
- **Standard Tokenization:** 28.70% accuracy
- **Character-Level:** 11.71% accuracy
- **Winner:** Standard tokenization (+16.99%)

## Key Findings

1. **Standard tokenization outperforms character-level** across all three research questions
2. **Exact match rates are 0%** - Mistral-7B-Instruct-v0.3 struggle significantly with cipher decryption
3. **Cipher identification is more feasible** than actual decryption
4. **Exception:** Playfair Cipher shows +9.15% improvement with character-level in RQ1
5. **Modern crypto (RSA, AES)** remains extremely challenging for LLMs regardless of tokenization

##  Running the Experiments

### Full Experiment
```python
# Load dataset
crypt_dataset = load_dataset("Sakonii/EncryptionDataset")

# Add character-level versions
updated_crypt_dataset = crypt_dataset.map(add_character_spacing)

# Run RQ1 (Decryption)
run_rq1_experiment(updated_crypt_dataset)

# Run RQ2 & RQ3 (Classification)
run_rq2_rq3_experiment(updated_crypt_dataset)
```

##  Evaluation Metrics

### Exact Match (EM)
Binary metric: Does the prediction exactly match the ground truth (normalized)?
```python
normalize(prediction) == normalize(ground_truth)
```

### Normalized Levenshtein Similarity
Continuous metric (0-1): How similar are the strings?
```python
similarity = 1.0 - (levenshtein_distance / max_length)
```
- **1.0** = Perfect match
- **0.0** = Completely different



### Model Parameters
- `temperature=0.1` (low randomness)
- `max_new_tokens=100` (concise outputs)
- `do_sample=True` (slight variation)


