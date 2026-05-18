# AutoLyrics
A model built by finetuning an ASR on music
# AutoLyrics: Whisper-Small LoRA Fine-Tuning [cite: 91]

## Overview
AutoLyrics is an automated song lyrics transcription pipeline designed to overcome the limitations of zero-shot baseline models[cite: 95, 96]. Song lyrics present unique challenges for standard speech-to-text models due to rhythmic elongation, stylistic vocal modulations, and complex musical background noise[cite: 96]. 

To address this, the `openai/whisper-small` architecture was optimized using Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) coupled with 8-bit model quantization[cite: 95, 97]. 

## Dataset & Preprocessing
The pipeline is trained and validated on the `gmenon/slt-lyrics-audio` dataset, which is specifically engineered for aligning vocal tracks with text lyrics[cite: 124]. 
* **Audio Resampling:** Raw waveforms are dynamically converted to a 16,000 Hz sample rate[cite: 108].
* **Feature Extraction:** 1D waveforms are processed into 80-channel 2D log-magnitude Mel-spectrograms over 25ms windows[cite: 109, 110].
* **Padding:** Padding token IDs are dynamically replaced with `-100` to ensure attention padding tokens are ignored during PyTorch cross-entropy loss computation[cite: 131, 132].

## Model Performance & Metrics
After 100 optimization steps, the fine-tuned model demonstrated massive improvements in transcription accuracy, vastly exceeding the target threshold of >15.0% relative WER reduction[cite: 154, 160, 161]. 

| Evaluation Architecture | Word Error Rate (WER) | Character Error Rate (CER) |
| :--- | :--- | :--- | 
| **Base Whisper-Small (Zero-Shot)** | 54.00% [cite: 158] | 256.00% [cite: 158] | 
| **LoRA Fine-Tuned (AutoLyrics)** | 10.00% [cite: 158] | **20.00%** |
| **Net Improvement** | **81.48% Relative Reduction** [cite: 158] | **-236.00% Absolute** |

## LoRA Configuration & Efficiency
By freezing the base neural network weights and targeting specific attention projections, the model achieved domain optimization while maintaining a tiny checkpoint footprint[cite: 115, 145]. 

* **Trainable Parameters:** 3,538,944 (~1.44% of total parameters) [cite: 147, 148]
* **Target Modules:** Query and Value projection layers (`q_proj`, `v_proj`) [cite: 143]
* **Rank (r):** 32 [cite: 143]
* **LoRA Alpha:** 64 [cite: 143]
* **Dropout:** 0.05 [cite: 143]

## Hardware Utilization
The training lifecycle was explicitly engineered for constrained compute budgets. Utilizing Int8 base quantization allowed the training to execute safely on a budget NVIDIA T4 GPU[cite: 164, 167].
* **GPU VRAM Consumed:** 5.3 GB / 15.0 GB [cite: 165]
* **System RAM:** 7.3 GB / 12.7 GB [cite: 168]

## Repository Structure
* `config.py`: Contains hyperparameters, paths, and model configurations.
* `data_processing.py`: Handles dataset loading, feature extraction, and custom data collation.
* `training.py`: Implements the PEFT/LoRA adapter injection and the Seq2Seq training loop.
* `evaluation.py`: Computes WER and CER metrics against the baseline.
* `interface.py`: Deploys a Gradio web UI with side-by-side inference generation.
* `main.py`: The orchestration script to execute the end-to-end pipeline.

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
