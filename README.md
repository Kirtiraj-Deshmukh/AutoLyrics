# AutoLyrics: Whisper-Small LoRA Fine-Tuning

## Overview
AutoLyrics is an automated song lyrics transcription pipeline designed to overcome the limitations of zero-shot baseline models. Song lyrics present unique challenges for standard speech-to-text models due to rhythmic elongation, stylistic vocal modulations, and complex musical background noise. 

To address this, the `openai/whisper-small` architecture was optimized using Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) coupled with 8-bit model quantization. 

## Dataset & Preprocessing
The pipeline is trained and validated on the `gmenon/slt-lyrics-audio` dataset, which is specifically engineered for aligning vocal tracks with text lyrics. 
* **Audio Resampling:** Raw waveforms are dynamically converted to a 16,000 Hz sample rate.
* **Feature Extraction:** 1D waveforms are processed into 80-channel 2D log-magnitude Mel-spectrograms over 25ms windows.
* **Padding:** Padding token IDs are dynamically replaced with `-100` to ensure attention padding tokens are ignored during PyTorch cross-entropy loss computation.

## Model Performance & Metrics
After 100 optimization steps, the fine-tuned model demonstrated massive improvements in transcription accuracy, vastly exceeding the target threshold of >15.0% relative WER reduction. 

| Evaluation Architecture | Word Error Rate (WER) | Character Error Rate (CER) |
| :--- | :--- | :--- | 
| **Base Whisper-Small (Zero-Shot)** | 54.00% | 256.00% | 
| **LoRA Fine-Tuned (AutoLyrics)** | 10.00% | **20.00%** |
| **Net Improvement** | **81.48% Relative Reduction** | **-236.00% Absolute** |

## LoRA Configuration & Efficiency
By freezing the base neural network weights and targeting specific attention projections, the model achieved domain optimization while maintaining a tiny checkpoint footprint. 

* **Trainable Parameters:** 3,538,944 (~1.44% of total parameters)
* **Target Modules:** Query and Value projection layers (`q_proj`, `v_proj`)
* **Rank (r):** 32
* **LoRA Alpha:** 64
* **Dropout:** 0.05

## Hardware Utilization
The training lifecycle was explicitly engineered for constrained compute budgets. Utilizing Int8 base quantization allowed the training to execute safely on a budget NVIDIA T4 GPU.
* **GPU VRAM Consumed:** 5.3 GB / 15.0 GB
* **System RAM:** 7.3 GB / 12.7 GB

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
