# SpeechPro — Lightweight Real-Time Speech Denoising

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-VoiceBank--DEMAND-purple)
![Params](https://img.shields.io/badge/Parameters-17%2C085-red)

A lightweight real-time speech denoising system using a **Depthwise Separable U-Net CNN** with residual connections. Trained and evaluated on the VoiceBank-DEMAND benchmark dataset. Designed for edge and real-time deployment with only **17,085 parameters**.

---

## Results

| Metric | Noisy Baseline | Enhanced (Proposed) | Improvement |
|--------|---------------|---------------------|-------------|
| PESQ   | 2.19          | **2.44**            | +0.25       |
| STOI   | 0.9175        | 0.9122              | ≈ same      |
| SNR    | 8.83 dB       | **13.80 dB**        | **+4.97 dB** |

> Evaluated on 50 test utterances from the VoiceBank-DEMAND test set (speaker p232).

---

## Model Architecture

```
Noisy WAV → STFT → log1p → DenoiseNet → expm1 → iSTFT → Enhanced WAV
```

**DenoiseNet** is a U-Net style depthwise separable CNN:

```
Input (1, 257, T)
  ├── Encoder Block 1 : 1  → 16  ch  (DW-Sep Conv + BN + ReLU)
  ├── Encoder Block 2 : 16 → 32  ch
  ├── Encoder Block 3 : 32 → 64  ch
  ├── Bottleneck      : 64 → 64  ch
  ├── Decoder Block 1 : 128→ 32  ch  (+ skip from Enc3)
  ├── Decoder Block 2 : 64 → 16  ch  (+ skip from Enc2)
  ├── Decoder Block 3 : 32 → 16  ch  (+ skip from Enc1)
  └── Output Conv 1×1 : 16 →  1  ch  + Residual (x + out)
Output (1, 257, T) — enhanced log-magnitude
```

- **Total parameters:** 17,085
- **Convolution type:** Depthwise Separable (8–9× less compute than standard Conv)
- **Skip connections:** U-Net style — preserves fine-grained spectral features
- **Residual output:** model learns a correction on the noisy input, not from scratch

---

## Dataset

[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)

| Split    | Speakers | Utterances | Noise Types | SNR Levels         |
|----------|----------|------------|-------------|--------------------|
| Training | 28       | 11,572     | 10          | 0, 5, 10, 15 dB    |
| Test     | 2        | 824        | 5 (unseen)  | 2.5, 7.5, 12.5, 17.5 dB |

---

## Project Structure

```
speechpro/
├── data/
│   ├── train/
│   │   ├── clean/          # Clean training WAVs
│   │   ├── noisy/          # Noisy training WAVs
│   │   └── specs/          # Pre-computed .npy spectrograms
│   └── test/
│       ├── clean/          # Clean test WAVs
│       ├── noisy/          # Noisy test WAVs
│       ├── enhanced/       # Model output WAVs (generated)
│       └── specs/          # Pre-computed .npy spectrograms
├── models/
│   └── denoise_net.pth     # Saved model weights
└── src/
    ├── model.py            # DenoiseNet architecture
    ├── precompute.py       # Pre-compute spectrograms (run once)
    ├── train.py            # Training loop
    ├── enhance.py          # Inference — enhance audio files
    ├── evaluate.py         # Compute PESQ, STOI, SNR metrics
    └── main.py             # Entry point
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/speechpro.git
cd speechpro
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchaudio soundfile pesq pystoi numpy
```

### 4. Download VoiceBank-DEMAND dataset
Download from: https://datashare.ed.ac.uk/handle/10283/2791

Place files as:
```
data/train/clean/   ← clean training WAVs
data/train/noisy/   ← noisy training WAVs
data/test/clean/    ← clean test WAVs
data/test/noisy/    ← noisy test WAVs
```

---

## Usage

### Step 1 — Pre-compute spectrograms (run once)
```bash
python src/precompute.py
```
This saves all log-magnitude spectrograms as `.npy` files, making training **5–8× faster**.

### Step 2 — Train
```bash
python src/train.py
```
Training config (edit in `train.py`):

| Parameter    | Value  |
|--------------|--------|
| Epochs       | 60     |
| Batch size   | 32     |
| Learning rate| 3e-4   |
| Optimizer    | Adam   |
| LR Scheduler | ReduceLROnPlateau (patience=5) |
| Loss         | MSE    |

Best model is auto-saved to `models/denoise_net.pth`.

### Step 3 — Enhance audio
```bash
python src/enhance.py
```
Enhanced files are saved to `data/test/enhanced/`.

To enhance a single file, edit these lines in `enhance.py`:
```python
INPUT_FILE  = "data/test/noisy/p232_001.wav"
OUTPUT_FILE = "data/test/enhanced/p232_001_enhanced.wav"
```

### Step 4 — Evaluate
```bash
python src/evaluate.py
```
Outputs a table of average PESQ, STOI, and SNR across the test set.

---

## Comparison with Related Lightweight Models

| Method                     | Params  | PESQ  | SNR Gain |
|----------------------------|---------|-------|----------|
| Noisy Baseline             | —       | 2.19  | 0 dB     |
| NXP tinyML Net (2022)      | 18K     | ~2.30 | ~3 dB    |
| RNNoise (Valin, 2018)      | 60K     | 2.33  | ~3 dB    |
| MiniGAN (Abad et al., 2024)| ~500K   | 2.95  | ~5 dB    |
| IMSE U-Net (2025)          | 427K    | 3.37  | ~7 dB    |
| **Proposed DenoiseNet**    | **17K** | **2.44** | **+4.97 dB** |

> Proposed model achieves best PESQ in the sub-20K parameter category, outperforming RNNoise at 3.5× fewer parameters.

---

## Requirements

```
torch>=2.0
torchaudio>=2.0
soundfile
pesq
pystoi
numpy
```

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU       | NVIDIA GeForce RTX 4060 8GB |
| OS        | Windows 11 |
| Python    | 3.13 |
| CUDA      | 12.x |

---

## Future Work

- [ ] Increase model capacity with more encoder-decoder stages
- [ ] Explore complex-valued STFT to also enhance phase
- [ ] Real-time streaming inference via PyAudio (live microphone)
- [ ] Deploy on Raspberry Pi / ARM board for embedded testing
- [ ] GAN-based training to directly optimize perceptual metrics (PESQ)

---

## Citation

If you use this project in your research or coursework, please cite:

```bibtex
@project{speechpro2026,
  title   = {Lightweight Real-Time Speech Denoising using Depthwise CNN},
  author  = {Naveen Kumaran},
  year    = {2026},
  note    = {Final Year Project, B.E. / B.Tech},
  url     = {https://github.com/yourusername/speechpro}
}
```

---

## References

1. Valentini-Botinhao et al. — *VoiceBank-DEMAND Dataset*, University of Edinburgh, 2017
2. Thiemann et al. — *DEMAND: Diverse Environments Multi-channel Acoustic Noise Database*, 2013
3. Valin J.-M. — *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement*, MMSP 2018 (RNNoise)
4. Fu et al. — *MetricGAN+*, Interspeech 2021
5. Sandler et al. — *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
6. Ronneberger et al. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
7. Rix et al. — *PESQ: Perceptual Evaluation of Speech Quality*, ICASSP 2001
8. Taal et al. — *STOI: An Algorithm for Intelligibility Prediction*, IEEE Trans. ASLP 2011

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
