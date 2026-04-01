import librosa
import numpy as np

clean, sr = librosa.load("data/train/clean/p226_001.wav", sr=16000)
noisy, sr = librosa.load("data/train/noisy/p226_001.wav", sr=16000)

print(f"Sample rate: {sr}")
print(f"Clean shape: {clean.shape}, Noisy shape: {noisy.shape}")

# Quick STFT test
import torch
import torchaudio
window = torch.hann_window(320)
waveform = torch.tensor(noisy).unsqueeze(0)
stft = torch.stft(waveform, n_fft=512, hop_length=160,
                  win_length=320, window=window, return_complex=True)
magnitude = stft.abs()
print(f"STFT magnitude shape: {magnitude.shape}")  # should be (1, 257, T)
