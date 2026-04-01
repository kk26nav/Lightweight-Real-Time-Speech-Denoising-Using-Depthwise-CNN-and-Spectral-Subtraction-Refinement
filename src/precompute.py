# precompute.py  — run ONCE before training
import os
import numpy as np
import soundfile as sf
import torch

N_FFT      = 512
WIN_LENGTH = 320
HOP_LENGTH = 160
SAMPLE_RATE = 16000
WINDOW = torch.hann_window(WIN_LENGTH)

SPLITS = {
    "train": ("data/train/clean", "data/train/noisy", "data/train/specs"),
    "test":  ("data/test/clean",  "data/test/noisy",  "data/test/specs"),
}

def wav_to_logmag(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.tensor(wav).unsqueeze(0)
    stft = torch.stft(t, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      win_length=WIN_LENGTH, window=WINDOW,
                      return_complex=True)
    return torch.log1p(stft.abs()).squeeze(0).numpy()   # (F, T)

if __name__ == "__main__":
    for split, (clean_dir, noisy_dir, out_dir) in SPLITS.items():
        os.makedirs(out_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".wav")])
        print(f"Processing {split}: {len(files)} files...")
        for i, fname in enumerate(files):
            stem = fname.replace(".wav", "")
            clean_spec = wav_to_logmag(os.path.join(clean_dir, fname))
            noisy_spec = wav_to_logmag(os.path.join(noisy_dir, fname))
            np.save(os.path.join(out_dir, f"{stem}_clean.npy"), clean_spec)
            np.save(os.path.join(out_dir, f"{stem}_noisy.npy"), noisy_spec)
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(files)} done")
        print(f"  {split} complete ✓")
