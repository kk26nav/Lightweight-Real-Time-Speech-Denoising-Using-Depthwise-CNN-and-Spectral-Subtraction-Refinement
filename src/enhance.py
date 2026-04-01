# enhance.py
import os
import torch
import numpy as np
import soundfile as sf
from model import DenoiseNet

SAMPLE_RATE = 16000
N_FFT       = 512
WIN_LENGTH  = 320
HOP_LENGTH  = 160
MODEL_PATH  = "models/denoise_net.pth"


def enhance_file(input_path, output_path, model, device):
    wav, _ = sf.read(input_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    window = torch.hann_window(WIN_LENGTH)
    t      = torch.tensor(wav).unsqueeze(0)

    stft  = torch.stft(t, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH, window=window,
                       return_complex=True)
    mag   = stft.abs()
    phase = torch.angle(stft)

    # log magnitude — SAME as training
    log_mag = torch.log1p(mag)               # (1, F, T)

    model.eval()
    with torch.no_grad():
        inp          = log_mag.unsqueeze(0).to(device)   # (1,1,F,T)
        enhanced_log = model(inp).squeeze(0).cpu()        # (1,F,T)

    # Convert back to linear magnitude
    enhanced_mag = torch.expm1(enhanced_log)              # (1,F,T)

    real     = enhanced_mag * torch.cos(phase)
    imag     = enhanced_mag * torch.sin(phase)
    wav_out  = torch.istft(torch.complex(real, imag),
                           n_fft=N_FFT, hop_length=HOP_LENGTH,
                           win_length=WIN_LENGTH, window=window,
                           length=len(wav))
    wav_out  = wav_out.squeeze(0).numpy()
    # Scale to same RMS as input (not peak-normalize)
    rms_in   = np.sqrt(np.mean(wav**2)) + 1e-8
    rms_out  = np.sqrt(np.mean(wav_out**2)) + 1e-8
    wav_out  = wav_out * (rms_in / rms_out)

    sf.write(output_path, wav_out, SAMPLE_RATE)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DenoiseNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    noisy_dir    = "data/test/noisy"
    enhanced_dir = "data/test/enhanced"
    os.makedirs(enhanced_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".wav")])
    for i, f in enumerate(files):
        enhance_file(os.path.join(noisy_dir, f),
                     os.path.join(enhanced_dir, f), model, device)
        print(f"[{i+1}/{len(files)}] {f}")
