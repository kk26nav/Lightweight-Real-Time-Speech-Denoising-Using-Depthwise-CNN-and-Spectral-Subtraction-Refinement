# evaluate.py
import os
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from enhance import enhance_file
from model import DenoiseNet

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
MODEL_PATH    = "models/denoise_net.pth"
CLEAN_DIR     = "data/test/clean"
NOISY_DIR     = "data/test/noisy"
ENHANCED_DIR  = "data/test/enhanced"
MAX_FILES     = 50   # evaluate on first 50 files (faster); set None for all


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


def match_length(ref, deg):
    """Trim both signals to the same length."""
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


def compute_snr(clean, enhanced):
    noise = clean - enhanced
    snr   = 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-8))
    return snr


# ── MAIN EVALUATION ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DenoiseNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    os.makedirs(ENHANCED_DIR, exist_ok=True)

    # Get file list
    files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(".wav")])
    if MAX_FILES:
        files = files[:MAX_FILES]
    print(f"Evaluating on {len(files)} files...\n")

    # Metric storage
    metrics = {
        "noisy":    {"pesq": [], "stoi": [], "snr": []},
        "enhanced": {"pesq": [], "stoi": [], "snr": []}
    }

    for i, fname in enumerate(files):
        clean_path    = os.path.join(CLEAN_DIR,    fname)
        noisy_path    = os.path.join(NOISY_DIR,    fname)
        enhanced_path = os.path.join(ENHANCED_DIR, fname)

        # Generate enhanced file if it doesn't exist yet
        if not os.path.exists(enhanced_path):
            enhance_file(noisy_path, enhanced_path, model, device)

        # Load all three
        clean    = load_wav(clean_path)
        noisy    = load_wav(noisy_path)
        enhanced = load_wav(enhanced_path)

        # Match lengths
        clean, noisy    = match_length(clean, noisy)
        clean, enhanced = match_length(clean, enhanced)

        try:
            # ── NOISY metrics (baseline) ──────────────────────────────────────
            metrics["noisy"]["pesq"].append(pesq(SAMPLE_RATE, clean, noisy, "wb"))
            metrics["noisy"]["stoi"].append(stoi(clean, noisy, SAMPLE_RATE, extended=False))
            metrics["noisy"]["snr"].append(compute_snr(clean, noisy))

            # ── ENHANCED metrics (proposed) ───────────────────────────────────
            metrics["enhanced"]["pesq"].append(pesq(SAMPLE_RATE, clean, enhanced, "wb"))
            metrics["enhanced"]["stoi"].append(stoi(clean, enhanced, SAMPLE_RATE, extended=False))
            metrics["enhanced"]["snr"].append(compute_snr(clean, enhanced))

            print(f"[{i+1:02d}/{len(files)}] {fname}  "
                  f"PESQ: {metrics['enhanced']['pesq'][-1]:.3f}  "
                  f"STOI: {metrics['enhanced']['stoi'][-1]:.3f}  "
                  f"SNR: {metrics['enhanced']['snr'][-1]:.2f} dB")

        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    # ── PRINT FINAL RESULTS TABLE ─────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Metric':<12} {'Noisy (baseline)':>18} {'Enhanced (proposed)':>20}")
    print("="*55)

    for metric in ["pesq", "stoi", "snr"]:
        noisy_avg    = np.mean(metrics["noisy"][metric])
        enhanced_avg = np.mean(metrics["enhanced"][metric])
        unit = " dB" if metric == "snr" else ""
        print(f"{metric.upper():<12} {noisy_avg:>17.4f}{unit}  {enhanced_avg:>18.4f}{unit}")

    print("="*55)
    print("Higher is better for all three metrics.")
    print(f"\nEnhanced files saved in: {ENHANCED_DIR}")
