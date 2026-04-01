# train.py
import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import DenoiseNet

TRAIN_SPECS = "data/train/specs"
TEST_SPECS  = "data/test/specs"
CLIP_FRAMES = 128      # ~0.8 seconds of frames
BATCH_SIZE  = 32       # larger batch = faster since data is precomputed
EPOCHS      = 20
LR          = 3e-4
SAVE_PATH   = "models/denoise_net.pth"


class SpecDataset(Dataset):
    def __init__(self, spec_dir):
        self.noisy = sorted(glob.glob(os.path.join(spec_dir, "*_noisy.npy")))
        self.clean = [f.replace("_noisy.npy", "_clean.npy") for f in self.noisy]

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = np.load(self.noisy[idx])   # (F, T)
        clean = np.load(self.clean[idx])   # (F, T)

        # Random time crop
        T = noisy.shape[1]
        if T > CLIP_FRAMES:
            start = np.random.randint(0, T - CLIP_FRAMES)
            noisy = noisy[:, start:start + CLIP_FRAMES]
            clean = clean[:, start:start + CLIP_FRAMES]
        else:
            pad = CLIP_FRAMES - T
            noisy = np.pad(noisy, ((0,0),(0,pad)))
            clean = np.pad(clean, ((0,0),(0,pad)))

        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)  # (1,F,T)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        return noisy, clean


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out  = model(noisy)
            loss = criterion(out, clean)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = DataLoader(SpecDataset(TRAIN_SPECS), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(SpecDataset(TEST_SPECS),  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model     = DenoiseNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                            factor=0.5)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl = run_epoch(model, test_loader,  optimizer, criterion, device, train=False)
        scheduler.step(vl)
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  Train: {tr:.5f}  Val: {vl:.5f}")
        if vl < best:
            best = vl
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved  (val={best:.5f})")

    print(f"\nDone. Best val loss: {best:.5f}")