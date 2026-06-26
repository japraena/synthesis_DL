#!/usr/bin/env python3
"""
Demo inference script for the IBG inverse-design CNN (V3 - Chirped gratings).

Usage:
    python demo_inference_v3.py input_spectra.csv

Input CSV format (no header):
    - Each row is one sample.
    - Columns   0–256 : gain at 257 frequency points (dB)
    - Columns 257–513 : phase at 257 frequency points (radians)

Output CSV: predictions_<input_filename>.csv
    Columns: Lambda_Bi_nm, Lambda_Bf_nm, L_periods, DW_max_nm

Required files (same directory as this script):
    V3_Gain_Phase_Chirp.pth   — trained model weights
    scaler_g.pkl              — gain input scaler
    scaler_p.pkl              — phase input scaler
    normalization_stats.npz   — output normalization statistics
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

N_FREQS = 257  # frequency points used during training

# ── Model architecture ─────────────────────────────────────────────────────────
class SpectralToGeometryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),     nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# ── Parse argument ─────────────────────────────────────────────────────────────
if len(sys.argv) != 2:
    print("Usage: python demo_inference_v3.py input_spectra.csv")
    sys.exit(1)

csv_path = sys.argv[1]

# ── Load auxiliary files ───────────────────────────────────────────────────────
scaler_g = joblib.load("scaler_g.pkl")
scaler_p = joblib.load("scaler_p.pkl")

stats  = np.load("normalization_stats.npz")
y_mean = stats["y_mean"]   # shape (4,)
y_std  = stats["y_std"]    # shape (4,)

# ── Load model weights ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SpectralToGeometryCNN().to(device)
model.load_state_dict(torch.load("V3_Gain_Phase_Chirp.pth", map_location=device))
model.eval()

# ── Load and validate input CSV ────────────────────────────────────────────────
data = pd.read_csv(csv_path, header=None).values.astype(np.float32)

assert data.shape[1] == N_FREQS * 2, (
    f"Expected {N_FREQS * 2} columns ({N_FREQS} reflectivity + {N_FREQS} phase), "
    f"got {data.shape[1]}."
)

gain  = data[:, :N_FREQS]   # (n_samples, 257) — dB
phase = data[:, N_FREQS:]   # (n_samples, 257) — radians

# ── Normalise inputs ───────────────────────────────────────────────────────────
gain_scaled  = scaler_g.transform(gain)    # (n_samples, 257)
phase_scaled = scaler_p.transform(phase)   # (n_samples, 257)

x = np.stack([gain_scaled, phase_scaled], axis=1).astype(np.float32)  # (n_samples, 2, N_FREQS)
x_tensor = torch.tensor(x).to(device)

# ── Inference ──────────────────────────────────────────────────────────────────
with torch.no_grad():
    pred_norm = model(x_tensor).cpu().numpy()   # (n_samples, 4) — normalised

# Denormalise to physical units
predictions = pred_norm * y_std + y_mean        # (n_samples, 4)

# ── Display results ────────────────────────────────────────────────────────────
print(f"\n{len(data)} sample(s) processed.\n")
print(f"{'Sample':<8} {'Lambda_Bi (nm)':<16} {'Lambda_Bf (nm)':<16} {'L (periods)':<14} {'DW_max (nm)'}")
print("-" * 68)
for i, p in enumerate(predictions):
    chirp_type = f"chirped ({p[1]-p[0]:+.3f} nm)" if abs(p[1] - p[0]) >= 0.1 else "uniform/apodized"
    print(f"{i+1:<8} {p[0]:<16.3f} {p[1]:<16.3f} {p[2]:<14.1f} {p[3]:.3f}   [{chirp_type}]")

# ── Save results to CSV ────────────────────────────────────────────────────────
out_name = "predictions_" + os.path.basename(csv_path)
df_out = pd.DataFrame(predictions, columns=["Lambda_Bi_nm", "Lambda_Bf_nm", "L_periods", "DW_max_nm"])
df_out.to_csv(out_name, index=False)
print(f"\nResults saved to '{out_name}'")
