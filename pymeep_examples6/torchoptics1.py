import math
import torch
import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, PhaseModulator

import numpy as np
import matplotlib.pyplot as plt

shape = 512
spacing = 10e-6
wavelength = 700e-9
f = 200e-3

# ★重要：DOE(PhaseModulator)とレンズを同一面(z=0)に置く
z_lens = 0.0
# ★フーリエ面＝レンズ焦点面（レンズから距離 f）
z_fourier = f

iters = 800
lr = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"

torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# target (square in Fourier plane)
fill_ratio = 0.25
side = int(fill_ratio * shape)
I_tgt = torch.zeros((shape, shape), device=device)
c = shape // 2
h = side // 2
I_tgt[c-h:c+h, c-h:c+h] = 1.0
I_tgt = I_tgt / (I_tgt.mean() + 1e-12)

# learnable phase parameter
phi_raw = torch.nn.Parameter(torch.zeros((shape, shape), device=device))

def wrapped_phase(p):
    return 2 * math.pi * torch.sigmoid(p)

lens = Lens(shape, f, z=z_lens).to(device)

# input plane wave at the DOE/lens plane
input_field = Field(torch.ones((shape, shape), device=device)).to(device)
opt = torch.optim.Adam([phi_raw], lr=lr)

def intensity(field):
    if hasattr(field, "intensity"):
        return field.intensity()
    return field.abs() ** 2

def corr_loss(I, T):
    I0 = I - I.mean()
    T0 = T - T.mean()
    corr = (I0*T0).mean() / (I0.std()+1e-12) / (T0.std()+1e-12)
    return 1 - corr

def plot_intensity_with_linecuts(I_torch, title="", log10=True, eps=1e-6,
                                 clip_percentiles=(1, 99.5), normalize="max"):
    I = I_torch.detach().float().cpu().numpy()

    if normalize == "max":
        I = I / (I.max() + 1e-12)
    elif normalize == "mean":
        I = I / (I.mean() + 1e-12)

    H, W = I.shape
    cy, cx = H // 2, W // 2

    if log10:
        I_show = np.log10(I + eps)
        vmin, vmax = None, None
        cbar_label = "log10(I + eps)"
    else:
        vmin, vmax = np.percentile(I, clip_percentiles)
        I_show = I
        cbar_label = "I (clipped)"

    cut_x = I[cy, :]
    cut_y = I[:, cx]

    fig = plt.figure(figsize=(12, 4))

    ax0 = fig.add_subplot(1, 3, 1)
    im = ax0.imshow(I_show, origin="lower", vmin=vmin, vmax=vmax)
    ax0.set_title(title if title else "Intensity")
    ax0.set_xlabel("x (pixel)")
    ax0.set_ylabel("y (pixel)")
    cb = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(cut_x)
    ax1.set_title("Center line cut (y=center)")
    ax1.set_xlabel("x (pixel)")
    ax1.set_ylabel("I")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1.0)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(cut_y)
    ax2.set_title("Center line cut (x=center)")
    ax2.set_xlabel("y (pixel)")
    ax2.set_ylabel("I")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 1.0)

    plt.tight_layout()
    plt.show()

for it in range(1, iters + 1):
    phi = wrapped_phase(phi_raw)

    phase_mod = PhaseModulator(phi).to(device)
    system = System(phase_mod, lens).to(device)

    # ★フーリエ面は z=f
    out_field = system.measure_at_z(input_field, z=z_fourier)

    I = intensity(out_field)
    I = I / (I.mean() + 1e-12)

    loss = torch.mean((I - I_tgt) ** 2) + 0.5 * corr_loss(I, I_tgt)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if it % 50 == 0 or it == 1:
        print(f"iter {it:4d} | loss {loss.item():.6e}")

with torch.no_grad():
    phi = wrapped_phase(phi_raw)
    phase_mod = PhaseModulator(phi).to(device)
    system = System(phase_mod, lens).to(device)

    out_field = system.measure_at_z(input_field, z=z_fourier)
    out_field.visualize(title="Achieved intensity @ Fourier plane (z=f)", vmax=1)

    plt.figure()
    plt.imshow(phi.detach().cpu().numpy(), origin="lower")
    plt.title("Optimized phase [rad] (0..2π)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    I = intensity(out_field)
    plot_intensity_with_linecuts(
        I, title="Fourier plane intensity (z=f)",
        log10=True, eps=1e-6, normalize="max"
    )
