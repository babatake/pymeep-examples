import math
import torch
import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, PhaseModulator

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 0) parameters
# ----------------------------
shape = 512
spacing = 10e-6
wavelength = 700e-9
f = 200e-3

# ★重要：レンズを z=0、フーリエ面を z=f にする
z_lens = 0.0
z_fourier = f

iters = 2000
lr = 0.03

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# ----------------------------
# 1) target (smiley intensity in Fourier plane) 反転なし
# ----------------------------
def make_smiley_target(shape, device="cpu",
                       face_radius_ratio=0.40,
                       eye_offset_ratio=(0.16, 0.14),   # dyを上げると目が上へ
                       eye_radius_ratio=0.035,
                       mouth_radius_ratio=0.23,
                       mouth_thickness_ratio=0.020,
                       mouth_y_offset_ratio=-0.06,      # 負で下
                       mouth_angle_deg=(210, 330),
                       blur_sigma_px=2.0):

    H = W = shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    x = (xx - cx)
    y = (yy - cy)  # ★反転しない

    r = torch.sqrt(x**2 + y**2)

    face_R = face_radius_ratio * shape
    face = (r <= face_R).float()

    eye_R = eye_radius_ratio * shape
    dx = eye_offset_ratio[0] * shape
    dy = eye_offset_ratio[1] * shape

    # ★dy>0で「上」に移動（origin="lower" 表示と整合）
    eye1 = (torch.sqrt((x - dx)**2 + (y - dy)**2) <= eye_R).float()
    eye2 = (torch.sqrt((x + dx)**2 + (y - dy)**2) <= eye_R).float()

    my = mouth_y_offset_ratio * shape
    mouth_R = mouth_radius_ratio * shape
    mouth_t = mouth_thickness_ratio * shape

    rr = torch.sqrt(x**2 + (y - my)**2)
    ring = ((rr >= (mouth_R - mouth_t/2)) & (rr <= (mouth_R + mouth_t/2))).float()

    ang = torch.atan2((y - my), x)
    ang_deg = (ang * 180.0 / math.pi) % 360.0

    a0, a1 = mouth_angle_deg
    if a0 <= a1:
        arc = ((ang_deg >= a0) & (ang_deg <= a1)).float()
    else:
        arc = ((ang_deg >= a0) | (ang_deg <= a1)).float()

    mouth = ring * arc

    I = 0.25 * face + 1.0 * (eye1 + eye2) + 0.9 * mouth

    # blur（これが無いと難易度が跳ねる）
    if blur_sigma_px and blur_sigma_px > 0:
        sigma = float(blur_sigma_px)
        k = int(6*sigma + 1) | 1
        ax = torch.arange(k, device=device) - k//2
        g = torch.exp(-(ax**2) / (2*sigma**2))
        g = g / g.sum()

        I_ = I[None, None, :, :]
        g1 = g[None, None, :, None]
        g2 = g[None, None, None, :]
        I_ = torch.nn.functional.conv2d(I_, g2, padding=(0, k//2))
        I_ = torch.nn.functional.conv2d(I_, g1, padding=(k//2, 0))
        I = I_[0, 0]

    I = I / (I.mean() + 1e-12)
    return I

I_tgt = make_smiley_target(shape, device=device)

plt.figure(figsize=(4,4))
plt.imshow(I_tgt.detach().cpu().numpy(), origin="lower")
plt.title("Target intensity: Smiley")
plt.colorbar()
plt.tight_layout()
plt.show()

# ----------------------------
# 2) learnable amplitude + phase (SLM at z=0 = lens plane)
# ----------------------------
amp_raw = torch.nn.Parameter(torch.zeros((shape, shape), device=device))
phi_raw = torch.nn.Parameter(torch.zeros((shape, shape), device=device))

def wrapped_phase(p):
    return 2 * math.pi * torch.sigmoid(p)

def bounded_amp(a):
    # 0..1
    return torch.sigmoid(a)

# optics (lens at z=0)
lens = Lens(shape, f, z=z_lens).to(device)

opt = torch.optim.Adam([amp_raw, phi_raw], lr=lr)

# ----------------------------
# 3) helpers
# ----------------------------
def intensity(field):
    if hasattr(field, "intensity"):
        return field.intensity()
    return field.abs() ** 2

def corr_loss(I, T):
    I0 = I - I.mean()
    T0 = T - T.mean()
    corr = (I0 * T0).mean() / (I0.std() + 1e-12) / (T0.std() + 1e-12)
    return 1 - corr

def tv_loss(x):
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]
    return (dx.abs().mean() + dy.abs().mean())

# 強度そのものより sqrt(I) の方が見た目に効きやすいことが多い
def mse_on_sqrtI(I, T, eps=1e-8):
    return torch.mean((torch.sqrt(I + eps) - torch.sqrt(T + eps))**2)

# regularization weights
w_corr = 0.2
w_tv   = 3e-2
w_l2a  = 1e-3

# ----------------------------
# 4) optimization loop
# ----------------------------
for it in range(1, iters + 1):
    amp = bounded_amp(amp_raw)
    phi = wrapped_phase(phi_raw)

    # 入力は平面波 * 振幅
    input_field = Field(torch.ones((shape, shape), device=device)).to(device)

    # SLM：位相はPhaseModulator、振幅はFieldに掛ける
    # （Fieldの data に掛ける）
    input_field = Field(amp * input_field.data).to(device)

    phase_mod = PhaseModulator(phi).to(device)
    system = System(phase_mod, lens).to(device)

    out_field = system.measure_at_z(input_field, z=z_fourier)
    I = intensity(out_field)
    I = I / (I.mean() + 1e-12)

    # loss（sqrt強度MSE + corr + 正則化）
    l_mse  = mse_on_sqrtI(I, I_tgt)
    l_corr = corr_loss(I, I_tgt)
    l_tv   = tv_loss(amp)
    l_l2a  = ((amp - 0.5)**2).mean()

    loss = l_mse + w_corr * l_corr + w_tv * l_tv + w_l2a * l_l2a

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if it % 100 == 0 or it == 1:
        print(f"iter {it:4d} | loss {loss.item():.3e} | mse {l_mse.item():.3e} | corr {l_corr.item():.3e} | tv {l_tv.item():.3e}")

# ----------------------------
# 5) visualize
# ----------------------------
with torch.no_grad():
    amp = bounded_amp(amp_raw)
    phi = wrapped_phase(phi_raw)

    input_field = Field(torch.ones((shape, shape), device=device)).to(device)
    input_field = Field(amp * input_field.data).to(device)

    phase_mod = PhaseModulator(phi).to(device)
    system = System(phase_mod, lens).to(device)

    out_field = system.measure_at_z(input_field, z=z_fourier)
    out_field.visualize(title="Achieved intensity @ Fourier plane (z=f)", vmax=1)

    I = intensity(out_field)
    I = I / (I.mean() + 1e-12)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(amp.detach().cpu().numpy(), origin="lower")
    plt.title("Optimized amplitude (0..1)")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(phi.detach().cpu().numpy(), origin="lower")
    plt.title("Optimized phase [rad] (0..2π)")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(I_tgt.detach().cpu().numpy(), origin="lower")
    plt.title("Target (Smiley)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # 見た目比較用（sqrt強度）
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(torch.sqrt(I + 1e-8).detach().cpu().numpy(), origin="lower")
    plt.title("sqrt(Achieved I)")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(torch.sqrt(I_tgt + 1e-8).detach().cpu().numpy(), origin="lower")
    plt.title("sqrt(Target I)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
