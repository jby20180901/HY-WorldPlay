import argparse
import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLHunyuanVideo15
from skimage.metrics import structural_similarity as ssim


def make_haar_kernels(num_channels: int, device, dtype) -> torch.Tensor:
    ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device, dtype=dtype) / 2.0
    lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device=device, dtype=dtype) / 2.0
    hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=device, dtype=dtype) / 2.0
    hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=device, dtype=dtype) / 2.0
    kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
    return kernel.repeat(num_channels, 1, 1, 1)


def haar_dwt2_spatial(z: torch.Tensor):
    assert z.ndim == 5, f"Expected [B,C,T,H,W], got {z.shape}"
    batch, channels, frames, height, width = z.shape

    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        z = F.pad(z, (0, pad_w, 0, pad_h))

    h2, w2 = z.shape[-2] // 2, z.shape[-1] // 2
    z_bt = z.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, z.shape[-2], z.shape[-1])

    kernels = make_haar_kernels(channels, z.device, z.dtype)
    coeffs = F.conv2d(z_bt, kernels, stride=2, groups=channels)
    coeffs = coeffs.view(batch, frames, channels, 4, h2, w2).permute(0, 2, 1, 3, 4, 5).contiguous()

    ll = coeffs[:, :, :, 0]
    lh = coeffs[:, :, :, 1]
    hl = coeffs[:, :, :, 2]
    hh = coeffs[:, :, :, 3]
    return (ll, lh, hl, hh), (height, width)


def haar_idwt2_spatial(coeffs, original_hw: Tuple[int, int]) -> torch.Tensor:
    ll, lh, hl, hh = coeffs
    batch, channels, frames, h2, w2 = ll.shape

    packed = torch.stack([ll, lh, hl, hh], dim=3)
    packed = packed.permute(0, 2, 1, 3, 4, 5).reshape(batch * frames, 4 * channels, h2, w2)

    kernels = make_haar_kernels(channels, packed.device, packed.dtype)
    recon = F.conv_transpose2d(packed, kernels, stride=2, groups=channels)
    recon = recon.view(batch, frames, channels, recon.shape[-2], recon.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()

    height, width = original_hw
    return recon[..., :height, :width]


def nearest_multiple_of_16(value: int) -> int:
    return max(16, int(round(value / 16.0)) * 16)


def read_video_to_tensor(video_path: str, num_frames: int = 17, target_height: int = 480) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in {video_path}")

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    sampled = [frames[i] for i in indices]

    h0, w0 = sampled[0].shape[:2]
    scale = target_height / float(h0)
    target_width = nearest_multiple_of_16(int(round(w0 * scale)))
    target_height = nearest_multiple_of_16(target_height)

    resized = [
        cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        for frame in sampled
    ]

    arr = np.stack(resized, axis=0).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (3, 0, 1, 2))
    return torch.from_numpy(arr).unsqueeze(0)


def latent_rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.norm(a - b) / (torch.norm(a) + 1e-12)).item()


def to_uint8_video(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(-1, 1).cpu()
    x = (x[0].permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


def video_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def video_ssim(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for i in range(a.shape[0]):
        vals.append(ssim(a[i], b[i], channel_axis=2, data_range=255))
    return float(np.mean(vals))


def gaussian_video(x: np.ndarray, ksize: int = 11, sigma: float = 2.0) -> np.ndarray:
    out = []
    for i in range(x.shape[0]):
        out.append(cv2.GaussianBlur(x[i], (ksize, ksize), sigma))
    return np.stack(out, axis=0)


def laplacian_video(x: np.ndarray) -> np.ndarray:
    out = []
    for i in range(x.shape[0]):
        chans = []
        for c in range(3):
            chans.append(cv2.Laplacian(x[i][..., c].astype(np.float32), cv2.CV_32F, ksize=3))
        out.append(np.stack(chans, axis=-1))
    return np.stack(out, axis=0)


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def total_variation(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    tv_h = np.mean(np.abs(x[:, 1:, :, :] - x[:, :-1, :, :]))
    tv_w = np.mean(np.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    return float(tv_h + tv_w)


def save_video_mp4(path: str, video_uint8: np.ndarray, fps: int = 8):
    frames, height, width, _ = video_uint8.shape
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for i in range(frames):
        frame_bgr = cv2.cvtColor(video_uint8[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


@torch.no_grad()
def decode_latent(vae, z_scaled: torch.Tensor) -> torch.Tensor:
    z_unscaled = z_scaled / vae.config.scaling_factor
    return vae.decode(z_unscaled).sample


def summarize_meaningfulness(metrics: Dict[str, float]) -> Dict[str, bool]:
    verdicts = {}
    verdicts["invertible_latent"] = metrics["latent_rel_l2"] < 1e-5
    verdicts["invertible_decoded"] = metrics["decoded_psnr_full_recon"] > 50.0
    verdicts["low_band_keeps_structure"] = metrics["blur_mse_ll"] < metrics["blur_mse_hf"]
    verdicts["high_band_is_more_edge_like"] = metrics["lap_corr_hf"] > metrics["lap_corr_ll"]
    verdicts["hf_has_higher_tv"] = metrics["tv_hf"] > metrics["tv_ll"]
    verdicts["overall_meaningful"] = all(verdicts.values())
    return verdicts


def parse_args():
    parser = argparse.ArgumentParser(description="Probe wavelet separability on HunyuanVideo-1.5 VAE latent for HY-WorldPlay")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HunyuanVideo-1.5 base model directory or HF model id.",
    )
    parser.add_argument("--input_video", type=str, required=True, help="Input short mp4 clip path.")
    parser.add_argument("--num_frames", type=int, default=17, help="Number of sampled frames for probing.")
    parser.add_argument("--target_height", type=int, default=480, help="Target video height (aligned to multiple of 16).")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device, e.g. cuda or cpu.")
    parser.add_argument("--outdir", type=str, default="./wavelet_probe_out", help="Output directory for results.")
    parser.add_argument("--fps", type=int, default=8, help="Saved visualization video FPS.")
    parser.add_argument(
        "--disable_tiling",
        action="store_true",
        help="Disable VAE tiling decode/encode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    if not args.disable_tiling:
        vae.enable_tiling()

    video = read_video_to_tensor(args.input_video, args.num_frames, args.target_height).to(device=device, dtype=torch.float32)

    posterior = vae.encode(video).latent_dist
    z = posterior.mode() * vae.config.scaling_factor

    (ll, lh, hl, hh), original_hw = haar_dwt2_spatial(z)

    z_recon = haar_idwt2_spatial((ll, lh, hl, hh), original_hw)
    zero = torch.zeros_like(ll)
    z_ll = haar_idwt2_spatial((ll, zero, zero, zero), original_hw)
    z_hf = haar_idwt2_spatial((zero, lh, hl, hh), original_hw)

    ref_video = decode_latent(vae, z)
    recon_video = decode_latent(vae, z_recon)
    ll_video = decode_latent(vae, z_ll)
    hf_video = decode_latent(vae, z_hf)

    ref_u8 = to_uint8_video(ref_video)
    recon_u8 = to_uint8_video(recon_video)
    ll_u8 = to_uint8_video(ll_video)
    hf_u8 = to_uint8_video(hf_video)

    total_energy = ((ll ** 2).sum() + (lh ** 2).sum() + (hl ** 2).sum() + (hh ** 2).sum()).item()

    blur_ref = gaussian_video(ref_u8)
    lap_ref = laplacian_video(ref_u8)
    lap_ll = laplacian_video(ll_u8)
    lap_hf = laplacian_video(hf_u8)

    metrics = {
        "latent_rel_l2": latent_rel_l2(z, z_recon),
        "decoded_psnr_full_recon": video_psnr(ref_u8, recon_u8),
        "decoded_ssim_full_recon": video_ssim(ref_u8, recon_u8),
        "energy_ratio_ll": float((ll ** 2).sum().item() / (total_energy + 1e-12)),
        "energy_ratio_lh": float((lh ** 2).sum().item() / (total_energy + 1e-12)),
        "energy_ratio_hl": float((hl ** 2).sum().item() / (total_energy + 1e-12)),
        "energy_ratio_hh": float((hh ** 2).sum().item() / (total_energy + 1e-12)),
        "blur_mse_ll": mse_np(blur_ref, ll_u8),
        "blur_mse_hf": mse_np(blur_ref, hf_u8),
        "lap_corr_ll": corr_np(lap_ref, lap_ll),
        "lap_corr_hf": corr_np(lap_ref, lap_hf),
        "tv_ll": total_variation(ll_u8),
        "tv_hf": total_variation(hf_u8),
    }

    verdicts = summarize_meaningfulness(metrics)

    save_video_mp4(os.path.join(args.outdir, "decoded_reference.mp4"), ref_u8, fps=args.fps)
    save_video_mp4(os.path.join(args.outdir, "decoded_full_recon.mp4"), recon_u8, fps=args.fps)
    save_video_mp4(os.path.join(args.outdir, "decoded_ll_only.mp4"), ll_u8, fps=args.fps)
    save_video_mp4(os.path.join(args.outdir, "decoded_hf_only.mp4"), hf_u8, fps=args.fps)

    payload = {
        "config": {
            "model_path": args.model_path,
            "input_video": args.input_video,
            "num_frames": args.num_frames,
            "target_height": args.target_height,
            "device": str(device),
            "fps": args.fps,
        },
        "metrics": metrics,
        "verdicts": verdicts,
    }

    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Quantitative metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n=== Verdict ===")
    print(json.dumps(verdicts, indent=2, ensure_ascii=False))

    print("\nSaved files:")
    print(f"  {os.path.join(args.outdir, 'decoded_reference.mp4')}")
    print(f"  {os.path.join(args.outdir, 'decoded_full_recon.mp4')}")
    print(f"  {os.path.join(args.outdir, 'decoded_ll_only.mp4')}")
    print(f"  {os.path.join(args.outdir, 'decoded_hf_only.mp4')}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
