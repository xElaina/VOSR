"""Tiled VAE (encoder/decoder) inference for the Qwen-Image 2D VAE (and SD2).

The VOSR inference scripts tile only the DiT forward pass; the VAE encode/decode
ran on the full image, which OOMs on large inputs (Qwen VAE is fp32; a 3840x2160
encode needs ~130 GiB). This module adds Gaussian-blended tiled encode/decode so
peak memory is bounded by one tile's activations.

Why Gaussian blending is sufficient (vs. the SD/LDM `VAEHook` pad+crop machinery):
Qwen's VAE uses RMSNorm2D (local, per-pixel/channel) and flat ModuleList blocks —
there are no cross-tile statistics (GroupNorm) to reconcile, so each tile's forward
is independent and a weighted blend is mathematically well-founded.

Design:
  - encode:  reflect-pad LQ to a multiple of 8, split into overlapping pixel tiles
             aligned to the 8x downsample, encode each tile with `latent_dist.mode()`
             (deterministic — `.sample()` draws uncorrelated randn per tile and would
             create seams), Gaussian-blend latent tiles into one full normalized latent.
  - decode:  split the SR latent into overlapping tiles, decode each to a pixel tile,
             Gaussian-blend in pixel space (3ch) -> full SR image. Bounded memory.

`encode_latent`/`decode_latent` replace the `_encode_latent`/`_decode_latent` helpers
duplicated in the inference scripts, adding a deterministic `posterior_mode=True` default.
"""

import math
import torch
import torch.nn.functional as F

AE_FACTOR = 8  # Qwen/SD2 VAE spatial compression ratio


def _gaussian_weights(tile_h, tile_w, channels, device):
    """2-D Gaussian blend mask (1, C, tile_h, tile_w) peaked at the centre."""
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]                       # (tile_h, tile_w)
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def _make_tile_grid(length, tile, overlap):
    """Return sorted, deduplicated starting positions that cover *length*."""
    stride = max(tile - overlap, 1)
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


def _pad_to_multiple(x, multiple=8):
    """Reflect-pad x to a multiple of `multiple`; return (padded, orig_h, orig_w)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def encode_latent(vae, x, args, device, posterior_mode=True):
    """Single-shot VAE encode -> normalized latent. Mirrors the old _encode_latent,
    but defaults to deterministic `latent_dist.mode()` instead of `.sample()`.

    qwen: (z - latents_mean) * (1/latents_std)   (latents_std returned as 1/std)
    sd2 : z * vae.config.scaling_factor
    Returns (latent, latents_mean, latents_std); sd2 -> (latent, None, None).
    """
    if args.ae_type == 'qwen':
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1).to(device)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1).to(device)
        posterior = vae.encode(x).latent_dist
        z = posterior.mode() if posterior_mode else posterior.sample()
        return (z - latents_mean) * latents_std, latents_mean, latents_std
    elif args.ae_type == 'sd2':
        posterior = vae.encode(x).latent_dist
        z = posterior.mode() if posterior_mode else posterior.sample()
        return z * vae.config.scaling_factor, None, None


def decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder=None):
    """Single-shot VAE decode -> pixels in [-1, 1]. Mirrors the old _decode_latent."""
    if args.ae_type == 'sd2':
        sr_u = sr_latent / vae.config.scaling_factor
        return light_decoder(sr_u).clamp(-1, 1)
    elif args.ae_type == 'qwen':
        sr_latent = sr_latent / latents_std + latents_mean
        return vae.decode(sr_latent, return_dict=False)[0].clamp(-1, 1)


def _tile_params(tile_size, tile_overlap, lh, lw):
    """Latent-space tile size/overlap from pixel `tile_size`, capped to the latent dims."""
    lt_size = max(tile_size // AE_FACTOR, 1)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def tiled_encode_latent(vae, x, args, device, tile_size, tile_overlap, posterior_mode=True):
    """Gaussian-blended tiled VAE encode -> full normalized latent.

    Output latent shape equals the untiled path: ceil(h/8) x ceil(w/8), so the
    downstream DiT tile grid (which reads lh/lw from the latent) is unchanged.
    Returns (latent, latents_mean, latents_std).
    """
    pad, orig_h, orig_w = _pad_to_multiple(x, 8)
    lh, lw = pad.shape[2] // 8, pad.shape[3] // 8
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    # Fast path: image fits in one tile -> single-shot encode of the padded image.
    if lh <= lt_size and lw <= lt_size:
        z, mean, std = encode_latent(vae, pad, args, device, posterior_mode)
        return z, mean, std

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
    print(f"[Tiled VAE encode]: pixel {orig_h}x{orig_w} -> latent {lh}x{lw}, "
          f"tile={lt_size}, overlap={lt_overlap}, grid={len(h_pos)}x{len(w_pos)}")

    b = pad.shape[0]
    acc = wacc = g = None
    mean = std = None
    for hi in h_pos:
        for wi in w_pos:
            # Every crop boundary is a multiple of 8 -> the encoder emits exactly lt_size.
            crop = pad[:, :, hi * 8:(hi + lt_size) * 8, wi * 8:(wi + lt_size) * 8]
            z_tile, mean, std = encode_latent(vae, crop, args, device, posterior_mode)
            if acc is None:
                lc = z_tile.shape[1]
                g = _gaussian_weights(lt_size, lt_size, lc, device)
                acc = torch.zeros(b, lc, lh, lw, device=device, dtype=z_tile.dtype)
                wacc = torch.zeros_like(acc)
            acc[:, :, hi:hi + lt_size, wi:wi + lt_size] += z_tile * g
            wacc[:, :, hi:hi + lt_size, wi:wi + lt_size] += g

    blended = acc / wacc
    # Crop back to the untiled latent size (ceil(orig/8)); padded right/bottom dropped.
    out_lh, out_lw = math.ceil(orig_h / 8), math.ceil(orig_w / 8)
    return blended[:, :, :out_lh, :out_lw], mean, std


def tiled_decode_latent(vae, sr_latent, args, latents_mean, latents_std,
                        light_decoder=None, tile_size=1024, tile_overlap=None):
    """Gaussian-blended tiled VAE decode -> full pixel image (B,3,8*lh,8*lw) in [-1,1].

    Decoding is done in pixel space (3ch): each latent tile maps to an exactly-8x
    pixel tile, and overlapping tiles are blended with the peaked Gaussian mask.
    Works for both qwen and sd2 (sd2 needs `light_decoder`).
    """
    b, _, lh, lw = sr_latent.shape
    if tile_overlap is None:
        tile_overlap = tile_size // 8
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder)

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
    print(f"[Tiled VAE decode]: latent {lh}x{lw}, tile={lt_size}, overlap={lt_overlap}, "
          f"grid={len(h_pos)}x{len(w_pos)}")

    out_h, out_w = lh * 8, lw * 8
    g = _gaussian_weights(lt_size * 8, lt_size * 8, 3, sr_latent.device)
    acc = torch.zeros(b, 3, out_h, out_w, device=sr_latent.device)
    wacc = torch.zeros_like(acc)
    for hi in h_pos:
        for wi in w_pos:
            he, we = hi + lt_size, wi + lt_size
            pix = decode_latent(vae, sr_latent[:, :, hi:he, wi:we],
                                args, latents_mean, latents_std, light_decoder)
            acc[:, :, hi * 8:he * 8, wi * 8:we * 8] += pix * g
            wacc[:, :, hi * 8:he * 8, wi * 8:we * 8] += g

    return (acc / wacc).clamp(-1, 1)


def encode_dispatch(vae, x, args, device):
    """encode_latent, but tiled when args.vae_tile_size > 0.

    Reads `args.vae_tile_size`, `args.vae_tile_overlap` (None -> follow tile_overlap),
    and `args.posterior_mode` (default True). Use this in the inference scripts so a
    single --vae_tile_size flag controls VAE tiling.
    """
    posterior_mode = bool(getattr(args, 'posterior_mode', True))
    ts = getattr(args, 'vae_tile_size', 0)
    if ts and ts > 0:
        overlap = getattr(args, 'vae_tile_overlap', None) or getattr(args, 'tile_overlap', ts // 8)
        return tiled_encode_latent(vae, x, args, device, ts, overlap, posterior_mode)
    return encode_latent(vae, x, args, device, posterior_mode)


def decode_dispatch(vae, sr_latent, args, latents_mean, latents_std, light_decoder=None):
    """decode_latent, but tiled when args.vae_tile_size > 0. See encode_dispatch."""
    ts = getattr(args, 'vae_tile_size', 0)
    if ts and ts > 0:
        overlap = getattr(args, 'vae_tile_overlap', None) or getattr(args, 'tile_overlap', ts // 8)
        return tiled_decode_latent(vae, sr_latent, args, latents_mean, latents_std,
                                   light_decoder, tile_size=ts, tile_overlap=overlap)
    return decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder)
