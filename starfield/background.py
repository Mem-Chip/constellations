"""全天背景贴图（银道坐标、等距柱状投影）：未分辨星光 + 尘埃暗带 + 星云 + 球状星团 + 外星系。

沿每个像素的视线积分 光度密度 × e^{-τ}，τ 用与恒星消光相同的三维尘埃场（平滑盘 + 尘埃云，云用解析 erf 累积）。
三个通道分别按 A_R=0.75A_V、A_G=A_V、A_B=1.3A_V 衰减，所以暗带边缘自然发红。
面亮度换算：μ_V = 26.40 − 2.5·log10(Σ [L☉/pc²])。
"""
import math

import numpy as np
from PIL import Image

from .clusters import DSO_KINDS
from .coords import unit
from .dust import erf
from .hashrng import module_rng

J0_LOCAL = 0.05          # 本地 V 波段光度密度 L☉/pc³
MU_ZERO = 26.40
UNRESOLVED_PC = 400.0    # 近于此距离的星光大多已作为单颗星生成
CH_EXT = np.array([0.75, 1.0, 1.3])
COL_OLD = np.array([1.0, 0.86, 0.68])
COL_YOUNG = np.array([0.72, 0.82, 1.0])


def _fractal_noise(rng, W, H, octaves):
    acc = np.zeros((H, W))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw, gh = 2 ** (o + 3), 2 ** (o + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        g = np.concatenate([g, g[:, :1]], axis=1)          # 经度方向首尾相接
        im = Image.fromarray(g, mode="F").resize((W + W // gw, H), Image.BICUBIC)
        acc += amp * np.asarray(im)[:, :W]
        tot += amp
        amp *= 0.55
    return acc / tot


def render_background(cfg, galaxy, dust, dso, W=2048, Hh=1024, nstep=200, log=print):
    g = cfg["galaxy"]
    Rd = float(g["thin_disk"]["scale_length_kpc"]) * 1000
    h_thin = float(g["thin_disk"]["scale_height_pc"])
    tk = g["thick_disk"]
    Rd_t, h_t, f_t = float(tk["scale_length_kpc"]) * 1000, float(tk["scale_height_pc"]), float(tk["local_fraction"])
    bul = g["bulge"]
    b_amp, b_rad = float(bul["brightness"]), float(bul["radius_kpc"]) * 1000
    elliptical = g.get("type", "spiral") != "spiral"

    lon = (np.arange(W) + 0.5) / W * 360.0
    lat = 90.0 - (np.arange(Hh) + 0.5) / Hh * 180.0
    s = np.geomspace(5.0, 25000.0, nstep)
    ds = np.gradient(s)
    out = np.zeros((Hh, W, 3), np.float32)
    rows_per = max(1, 120000 // W)
    z0 = galaxy.z_obs
    k_ext = 1.0 / 1.0857

    for r0 in range(0, Hh, rows_per):
        r1 = min(Hh, r0 + rows_per)
        LON, LAT = np.meshgrid(lon, lat[r0:r1])
        u = unit(LON.ravel(), LAT.ravel())                  # (P,3)
        P = u.shape[0]
        X = u[:, 0:1] * s[None, :]
        Y = u[:, 1:2] * s[None, :]
        Z = u[:, 2:3] * s[None, :] + z0                     # 相对银盘中面
        Rg, _ = galaxy.gc_polar(X, Y)
        if elliptical:
            rr = np.sqrt(Rg ** 2 + Z ** 2)
            j_old = J0_LOCAL * np.exp(-(rr - galaxy.R0) / 3000.0)
            j_young = np.zeros_like(j_old)
        else:
            rad = np.exp(-(Rg - galaxy.R0) / Rd)
            rad = np.where(Rg < 1500, rad * (Rg / 1500.0) ** 2, rad)     # 盘的中心空洞（被核球取代）
            j_thin = J0_LOCAL * (1 - f_t) * rad * np.exp(-np.abs(Z) / h_thin)
            arm = galaxy.arm_factor(X, Y, 2.5) if galaxy.arms_on else 1.0
            j_young = 0.35 * j_thin * arm * np.exp(-np.abs(Z) / 90.0) / np.exp(-np.abs(Z) / h_thin)
            j_old = 0.65 * j_thin + J0_LOCAL * f_t * np.exp(-(Rg - galaxy.R0) / Rd_t) * np.exp(-np.abs(Z) / h_t)
            rb = np.sqrt(Rg ** 2 + (1.7 * Z) ** 2)
            j_old = j_old + b_amp * J0_LOCAL * 60.0 * np.exp(-rb / (b_rad / 3.0))
        # 尘埃：平滑部分逐步累积
        kap = dust.kappa0 * np.exp(-np.abs(Z) / dust.h) * np.exp(-(Rg - galaxy.R0) / dust.Rd)
        if dust.turb_on:
            kap = kap * (1.0 + dust.turb_sample(X, Y, Z))
        A = np.cumsum(kap * ds[None, :], axis=1)
        # 尘埃云：解析累积，只算覆盖到的像素
        for c, sg, pk, cd in zip(dust.c_pos, dust.c_sigma, dust.c_peak, dust.c_dist):
            s0 = u @ c
            p2 = cd * cd - s0 * s0
            m = np.nonzero((p2 < 16 * sg * sg) & (s0 > -4 * sg))[0]
            if m.size == 0:
                continue
            kk = 1.0 / (math.sqrt(2) * sg)
            amp = pk * np.exp(-p2[m] / (2 * sg * sg)) * 0.5
            A[m] += amp[:, None] * (erf((s[None, :] - s0[m, None]) * kk) + erf(s0[m, None] * kk))
        # 只计"分辨不出"的星光：近处的星已经作为单颗星画出
        unres = (1.0 - np.exp(-(s / UNRESOLVED_PC) ** 2))[None, :]
        j_old = j_old * unres
        j_young = j_young * unres
        for ch in range(3):
            att = np.exp(-A * CH_EXT[ch] * k_ext)
            So = np.sum(j_old * att * ds[None, :], axis=1)
            Sy = np.sum(j_young * att * ds[None, :], axis=1)
            out[r0:r1, :, ch] = (So * COL_OLD[ch] + Sy * COL_YOUNG[ch]).reshape(r1 - r0, W)
        if (r0 // rows_per) % 4 == 0:
            log(f"  背景积分 {r1}/{Hh} 行")

    # 细碎结构
    nz = g.get("milky_way_background", {}).get("noise", {}) or {}
    rng = module_rng(cfg["meta"]["seed"], "background")
    if nz.get("strength", 0) > 0:
        n = _fractal_noise(rng, W, Hh, int(nz.get("octaves", 6)))
        out *= (1 + 2 * float(nz["strength"]) * (n - 0.5))[:, :, None].clip(0.05, None)

    boost = float(cfg["events"]["supernovae"].get("remnant_render_boost", 1.0))
    _paint_dso(out, dso, lon, lat, remnant_boost=boost)
    return out


def _paint_dso(out, dso, lon, lat, remnant_boost=1.0):
    Hh, W, _ = out.shape
    pix_deg = 360.0 / W
    colors = {"nebula": np.array([1.0, 0.42, 0.52]), "globular": np.array([1.0, 0.9, 0.75]),
              "galaxy": np.array([0.95, 0.9, 0.85]), "remnant": np.array([1.0, 0.45, 0.5])}
    teal = np.array([0.45, 0.95, 0.9])
    for j in range(dso["kind"].size):
        kind = DSO_KINDS[int(dso["kind"][j])]
        if kind not in colors:
            continue
        V = float(dso["V_int"][j])
        if not np.isfinite(V) or V > 14:
            continue
        sigma = max(float(dso["ang_size_deg"][j]) / 4.0, pix_deg * 0.7)
        if kind == "remnant":
            sigma = max(float(dso["ang_size_deg"][j]) / 2.0, pix_deg)    # 这里当作壳层半径用
        l0, b0 = float(dso["gal_lon"][j]), float(dso["gal_lat"][j])
        sig_as2 = (sigma * 3600) ** 2
        mu0 = V + 2.5 * math.log10(2 * math.pi * sig_as2)
        S0 = 10 ** ((MU_ZERO - mu0) / 2.5)
        rad = 4 * sigma if kind != "remnant" else 1.3 * sigma
        rlo = max(0, int((90 - (b0 + rad)) / 180 * Hh))
        rhi = min(Hh, int((90 - (b0 - rad)) / 180 * Hh) + 1)
        cosb = max(math.cos(math.radians(b0)), 0.05)
        half = min(180.0, rad / cosb)
        cols = (np.arange(int((l0 - half) / pix_deg) - 1, int((l0 + half) / pix_deg) + 2)) % W
        cols = np.unique(cols)
        LON, LAT = np.meshgrid(lon[cols], lat[rlo:rhi])
        uu = unit(LON, LAT)
        c = unit(l0, b0)
        th = np.degrees(np.arccos(np.clip(uu @ c, -1, 1)))
        if kind == "remnant":
            # 边缘增亮的壳 + 沿方位角的丝状起伏；外缘 Hα 红，内部偏 [O III] 青；总流量归一到积分星等
            r0 = sigma
            w = max(0.1 * r0, pix_deg * 0.8)
            pa = np.arctan2(LAT - b0, ((LON - l0 + 180) % 360 - 180) * cosb)
            rr = np.random.default_rng(j)
            ph = rr.uniform(0, 2 * np.pi, 4)
            fil = 0.35 + 0.65 * np.abs(np.sin(3 * pa + ph[0]) * np.sin(7 * pa + ph[1]) + 0.5 * np.sin(13 * pa + ph[2]))
            fil = fil * (0.7 + 0.3 * np.sin(th / r0 * 9 + ph[3]))
            shell = np.exp(-0.5 * ((th - r0) / w) ** 2) * fil
            inner = 0.25 * np.exp(-0.5 * ((th - 0.8 * r0) / (2 * w)) ** 2) * fil
            area = (pix_deg * 3600) ** 2 * np.cos(np.radians(LAT))
            norm = float(np.sum((shell + inner) * area))
            if norm <= 0:
                continue
            F = remnant_boost * 10 ** ((MU_ZERO - V) / 2.5) / norm
            out[np.ix_(np.arange(rlo, rhi), cols)] += (shell[:, :, None] * colors[kind] + inner[:, :, None] * teal) * F
            continue
        prof = S0 * np.exp(-0.5 * (th / sigma) ** 2)
        if kind == "galaxy":     # 拉长一点，像个斜着的盘
            prof = prof * (0.6 + 0.4 * np.cos(np.radians(LAT - b0) * 3) ** 2)
        out[np.ix_(np.arange(rlo, rhi), cols)] += prof[:, :, None] * colors[kind]


def save_background(arr, path_png, path_npy=None, out_size=None):
    """PNG：线性亮度开平方压缩；白点 = 亮度的 99.95 百分位。返回白点对应的 Σ（L☉/pc²）。"""
    if path_npy is not None:
        np.save(path_npy, arr.astype(np.float32))
    white = float(np.percentile(arr[:, :, 1], 99.7))
    img = np.arcsinh(np.clip(arr / white, 0, None) * 8.0) / np.arcsinh(8.0)
    img = np.clip(img, 0, 1)
    im = Image.fromarray((img * 255 + 0.5).astype(np.uint8), mode="RGB")
    if out_size and tuple(out_size) != im.size:
        im = im.resize(tuple(out_size), Image.BICUBIC)
    im.save(path_png, optimize=True)
    return white
