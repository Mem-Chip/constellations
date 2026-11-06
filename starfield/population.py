"""分层采样恒星：族群 × 质量段 × 演化阶段 × 阶段内子段 = 一个"层"。

每层有自己能达到的最亮绝对星等，因此有自己的最远可见距离 d_max，只在 d_max 内生成。
空间被切成格子；每个 (格子, 层) 的星数是泊松数，由哈希决定 → 结果稳定、可按需生成。
星的编号 = (格子, 层) 的哈希 + 格内序号。
"""
import math
import time

import numpy as np

from . import evolution as E
from . import hashrng as H

N_MASS_BINS = 36
FINE = 48
SUB_EDGES = {
    E.MS: [0, 0.5, 0.8, 1.0],
    E.SG: [0, 1.0],
    E.RGB: [0, 0.5, 0.75, 0.88, 0.94, 0.97, 0.985, 0.993, 1.0],   # 每段覆盖差不多的光度对数区间
    E.CHEB: [0, 0.25, 0.29, 0.82, 0.86, 1.0],                     # 对齐蓝回环的转折点
    E.AGB: [0, 0.5, 0.75, 0.88, 0.95, 0.98, 1.0],
}
WD_COOL_EDGES = [0, 0.003, 0.03, 0.3, 1.0, 3.0, 14.0]
BINARY_MARGIN_MAG = 0.8     # 等质量双星合起来亮 0.75 等；d_max 留这点余量
D_CAP_PC = 25000.0
CELL_BASE_PC = 3.125
CELL_MAX_LEVEL = 6          # 3.125·2⁶ = 200 pc
CELL_Z_MAX_PC = 100.0
CELL_REF_MAG = 12.0         # 决定格子大小的参考星等（固定，保证不同生成深度下结果一致）


# ---------------- 初始质量函数与恒星形成史 ----------------
class IMF:
    def __init__(self, cfg):
        i = cfg["stars"]["imf"]
        self.m_lo, self.m_hi = i["mass_range_msun"]
        s = i["slopes"]
        self.a_lo, self.a_hi, self.brk = s["low"], s["high"], s["break_msun"]
        self.a_mass = s.get("massive", self.a_hi)
        self.brk2 = s.get("break2_msun", self.m_hi)
        me = np.logspace(math.log10(self.m_lo), math.log10(self.m_hi), 20001)
        mm = 0.5 * (me[1:] + me[:-1])
        self.total = float(np.sum(self(mm) * np.diff(me)))

    def __call__(self, m):
        m = np.asarray(m, dtype=np.float64)
        mid = (m / self.brk) ** -self.a_hi
        k2 = (self.brk2 / self.brk) ** -self.a_hi
        return np.where(m < self.brk, (m / self.brk) ** -self.a_lo,
               np.where(m < self.brk2, mid, k2 * (m / self.brk2) ** -self.a_mass))


class SFR:
    """年龄分布的累积函数 C(t)：族群里年龄 < t 的星占所有形成恒星的比例。"""

    def __init__(self, span, kind="constant", burst=None):
        a0, a1 = map(float, span)
        t = a0 + (a1 - a0) * np.concatenate([[0.0], np.logspace(-6, 0, 20000)])
        rate = np.ones_like(t)
        if kind == "declining":
            rate = np.exp(t / 4.0)            # 越早形成率越高
        if burst and burst.get("enabled"):
            ab = float(burst["age_myr"]) / 1000.0
            rate = rate * np.where(np.abs(t - ab) < 0.5 * ab, float(burst["strength"]), 1.0)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        self.t = t
        self.Ct = c / c[-1]

    def C(self, t):
        t = np.nan_to_num(np.asarray(t, dtype=np.float64), nan=np.inf, posinf=1e9)
        return np.interp(t, self.t, self.Ct)

    def inv(self, c):
        return np.interp(c, self.Ct, self.t)


class Pop:
    def __init__(self, code, name, frac_local, sfr, Rd_pc, h_fn, sigma_fn, feh_mean, feh_sigma):
        self.code, self.name = code, name
        self.frac_local = frac_local
        self.sfr = sfr
        self.Rd = Rd_pc
        self.h = h_fn
        self.sigma = sigma_fn
        self.feh_mean, self.feh_sigma = feh_mean, feh_sigma
        self.sigma0 = None     # 归一化后的面密度（形成的恒星数 / pc²，在 R0 处）


def make_pops(cfg):
    st = cfg["stars"]
    g = cfg["galaxy"]
    thin_h = float(g["thin_disk"]["scale_height_pc"])
    vd = st["kinematics"]["velocity_dispersion_kms"]
    s_young, s_old = float(vd["young"]), float(vd["old"])
    sfh = st["star_formation_history"]
    thick_frac = float(g["thick_disk"]["local_fraction"])
    met = st["metallicity"]
    thin = Pop(
        0, "thin", 1.0 - thick_frac,
        SFR(sfh["span_gyr"], sfh.get("type", "constant"), sfh.get("recent_burst")),
        float(g["thin_disk"]["scale_length_kpc"]) * 1000.0,
        lambda age: thin_h * np.clip(np.sqrt(np.maximum(age, 1e-6) / 5.0), 0.25, 1.3),
        lambda age: s_young + (s_old - s_young) * np.sqrt(np.clip(age / 10.0, 0.0, 1.0)),
        float(met["mean_feh"]), float(met["sigma"]),
    )
    th_h = float(g["thick_disk"]["scale_height_pc"])
    thick = Pop(
        1, "thick", thick_frac,
        SFR(g["thick_disk"]["age_gyr"]),
        float(g["thick_disk"]["scale_length_kpc"]) * 1000.0,
        lambda age: th_h + 0.0 * np.asarray(age, dtype=np.float64),
        lambda age: 1.25 * s_old + 0.0 * np.asarray(age, dtype=np.float64),
        float(met["mean_feh"]) - 0.5, float(met["sigma"]),
    )
    return [p for p in (thin, thick) if p.frac_local > 0]


def sub_interval(m, phase, sub):
    start, end = E.phase_times(m)
    if phase == E.WD:
        s = start[E.WD]
        return s + WD_COOL_EDGES[sub], s + WD_COOL_EDGES[sub + 1]
    s, e = start[phase], end[phase]
    ed = SUB_EDGES[phase]
    return s + ed[sub] * (e - s), s + ed[sub + 1] * (e - s)


def n_subs(phase):
    return len(WD_COOL_EDGES) - 1 if phase == E.WD else len(SUB_EDGES[phase]) - 1


class Stratum:
    __slots__ = ("pop", "bin", "phase", "sub", "code", "w", "m_edges", "cdf", "mean_age", "h", "arm_c", "M_bright")


def build_strata(cfg, galaxy, pops, imf):
    edges = np.logspace(math.log10(imf.m_lo), math.log10(imf.m_hi), N_MASS_BINS + 1)
    strata = []
    for pop in pops:
        sfr = pop.sfr
        for b in range(N_MASS_BINS):
            me = np.logspace(math.log10(edges[b]), math.log10(edges[b + 1]), FINE + 1)
            mg = np.sqrt(me[1:] * me[:-1])
            imfw = imf(mg) * np.diff(me) / imf.total
            for phase in range(6):
                for sub in range(n_subs(phase)):
                    lo, hi = sub_interval(mg, phase, sub)
                    clo, chi = sfr.C(lo), sfr.C(hi)
                    W = np.where(np.isfinite(lo), chi - clo, 0.0)
                    gw = imfw * W
                    w = float(gw.sum())
                    if w < 1e-13:
                        continue
                    s = Stratum()
                    s.pop, s.bin, s.phase, s.sub = pop, b, phase, sub
                    s.code = pop.code * 100000 + b * 1000 + phase * 20 + sub
                    s.w = w
                    s.m_edges = me
                    s.cdf = np.concatenate([[0.0], np.cumsum(gw)]) / w
                    age_mid = sfr.inv(0.5 * (clo + chi))
                    s.mean_age = float(np.sum(gw * np.nan_to_num(age_mid)) / w)
                    s.h = float(pop.h(s.mean_age))
                    s.arm_c = float(galaxy.arm_contrast_for_age(s.mean_age)) if pop.code == 0 else 1.0
                    # 这一层能达到的最亮绝对星等
                    ok = np.nonzero(W > 0)[0]
                    pick = ok[np.unique(np.linspace(0, ok.size - 1, min(8, ok.size)).round().astype(int))]
                    q = np.linspace(0, 1, 9)
                    mm = np.repeat(mg[pick], q.size)
                    cc = np.repeat(clo[pick], q.size) + np.tile(q, pick.size) * np.repeat(chi[pick] - clo[pick], q.size)
                    ages = sfr.inv(cc)
                    ev = E.evolve(mm, np.full(mm.size, phase), ages)
                    s.M_bright = float(np.min(E.abs_mag_v(ev["L"], ev["Teff"])))
                    strata.append(s)
    # 归一化：观察者处银盘中面上所有恒星（含白矮星、含伴星）的数密度 = local_stellar_density
    # 这里采样的是"系统"（主星），伴星之后再加，所以每个系统按 1 + 平均伴星数 计
    n_local = float(cfg["galaxy"]["local_stellar_density_per_pc3"])
    mc = cfg["multiplicity"]
    tf = float(mc["triple_fraction_of_multiples"])

    def companions(m):
        f = 0.0
        for row in mc["binary_fraction_by_mass"]:
            if row["mass_msun"][0] <= m:
                f = row["fraction"]
        return f * (1 + tf)

    zero = np.zeros(1)
    for pop in pops:
        tot = sum(s.w / (2 * s.h) * np.exp(-abs(galaxy.z_obs) / s.h) * galaxy.arm_factor(zero, zero, s.arm_c)[0]
                  * (1 + companions(math.sqrt(s.m_edges[0] * s.m_edges[-1])))
                  for s in strata if s.pop is pop)
        pop.sigma0 = n_local * pop.frac_local / tot
    return strata


# ---------------- 采样一层 ----------------
def _G(z, h):
    return np.sign(z) * (1.0 - np.exp(-np.abs(z) / h)) * 0.5


def _Ginv(g, h):
    return np.sign(g) * (-h * np.log1p(-2.0 * np.abs(g)))


def sample_stratum(s, galaxy, mag_lim, seed):
    d_max = min(D_CAP_PC, 10 ** ((mag_lim - (s.M_bright - BINARY_MARGIN_MAG) + 5) / 5))
    if d_max < 0.3:
        return None
    # 格子大小只由层本身决定（按固定的参考星等），与本次生成深度无关：
    # 加深/变浅生成时，已有的星原样保留，编号不变
    d_ref = min(D_CAP_PC, 10 ** ((CELL_REF_MAG - (s.M_bright - BINARY_MARGIN_MAG) + 5) / 5))
    k = int(np.clip(math.ceil(math.log2(max(d_ref / 12.0 / CELL_BASE_PC, 1e-9))), 0, CELL_MAX_LEVEL))
    L = CELL_BASE_PC * 2 ** k
    Lz = min(L, CELL_Z_MAX_PC)
    h = s.h
    z_obs = galaxy.z_obs
    zc = 6.0 * h
    zlo, zhi = max(-zc, z_obs - d_max), min(zc, z_obs + d_max)
    if zlo >= zhi:
        return None

    n = math.ceil(d_max / L)
    ix = np.arange(-n, n)
    gx, gy = np.meshgrid(ix, ix, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    x0, y0 = gx * L, gy * L
    dx = np.maximum(0.0, np.maximum(x0, -(x0 + L)))
    dy = np.maximum(0.0, np.maximum(y0, -(y0 + L)))
    col = dx * dx + dy * dy <= d_max * d_max
    gx, gy, x0, y0, dxy2 = gx[col], gy[col], x0[col], y0[col], (dx * dx + dy * dy)[col]
    iz = np.arange(math.floor(zlo / Lz), math.ceil(zhi / Lz))
    nc, nz = gx.size, iz.size
    gx = np.repeat(gx, nz); gy = np.repeat(gy, nz); x0 = np.repeat(x0, nz); y0 = np.repeat(y0, nz)
    dxy2 = np.repeat(dxy2, nz)
    gz = np.tile(iz, nc)
    z0 = gz * Lz
    z1 = z0 + Lz
    dz = np.maximum(0.0, np.maximum(z0 - z_obs, z_obs - z1))
    ok = dxy2 + dz * dz <= d_max * d_max
    gx, gy, gz, x0, y0, z0, z1 = gx[ok], gy[ok], gz[ok], x0[ok], y0[ok], z0[ok], z1[ok]

    xc, yc = x0 + 0.5 * L, y0 + 0.5 * L
    lam = (s.pop.sigma0 * s.w * L * L * (_G(z1, h) - _G(z0, h))
           * galaxy.radial_factor(xc, yc, s.pop.Rd) * galaxy.arm_factor(xc, yc, s.arm_c))

    level_key = H.seed_key(seed, 1, k)
    cs = H.combine(H.combine(H.combine(H.combine(level_key, gx), gy), gz), s.code)
    counts = H.poisson(lam, H.uniform(cs, 0), H.uniform(cs, 1))
    N = int(counts.sum())
    if N == 0:
        return None
    cell = np.repeat(np.arange(cs.size), counts)
    idx = np.arange(N) - np.repeat(np.cumsum(counts) - counts, counts)
    csh = cs[cell]
    sh = H.combine(csh, idx)

    x = x0[cell] + L * H.uniform(sh, 2)
    y = y0[cell] + L * H.uniform(sh, 3)
    ga, gb = _G(z0[cell], h), _G(z1[cell], h)
    z_abs = _Ginv(ga + H.uniform(sh, 4) * (gb - ga), h)
    z = z_abs - z_obs
    d = np.sqrt(x * x + y * y + z * z)
    keep = (d <= d_max) & (d > 1e-3)
    if not keep.any():
        return None
    x, y, z, d, sh, csh, idx = x[keep], y[keep], z[keep], d[keep], sh[keep], csh[keep], idx[keep]

    m = np.interp(H.uniform(sh, 5), s.cdf, s.m_edges)
    lo, hi = sub_interval(m, s.phase, s.sub)
    clo, chi = s.pop.sfr.C(lo), s.pop.sfr.C(hi)
    # 质量格子边缘处，这个具体质量可能在族群年龄范围内根本到不了这一阶段 → 丢弃
    real = (chi - clo > 0) & np.isfinite(lo)
    if not real.all():
        x, y, z, d, sh, csh, idx = x[real], y[real], z[real], d[real], sh[real], csh[real], idx[real]
        m, lo, hi, clo, chi = m[real], lo[real], hi[real], clo[real], chi[real]
        if m.size == 0:
            return {"n_raw": N, "n_cells": cs.size}
    age = np.clip(s.pop.sfr.inv(clo + H.uniform(sh, 6) * (chi - clo)), lo, hi)
    phase = np.full(m.size, s.phase, dtype=np.int8)
    ev = E.evolve(m, phase, age)
    MV = E.abs_mag_v(ev["L"], ev["Teff"])
    vis = MV + 5 * np.log10(d) - 5 - BINARY_MARGIN_MAG <= mag_lim
    if not vis.any():
        return {"n_raw": N, "n_cells": cs.size}
    out = {
        "id_hi": csh[vis] >> np.uint64(16), "id_idx": idx[vis].astype(np.int32), "hash": sh[vis],
        "x": x[vis], "y": y[vis], "z": z[vis], "d": d[vis],
        "mass": m[vis], "age": age[vis], "phase": phase[vis], "f": ev["f"][vis],
        "L": ev["L"][vis], "Teff": ev["Teff"][vis], "R": ev["R"][vis], "MV": MV[vis],
        "pop": np.full(int(vis.sum()), s.pop.code, dtype=np.int8),
        "stratum": np.full(int(vis.sum()), s.code, dtype=np.int32),
    }
    out["n_raw"] = N
    out["n_cells"] = cs.size
    return out


def generate_stars(cfg, galaxy, mag_lim, log=print):
    seed = int(cfg["meta"]["seed"])
    imf = IMF(cfg)
    pops = make_pops(cfg)
    t0 = time.time()
    strata = build_strata(cfg, galaxy, pops, imf)
    log(f"  层数 {len(strata)}，建层用时 {time.time() - t0:.1f}s；"
        + "，".join(f"{p.name} 面密度 {p.sigma0:.1f}/pc²" for p in pops))
    parts = []
    n_raw = n_cells = 0
    t0 = time.time()
    for i, s in enumerate(strata):
        r = sample_stratum(s, galaxy, mag_lim, seed)
        if r is not None:
            n_raw += r.pop("n_raw")
            n_cells += r.pop("n_cells")
            if r:
                parts.append(r)
        if (i + 1) % 200 == 0:
            log(f"  已采样 {i + 1}/{len(strata)} 层，格子 {n_cells:,}，原始星 {n_raw:,}，用时 {time.time() - t0:.1f}s")
    keys = parts[0].keys()
    stars = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    log(f"  采样完成：格子 {n_cells:,}，原始星 {n_raw:,}，预筛后 {stars['x'].size:,}，用时 {time.time() - t0:.1f}s")

    # 速度与金属丰度
    pop_by_code = {p.code: p for p in pops}
    sigma = np.empty(stars["x"].size)
    feh = np.empty(stars["x"].size)
    sh = stars["hash"]
    for code, p in pop_by_code.items():
        k = stars["pop"] == code
        sigma[k] = p.sigma(stars["age"][k])
        feh[k] = p.feh_mean + p.feh_sigma * H.normal(sh[k], 13)
    z_abs = stars["z"] + galaxy.z_obs
    v = galaxy.velocities(stars["x"], stars["y"], z_abs, sigma, H.normal(sh, 10), H.normal(sh, 11), H.normal(sh, 12))
    stars["vx"], stars["vy"], stars["vz"] = v[:, 0], v[:, 1], v[:, 2]
    stars["feh"] = feh
    return stars, strata
