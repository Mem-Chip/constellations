"""星团、星协、球状星团、亮星云，以及"深空天体表"(DSO)。

- 疏散星团、OB 星协：成员按初始质量函数抽取、同龄演化，只保留可能看得见的成员，加入恒星表（cluster_id 指向 DSO 表的行）。
  年轻的星团/星协优先落在旋臂里；最年轻的一部分落在尘埃云旁边。
- 球状星团：只作为深空天体（积分亮度 + 角大小），不生成单颗星。
- 亮星云：由最年轻的星团/星协照亮。
- 外星系：来自 landmarks.external_galaxies。
"""
import math

import numpy as np

from . import evolution as E
from . import hashrng as H
from .coords import lonlat, unit
from .population import BINARY_MARGIN_MAG

OC_REGION_PC = 4000.0
OB_REGION_PC = 3000.0
DSO_KINDS = ["open", "ob_assoc", "globular", "nebula", "galaxy", "remnant"]


class IMFSampler:
    def __init__(self, imf):
        me = np.logspace(math.log10(imf.m_lo), math.log10(imf.m_hi), 4001)
        mm = np.sqrt(me[1:] * me[:-1])
        c = np.concatenate([[0.0], np.cumsum(imf(mm) * np.diff(me))])
        self.me, self.c = me, c / c[-1]

    def __call__(self, u, m_lo=None, m_hi=None):
        c0 = 0.0 if m_lo is None else np.interp(m_lo, self.me, self.c)
        c1 = 1.0 if m_hi is None else np.interp(m_hi, self.me, self.c)
        return np.interp(c0 + u * (c1 - c0), self.c, self.me)


def _thin_h(cfg, age_gyr):
    h = float(cfg["galaxy"]["thin_disk"]["scale_height_pc"])
    return h * np.clip(np.sqrt(np.maximum(age_gyr, 1e-6) / 5.0), 0.25, 1.3)


def _place_disk(rng, galaxy, ages, R, cfg, strength=1.0, K=64):
    """给每个年龄为 ages[i] 的群在半径 R 的盘内抽一个位置，按 径向密度 × 旋臂因子 加权（每个群 K 个候选的拒绝采样）。"""
    n = ages.size
    Rd = float(cfg["galaxy"]["thin_disk"]["scale_length_kpc"]) * 1000.0
    r = R * np.sqrt(rng.random((n, K)))
    th = rng.random((n, K)) * 2 * np.pi
    x, y = r * np.cos(th), r * np.sin(th)
    con = 1 + (np.asarray(galaxy.arm_contrast_for_age(ages)) - 1) * strength
    w = galaxy.radial_factor(x, y, Rd) * galaxy.arm_factor(x, y, np.broadcast_to(np.atleast_1d(con)[:, None], (n, K)))
    acc = rng.random((n, K)) < w / w.max()
    pick = np.where(acc.any(axis=1), acc.argmax(axis=1), w.argmax(axis=1))
    rows = np.arange(n)
    z = rng.laplace(0, _thin_h(cfg, ages)) - galaxy.z_obs
    return np.column_stack([x[rows, pick], y[rows, pick], z]), ages


def _members(gkey, n, m_lo, sampler, age_gyr, center, scale, profile, r_t, v_bulk, v_disp, mag_lim):
    """生成一个群的成员；返回 (全部成员积分 M_V, 可见成员字典或 None)。"""
    j = np.arange(n)
    h = H.combine(gkey, j)
    m = sampler(H.uniform(h, 1), m_lo=m_lo)
    ev = E.evolve_any(m, np.full(n, age_gyr))
    F = np.where(ev["MV"] < 90, 10 ** (-0.4 * ev["MV"]), 0.0)
    MV_int = -2.5 * math.log10(max(F.sum(), 1e-30))
    d0 = float(np.linalg.norm(center))
    vis = (ev["phase"] >= 0) & (ev["MV"] + 5 * math.log10(max(d0, 1.0)) - 5 - BINARY_MARGIN_MAG - 0.5 <= mag_lim)
    if not vis.any():
        return MV_int, None
    h = h[vis]
    k = int(vis.sum())
    if profile == "king":
        u_min = (1 + (r_t / scale) ** 2) ** -1.5
        u = u_min + (1 - u_min) * H.uniform(h, 2)
        r = scale * np.sqrt(np.maximum(u ** (-2 / 3) - 1, 0))
    else:   # 高斯
        r = scale * np.sqrt(-2 * np.log(H.uniform(h, 2)))
    cz = 2 * H.uniform(h, 3) - 1
    ph = 2 * np.pi * H.uniform(h, 4)
    sz = np.sqrt(1 - cz * cz)
    p = center + np.column_stack([r * sz * np.cos(ph), r * sz * np.sin(ph), r * cz])
    v = v_bulk + v_disp * np.column_stack([H.normal(h, 5), H.normal(h, 6), H.normal(h, 7)])
    d = np.linalg.norm(p, axis=1)
    out = {
        "id_hi": np.full(k, gkey >> np.uint64(16), dtype=np.uint64), "id_idx": j[vis].astype(np.int32),
        "hash": h, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2], "d": d,
        "mass": m[vis], "age": np.full(k, age_gyr), "phase": ev["phase"][vis].astype(np.int8), "f": ev["f"][vis],
        "L": ev["L"][vis], "Teff": ev["Teff"][vis], "R": ev["R"][vis], "MV": ev["MV"][vis],
        "pop": np.zeros(k, dtype=np.int8), "stratum": np.full(k, -1, dtype=np.int32),
        "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2], "feh": 0.0 * d,
    }
    return MV_int, out


def make_groups(cfg, galaxy, dust, imf, mag_lim, log=print):
    seed = int(cfg["meta"]["seed"])
    rng = H.module_rng(seed, "clusters")
    cc = cfg["clusters"]
    sampler = IMFSampler(imf)
    vd = cfg["stars"]["kinematics"]["velocity_dispersion_kms"]
    dso = {k: [] for k in ["kind", "name", "x", "y", "z", "radius_pc", "MV_int", "V_int", "age_myr", "n_members", "ang_size_deg", "host"]}
    members = []

    def add_dso(kind, name, pos, radius, MV, age_myr, n, V=np.nan, ang=np.nan, host=-1):
        dso["kind"].append(DSO_KINDS.index(kind)); dso["name"].append(name)
        dso["x"].append(pos[0]); dso["y"].append(pos[1]); dso["z"].append(pos[2])
        dso["radius_pc"].append(radius); dso["MV_int"].append(MV); dso["V_int"].append(V)
        dso["age_myr"].append(age_myr); dso["n_members"].append(n); dso["ang_size_deg"].append(ang); dso["host"].append(host)
        return len(dso["kind"]) - 1

    def bulk_velocity(pos, age_gyr):
        sig = vd["young"] + (vd["old"] - vd["young"]) * math.sqrt(min(age_gyr / 10, 1))
        n = rng.standard_normal(3)
        return galaxy.velocities(np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2] + galaxy.z_obs]),
                                 np.array([sig]), n[:1], n[1:2], n[2:])[0]

    young_hosts = []

    # ---- 疏散星团 ----
    oc = cc["open_clusters"]
    vol_kpc3 = math.pi * (OC_REGION_PC / 1000) ** 2 * 0.2
    n_oc = int(rng.poisson(float(oc["density_per_kpc3"]) * vol_kpc3))
    a0, a1 = oc["age_myr"]
    ages = np.exp(rng.uniform(math.log(a0), math.log(a1), n_oc)) / 1000.0
    pos, ages = _place_disk(rng, galaxy, ages, OC_REGION_PC, cfg)
    # 最年轻的一半落到尘埃云附近
    if len(dust.c_pos):
        yi = np.nonzero((ages < 0.02) & (rng.random(n_oc) < 0.5))[0]
        ci = rng.integers(0, len(dust.c_pos), yi.size)
        pos[yi] = dust.c_pos[ci] + rng.normal(0, 1, (yi.size, 3)) * dust.c_sigma[ci, None] * 1.2
    nm0, nm1 = oc["members"]
    rc0, rc1 = oc["core_radius_pc"]
    for i in range(n_oc):
        n = int(math.exp(rng.uniform(math.log(nm0), math.log(nm1))))
        rc = rng.uniform(rc0, rc1)
        rt = min(rc * rng.uniform(5, 10), 20.0)
        vb = bulk_velocity(pos[i], ages[i]) if oc.get("shared_motion", True) else np.zeros(3)
        row = len(dso["kind"])
        gkey = H.seed_key(seed, 7, 0, i)
        MV_int, mem = _members(gkey, n, None, sampler, ages[i], pos[i], rc, oc.get("profile", "king"), rt, vb, 0.5, mag_lim)
        add_dso("open", f"OC{i:04d}", pos[i], min(3 * rc, rt), MV_int, ages[i] * 1000, n)   # 看得出的范围约 3 倍核半径
        if mem is not None:
            mem["cluster_id"] = np.full(mem["x"].size, row, dtype=np.int32)
            members.append(mem)
        if ages[i] * 1000 <= cc["bright_nebulae"]["max_host_age_myr"]:
            young_hosts.append(row)

    # ---- OB 星协 ----
    ob = cc["ob_associations"]
    n_ob = int(rng.poisson(float(ob["count_within_1kpc"]) * (OB_REGION_PC / 1000) ** 2))
    a0, a1 = ob["age_myr"]
    ages = rng.uniform(a0, a1, n_ob) / 1000.0
    pos, ages = _place_disk(rng, galaxy, ages, OB_REGION_PC, cfg, strength=1.5)
    nm0, nm1 = ob["members"]
    r0, r1 = ob["radius_pc"]
    for i in range(n_ob):
        n = int(math.exp(rng.uniform(math.log(nm0), math.log(nm1))))
        rad = rng.uniform(r0, r1)
        vb = bulk_velocity(pos[i], ages[i])
        row = len(dso["kind"])
        gkey = H.seed_key(seed, 7, 1, i)
        MV_int, mem = _members(gkey, n, 2.0, sampler, ages[i], pos[i], rad, "gauss", rad * 3, vb, 3.0, mag_lim)
        add_dso("ob_assoc", f"OB{i:03d}", pos[i], rad, MV_int, ages[i] * 1000, n)
        if mem is not None:
            mem["cluster_id"] = np.full(mem["x"].size, row, dtype=np.int32)
            members.append(mem)
        if ages[i] * 1000 <= cc["bright_nebulae"]["max_host_age_myr"]:
            young_hosts.append(row)

    # ---- 球状星团（星系晕） ----
    n_gc = int(cc["globular_clusters"]["count"])
    r_gc = []
    while len(r_gc) < n_gc:
        r = rng.uniform(0.3, 40.0, 4000)
        w = r ** 2 * (1 + (r / 1.0) ** 2) ** -1.75
        acc = rng.random(4000) < w / 0.4
        r_gc.extend(r[acc].tolist())
    r_gc = np.array(r_gc[:n_gc]) * 1000.0
    dirs = unit(rng.uniform(0, 360, n_gc), np.degrees(np.arcsin(rng.uniform(-1, 1, n_gc))))
    gc_pos = np.array([galaxy.R0, 0.0, -galaxy.z_obs]) + dirs * r_gc[:, None]
    MVg = rng.normal(-7.4, 1.1, n_gc)
    rh = np.exp(rng.normal(math.log(3.0), 0.4, n_gc))
    for i in range(n_gc):
        add_dso("globular", f"GC{i:03d}", gc_pos[i], rh[i], MVg[i], 12000.0, 0)

    # ---- 亮星云：最近的若干年轻宿主 ----
    nb = cc["bright_nebulae"]
    if young_hosts:
        dist = np.array([math.hypot(math.hypot(dso["x"][r], dso["y"][r]), dso["z"][r]) for r in young_hosts])
        order = np.argsort(dist)[: int(nb["count"])]
        for j, oi in enumerate(order):
            r = young_hosts[oi]
            rad = 0.5 * dso["radius_pc"][r] + rng.uniform(5, 25)
            add_dso("nebula", f"NEB{j:02d}", (dso["x"][r], dso["y"][r], dso["z"][r]), rad,
                    dso["MV_int"][r] - 1.0, dso["age_myr"][r], 0, host=r)

    # ---- 外星系 ----
    for j, g in enumerate(cfg["landmarks"].get("external_galaxies", []) or []):
        if g.get("mode", "auto") == "auto":
            l = rng.uniform(0, 360)
            b = rng.choice([-1, 1]) * rng.uniform(25, 65)
        else:
            l, b = g["gal_lon_lat_deg"]
        p = unit(l, b) * 1e6
        add_dso("galaxy", g.get("name", f"GAL{j}"), p, 0.0, np.nan, np.nan, 0,
                V=float(g["apparent_mag"]), ang=float(g["size_deg"]))

    dso = {k: np.array(v) for k, v in dso.items()}
    log(f"  疏散星团 {n_oc}，OB 星协 {n_ob}，球状星团 {n_gc}，亮星云 {int((dso['kind'] == 3).sum())}，外星系 {int((dso['kind'] == 4).sum())}；"
        f"可见成员 {sum(m['x'].size for m in members):,}")
    return members, dso


def finish_dso(dso, dust, mag_lim=None):
    """算深空天体的距离、方向、角大小、消光后的积分星等。"""
    p = np.column_stack([dso["x"], dso["y"], dso["z"]]).astype(np.float64)
    d = np.linalg.norm(p, axis=1)
    u = p / d[:, None]
    gal = dso["kind"] == DSO_KINDS.index("galaxy")
    A = np.zeros(d.size)
    ng = ~gal
    A[ng] = dust.extinction(u[ng], d[ng])
    # 外星系：整条视线穿出银盘
    if gal.any():
        A[gal] = dust.extinction(u[gal], np.full(int(gal.sum()), 20000.0))
    dso["d"] = d
    dso["A_V"] = A
    V = dso["MV_int"] + 5 * np.log10(np.maximum(d, 1)) - 5 + A
    dso["V_int"] = np.where(gal, dso["V_int"] + A, V)
    ang = np.degrees(np.arctan2(dso["radius_pc"], d))
    dso["ang_size_deg"] = np.where(gal, dso["ang_size_deg"], 2 * ang)
    dso["gal_lon"], dso["gal_lat"] = lonlat(u)
    return dso


def add_remnants(dso, rem, dust, mag_lim):
    """把超新星遗迹接到深空天体表末尾（先去掉旧的遗迹，所以可以重复调用）。星团编号不受影响。"""
    ri = DSO_KINDS.index("remnant")
    keep = dso["kind"] != ri
    base = {k: v[keep] for k, v in dso.items()}
    if rem is None or rem["radius_pc"].size == 0:
        return base
    n = rem["radius_pc"].size
    r = {"kind": np.full(n, ri), "name": np.array([""] * n),
         "x": rem["pos"][:, 0], "y": rem["pos"][:, 1], "z": rem["pos"][:, 2],
         "radius_pc": rem["radius_pc"], "MV_int": rem["MV"], "V_int": np.full(n, np.nan),
         "age_myr": rem["age_myr"], "n_members": np.zeros(n, int), "ang_size_deg": np.full(n, np.nan), "host": np.full(n, -1)}
    r = finish_dso(r, dust)
    ok = (r["V_int"] <= mag_lim) | rem.get("force", np.zeros(n, bool))
    order = np.nonzero(ok)[0][np.argsort(r["V_int"][ok])]
    r = {k: v[order] for k, v in r.items()}
    r["name"] = np.array([f"SNR{i:03d}" for i in range(order.size)])
    return {k: np.concatenate([base[k], r[k].astype(base[k].dtype) if base[k].dtype.kind != "U" else r[k]]) for k in base}
