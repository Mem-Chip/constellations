"""地标：自动生成之后检查约束，不满足就补一颗合适的星；也支持手动放置。

补的星和普通星走同一条流水线（演化 → 变星 → 伴星 → 消光），编号以 seed_key(seed, 9, k) 派生。
"""
import math

import numpy as np

from . import evolution as E
from . import hashrng as H
from .coords import unit
from .multiplicity import add_companions
from .observe import add_extinction
from .variables import assign_eclipsing, assign_pulsators

_PHASE = {n: i for i, n in enumerate(E.PHASE_NAMES)}


def concat(parts):
    parts = [p for p in parts if p is not None and len(p) and np.size(next(iter(p.values())))]
    keys = []
    for p in parts:
        for k in p:
            if k not in keys and isinstance(p[k], np.ndarray):
                keys.append(k)
    out = {}
    for k in keys:
        ref = next(p[k] for p in parts if k in p)
        arrs = []
        for p in parts:
            n = next(iter(p.values())).shape[0]
            if k in p:
                arrs.append(p[k])
            else:
                fill = np.nan if ref.dtype.kind == "f" else (-1 if ref.dtype.kind == "i" else 0)
                arrs.append(np.full(n, fill, dtype=ref.dtype))
        out[k] = np.concatenate(arrs)
    return out


def _solve_distance(MV, u, V_target, dust):
    d = 10 ** ((V_target - MV + 5) / 5)
    for _ in range(4):
        A = dust.extinction(u[None, :], np.array([d]))[0]
        d = 10 ** ((V_target - MV - A + 5) / 5)
    return d


def make_star(cfg, galaxy, dust, k, mass, phase, age, l, b, d=None, V_target=None, name=""):
    seed = int(cfg["meta"]["seed"])
    gkey = H.seed_key(seed, 9, k)
    ev = E.evolve(np.array([mass]), np.array([phase]), np.array([age]))
    MV = E.abs_mag_v(ev["L"], ev["Teff"])
    u = unit(l, b)
    if d is None:
        d = _solve_distance(float(MV[0]), u, V_target, dust)
    p = u * d
    s = {
        "id_hi": np.atleast_1d(gkey >> np.uint64(16)).astype(np.uint64), "id_idx": np.zeros(1, np.int32),
        "hash": np.atleast_1d(H.combine(gkey, 0)),
        "x": p[None, 0], "y": p[None, 1], "z": p[None, 2], "d": np.array([d]),
        "mass": np.array([mass]), "age": np.array([age]), "phase": np.array([phase], np.int8), "f": ev["f"],
        "L": ev["L"], "Teff": ev["Teff"], "R": ev["R"], "MV": MV,
        "pop": np.zeros(1, np.int8), "stratum": np.full(1, -2, np.int32), "cluster_id": np.full(1, -1, np.int32),
        "feh": np.zeros(1), "landmark": np.ones(1, np.int8),
    }
    sig = np.array([10.0])
    v = galaxy.velocities(s["x"], s["y"], s["z"] + galaxy.z_obs, sig, H.normal(s["hash"], 10), H.normal(s["hash"], 11), H.normal(s["hash"], 12))
    s["vx"], s["vy"], s["vz"] = v[:, 0], v[:, 1], v[:, 2]
    assign_pulsators(s, cfg)
    add_companions(s, cfg)
    assign_eclipsing(s)
    add_extinction(s, dust)
    s["_name"] = name
    return s


def _phase_mid_age(mass, phase, frac=0.5):
    start, end = E.phase_times(np.array([mass]))
    return float(start[phase, 0] + frac * (end[phase, 0] - start[phase, 0]))


def apply_landmarks(stars, dso, cfg, galaxy, dust, log=print):
    lm = cfg["landmarks"]
    rng = H.module_rng(cfg["meta"]["seed"], "landmarks")
    added = []
    names = {}
    k = 0
    if "landmark" not in stars:
        stars["landmark"] = np.zeros(stars["x"].size, np.int8)

    def add(s):
        nonlocal k
        added.append(s)
        k += 1

    if "auto" in str(lm.get("mode", "auto")):
        con = lm["auto_constraints"]
        V = stars["V"]
        # 1. 全天最亮星
        if V.min() > con["brightest_star_mag_max"]:
            l, b = rng.uniform(0, 360), rng.uniform(-30, 30)
            s = make_star(cfg, galaxy, dust, k, 2.1, E.MS, 0.25, l, b, V_target=con["brightest_star_mag_max"] - 0.25)
            add(s)
            log(f"  [地标] 补最亮星：A 型主序星，V={s['V'][0]:.2f}，{s['d'][0]:.1f} pc")
        # 2. 显眼的红超巨星
        cls = E.lum_class(stars["phase"], stars["L"], stars["mass"])
        rsg = (cls >= 4) & (stars["Teff"] < 4200) & (stars["V"] < 1.5)
        if con.get("bright_red_supergiant") and not rsg.any():
            ob = np.nonzero(dso["kind"] == 1)[0]
            if ob.size:
                dd = np.hypot(np.hypot(dso["x"][ob], dso["y"][ob]), dso["z"][ob])
                j = ob[np.argmin(dd)]
                from .coords import lonlat
                l, b = lonlat(np.array([dso["x"][j], dso["y"][j], dso["z"][j]]))
                l, b = float(l) + rng.normal(0, 3), float(b) + rng.normal(0, 2)
            else:
                l, b = rng.uniform(0, 360), rng.uniform(-10, 10)
            m = 16.0
            s = make_star(cfg, galaxy, dust, k, m, E.CHEB, _phase_mid_age(m, E.CHEB), l, b, V_target=0.6)
            add(s)
            log(f"  [地标] 补红超巨星：V={s['V'][0]:.2f}，{s['d'][0]:.0f} pc，{s['Teff'][0]:.0f} K")
        # 3. 1 等以上亮星的数量
        lo, hi = con["stars_brighter_than_1mag"]
        n1 = int((V < 1.0).sum()) + sum(int(s["V"][0] < 1.0) for s in added)
        if n1 < lo:
            kinds = [(5.0, E.MS, 0.3), (2.6, E.CHEB, 0.5), (1.4, E.RGB, 0.9), (3.5, E.MS, 0.5), (9.0, E.MS, 0.5)]
            for i in range(lo - n1):
                m, ph, fr = kinds[i % len(kinds)]
                l, b = rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-1, 1)))
                s = make_star(cfg, galaxy, dust, k, m, ph, _phase_mid_age(m, ph, fr), l, b, V_target=rng.uniform(0.0, 0.95))
                add(s)
            log(f"  [地标] 1 等以上只有 {n1} 颗，补了 {lo - n1} 颗")
        elif n1 > hi:
            log(f"  [提示] 1 等以上亮星 {n1} 颗，多于约束上限 {hi}（不删星；可调 local_stellar_density）")

    for j, mspec in enumerate(lm.get("manual", []) or []):
        ph = _PHASE.get(mspec.get("phase", "MS"), E.MS)
        m = float(mspec.get("mass_msun", 2.0))
        age = float(mspec.get("age_gyr", _phase_mid_age(m, ph)))
        l, b = mspec["gal_lon_lat_deg"]
        s = make_star(cfg, galaxy, dust, 1000 + j, m, ph, age, l, b,
                      d=mspec.get("distance_pc"), V_target=mspec.get("apparent_mag"), name=mspec.get("name", ""))
        add(s)

    for s in added:
        if s.get("_name"):
            names[(int(s["id_hi"][0]), 0)] = s["_name"]
        s.pop("_name", None)
    if added:
        stars = concat([stars] + added)
    return stars, names
