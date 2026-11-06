"""事件时间线：超新星、新星、大彗星（含故事当下之后的"未来"段），以及超新星遗迹。

在整个星系里按发生率抽事件，用同一份尘埃场算消光，只保留观察者可能看见的（峰值 ≤ VISIBLE_MAG）。
位置按事件当年的岁差换算赤道坐标，并判断北纬 35° 是否升得起来。
"""
import math

import numpy as np

from .coords import apply, lonlat, unit
from .hashrng import module_rng

VISIBLE_MAG = 6.0
NOVA_RECORD_MAG = 4.0    # 新星只记"会被古人注意到"的（≤4 等的"客星"）；更暗的太多也不会进史书


def _galaxy_positions(rng, galaxy, n, kind):
    """kind='young'：薄盘、偏向旋臂；kind='old'：盘 + 核球（按光度）。返回相对观察者的坐标。"""
    out = []
    need = n
    while need > 0:
        m = 3 * need + 64
        bulge = (rng.random(m) < 0.3) if kind == "old" else np.zeros(m, bool)
        Rd = 2800.0
        R = rng.gamma(2.0, Rd, m)                         # 指数盘的面密度 ∝ R·e^{-R/Rd}
        ph = rng.random(m) * 2 * np.pi
        X, Y = R * np.cos(ph), R * np.sin(ph)
        h = 75.0 if kind == "young" else 300.0
        Z = rng.laplace(0, h, m)
        rb = np.abs(rng.normal(0, 700.0, (m, 3)))
        sb = rng.choice([-1, 1], (m, 3))
        X = np.where(bulge, rb[:, 0] * sb[:, 0], X)
        Y = np.where(bulge, rb[:, 1] * sb[:, 1], Y)
        Z = np.where(bulge, rb[:, 2] * sb[:, 2] * 0.6, Z)
        x = X + galaxy.R0
        y = Y
        if kind == "young" and galaxy.arms_on:
            w = galaxy.arm_factor(x, y, galaxy.arm_contrast)
            keep = rng.random(m) < w / w.max()
        else:
            keep = np.ones(m, bool)
        R_ok = np.hypot(X, Y) < 16000
        sel = keep & R_ok
        pts = np.column_stack([x, y, Z - galaxy.z_obs])[sel][:need]
        out.append(pts)
        need -= pts.shape[0]
    return np.vstack(out)


def _sky(clock, p, years, site_lat):
    d = np.linalg.norm(p, axis=1)
    u = p / d[:, None]
    l, b = lonlat(u)
    ra = np.empty(d.size)
    dec = np.empty(d.size)
    for yr in np.unique(np.round(years, -1)):
        k = np.round(years, -1) == yr
        ra[k], dec[k] = lonlat(apply(clock.gal_to_equ(yr), u[k]))
    max_alt = 90 - np.abs(site_lat - dec)
    return d, u, l, b, ra, dec, max_alt


def _sample_supernovae(rng, sn, galaxy, y0, y1):
    """[y0, y1) 年间整个星系的超新星：年份、相对观察者的位置、峰值绝对星等、是否 Ia。"""
    n = int(rng.poisson(sn["galactic_rate_per_century"] * (y1 - y0) / 100))
    ia = rng.random(n) < sn["type_ia_fraction"]
    yrs = rng.uniform(y0, y1, n)
    p = np.zeros((n, 3))
    Mp = np.zeros(n)
    for typ, mask, kind, Mpk in ((True, ia, "old", sn["peak_abs_mag"]["type_ia"]), (False, ~ia, "young", sn["peak_abs_mag"]["core_collapse"])):
        k = int(mask.sum())
        if k:
            p[mask] = _galaxy_positions(rng, galaxy, k, kind)
            Mp[mask] = Mpk + rng.normal(0, 0.4 if typ else 1.0, k)
    return yrs, p, Mp, ia


def make_events(cfg, galaxy, dust, clock, log=print):
    """返回 (事件表, 超新星全表)。超新星全表含看不见的，用来生成遗迹。"""
    ev = cfg["events"]
    rng = module_rng(cfg["meta"]["seed"], "events")
    y0, y1 = ev["timeline_yr"]
    span = y1 - y0
    present = clock.ref_year
    site_lat = cfg["sky_clock"]["site"]["latitude_deg"]
    rows = []

    def add(kind, years, p, peak_abs, fade, lim=VISIBLE_MAG):
        if len(years) == 0:
            return
        d, u, l, b, ra, dec, max_alt = _sky(clock, p, years, site_lat)
        A = dust.extinction(u, d)
        V = peak_abs + 5 * np.log10(d) - 5 + A
        vis = V <= lim
        for i in np.nonzero(vis)[0]:
            days = fade(V[i], i)
            rows.append({
                "year": int(years[i]), "day": int(rng.integers(0, 365)), "type": kind,
                "gal_lon": round(float(l[i]), 3), "gal_lat": round(float(b[i]), 3),
                "ra": round(float(ra[i]), 3), "dec": round(float(dec[i]), 3),
                "distance_pc": round(float(d[i]), 1), "A_V": round(float(A[i]), 2),
                "peak_mag": round(float(V[i]), 2), "visible_days": int(days),
                "max_altitude_deg": round(float(max_alt[i]), 1), "observable": bool(max_alt[i] > 5),
                "future": bool(years[i] > present),
            })

    # 超新星
    sn = ev["supernovae"]
    yrs, p_sn, Mp_sn, ia = _sample_supernovae(rng, sn, galaxy, y0, y1)
    n_sn = yrs.size
    for typ, mask in (("Ia", ia), ("CC", ~ia)):
        if not mask.any():
            continue
        rate = 0.03 if typ == "Ia" else 0.012                    # 每天变暗多少等（粗略）
        plateau = 20 if typ == "Ia" else 90
        add(f"supernova_{typ}", yrs[mask], p_sn[mask], Mp_sn[mask], lambda V, i, r=rate, pl=plateau: pl + (VISIBLE_MAG - V) / r)
    sn_all = {"year": yrs, "pos": p_sn, "ia": ia}
    n_sn_vis = sum(r["type"].startswith("supernova") for r in rows)

    # 新星：数量巨大，只抽"有可能看得见"的部分——先按距离上限筛
    nv = ev["novae"]
    n_nova = int(rng.poisson(nv["galactic_rate_per_year"] * span))
    batch = 200000
    left = n_nova
    while left > 0:
        k = min(batch, left)
        left -= k
        p = _galaxy_positions(rng, galaxy, k, "old")
        lo_M, hi_M = sorted(nv["peak_abs_mag"])            # 亮的新星少：偏向暗端
        Mp = hi_M - (hi_M - lo_M) * rng.random(k) ** 2
        near = 5 * np.log10(np.linalg.norm(p, axis=1)) - 5 + Mp <= NOVA_RECORD_MAG
        if not near.any():
            continue
        yrs = rng.uniform(y0, y1, int(near.sum()))
        t2 = 10 ** rng.uniform(1, 2, int(near.sum()))            # 下降 2 等所需天数
        add("nova", yrs, p[near], Mp[near], lambda V, i, t=t2: t[i] * (VISIBLE_MAG - V) / 2, lim=NOVA_RECORD_MAG)
    n_nova_vis = sum(r["type"] == "nova" for r in rows)

    # 大彗星
    cm = ev.get("comets", {}) or {}
    if cm.get("enabled"):
        n_c = int(rng.poisson(cm["great_comets_per_century"] * span / 100))
        for _ in range(n_c):
            lam = rng.uniform(0, 360)
            beta = rng.normal(0, 25)
            yr = rng.uniform(y0, y1)
            u_ecl = unit(lam, beta)
            ra, dec = lonlat(apply(clock.ecl_of_date_to_equ(yr), u_ecl))
            rows.append({"year": int(yr), "day": int(rng.integers(0, 365)), "type": "great_comet",
                         "ecl_lon": round(float(lam), 2), "ecl_lat": round(float(beta), 2),
                         "ra": round(float(ra), 3), "dec": round(float(dec), 3),
                         "peak_mag": round(float(rng.uniform(-4, 2)), 1), "tail_deg": round(float(rng.uniform(5, 90)), 0),
                         "visible_days": int(rng.uniform(20, 120)),
                         "max_altitude_deg": round(float(90 - abs(site_lat - dec)), 1), "observable": bool(90 - abs(site_lat - dec) > 5),
                         "future": bool(yr > present)})

    for m in ev.get("manual", []) or []:
        r = dict(m)
        r.setdefault("type", "manual")
        rows.append(r)

    rows.sort(key=lambda r: (r["year"], r.get("day", 0)))
    n_fut = sum(r.get("future", False) and r["type"].startswith("supernova") for r in rows)
    log(f"  {y0}–{y1} 年：星系超新星 {n_sn} 次，其中峰值 ≤{VISIBLE_MAG} 等 {n_sn_vis} 次（{n_fut} 次在 {present} 年之后）；"
        f"新星（≤{NOVA_RECORD_MAG} 等）{n_nova_vis} 次；大彗星 {sum(r['type'] == 'great_comet' for r in rows)} 颗")
    return rows, sn_all


def add_star_supernovae(events, sn_all, stars, cfg, clock):
    """覆盖层里指定爆发的星（stars["sn_year"] 有值）：写进事件表（带 star_id），并加入超新星全表以便生成遗迹。"""
    if "sn_year" not in stars:
        return events, sn_all
    k = np.nonzero(np.isfinite(stars["sn_year"]))[0]
    if k.size == 0:
        return events, sn_all
    from .export import star_ids
    sn = cfg["events"]["supernovae"]
    present = clock.ref_year
    site_lat = cfg["sky_clock"]["site"]["latitude_deg"]
    ids = star_ids({"id_hi": stars["id_hi"][k], "id_idx": stars["id_idx"][k]})
    p = np.column_stack([stars["x"][k], stars["y"][k], stars["z"][k]])
    yrs = stars["sn_year"][k]
    d, u, l, b, ra, dec, max_alt = _sky(clock, p, yrs, site_lat)
    ia = stars["phase"][k] == 5                                  # 白矮星 → Ia 型
    Mp = np.where(np.isfinite(stars["sn_peak_abs"][k]), stars["sn_peak_abs"][k],
                  np.where(ia, sn["peak_abs_mag"]["type_ia"], sn["peak_abs_mag"]["core_collapse"]))
    V = Mp + 5 * np.log10(d) - 5 + stars["A_V"][k]
    rows = list(events)
    for i in range(k.size):
        rate, plateau = (0.03, 20) if ia[i] else (0.012, 90)
        rows.append({
            "year": int(yrs[i]), "day": int(stars["sn_day"][k][i]), "type": "supernova_Ia" if ia[i] else "supernova_CC",
            "gal_lon": round(float(l[i]), 3), "gal_lat": round(float(b[i]), 3),
            "ra": round(float(ra[i]), 3), "dec": round(float(dec[i]), 3),
            "distance_pc": round(float(d[i]), 1), "A_V": round(float(stars["A_V"][k][i]), 2),
            "peak_mag": round(float(V[i]), 2), "visible_days": int(plateau + max(0.0, VISIBLE_MAG - V[i]) / rate),
            "max_altitude_deg": round(float(max_alt[i]), 1), "observable": bool(max_alt[i] > 5),
            "future": bool(yrs[i] > present), "star_id": str(ids[i]), "manual": True,
        })
    rows.sort(key=lambda r: (r["year"], r.get("day", 0)))
    sn_all = {"year": np.concatenate([sn_all["year"], yrs]), "pos": np.vstack([sn_all["pos"], p]),
              "ia": np.concatenate([sn_all["ia"], ia]),
              "force": np.concatenate([sn_all.get("force", np.zeros(sn_all["year"].size, bool)), np.ones(k.size, bool)])}
    return rows, sn_all


REMNANT_MAX_AGE_YR = 60000.0
REMNANT_MAG_LIMIT = 11.0


def make_remnants(cfg, galaxy, sn_all, clock):
    """故事当下可见的超新星遗迹：时间线里已经爆发的 + 时间线开始之前 6 万年内的。

    半径：自由膨胀（约 5000 km/s）与 Sedov 解 R ≈ 12.5 pc·(t/10⁴ yr)^0.4 取小者。
    积分亮度（Hα 为主）：M_V ≈ −4.5 + 2·log10(t/1000 yr)，个体散布 0.7 等；蟹状星云、天鹅圈大致落在这条线上。
    返回要追加进深空天体表的字段（还没算消光，交给 finish_dso）。
    """
    ev = cfg["events"]
    if not ev["supernovae"].get("remnants", True):
        return None
    present = clock.ref_year
    y0 = ev["timeline_yr"][0]
    rng = module_rng(cfg["meta"]["seed"], "snrold")
    yo, po, _, iao = _sample_supernovae(rng, ev["supernovae"], galaxy, present - REMNANT_MAX_AGE_YR, y0)
    k = sn_all["year"] <= present
    yrs = np.concatenate([yo, sn_all["year"][k]])
    pos = np.vstack([po, sn_all["pos"][k]])
    force = np.concatenate([np.zeros(yo.size, bool), sn_all.get("force", np.zeros(sn_all["year"].size, bool))[k]])
    t = np.maximum(present - yrs, 30.0)
    R = np.minimum(0.0051 * t, 12.5 * (t / 1e4) ** 0.4)
    MV = -4.5 + 2.0 * np.log10(np.maximum(t, 300.0) / 1000.0) + rng.normal(0, 0.7, t.size)
    d = np.linalg.norm(pos, axis=1)
    near = (MV + 5 * np.log10(d) - 5 <= REMNANT_MAG_LIMIT) | force       # 不计消光的粗筛；手动指定的一律保留
    return {"pos": pos[near], "radius_pc": R[near], "MV": MV[near], "age_myr": t[near] / 1e6, "year": yrs[near], "force": force[near]}
