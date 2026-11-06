"""输出：真值星表、世界内星表、各年代肉眼星表、观星器数据。"""
import json
from pathlib import Path

import numpy as np

from . import evolution as E
from . import hashrng as H
from .clusters import DSO_KINDS
from .components import component_names, component_table, offsets_deg
from .coords import apply, lonlat
from .galaxy import KMS_TO_PCYR
from .observe import RES_NAKED, RES_TELESCOPE, sky_positions
from .variables import VAR_TYPES

RES_NAMES = np.array(["none", "telescope", "naked_eye"])
COLOR_WORDS = [(-9, "蓝白"), (0.0, "白"), (0.3, "黄白"), (0.6, "黄"), (1.0, "橙"), (1.4, "红")]


def star_ids(stars):
    hi = stars["id_hi"].astype(np.uint64)
    return np.array([f"{int(h):012x}-{int(i):03d}" for h, i in zip(hi, stars["id_idx"])])


def _fmt(a, f):
    a = np.asarray(a)
    if a.dtype.kind in "US":
        return a.astype(str)
    s = np.char.mod(f, a)
    return np.where(np.isfinite(a.astype(np.float64)) if a.dtype.kind == "f" else True, s, "")


def write_csv(path, cols):
    """cols: [(表头, 数组, 格式)]；按列向量化格式化。"""
    lines = None
    for name, arr, f in cols:
        s = _fmt(arr, f)
        lines = s if lines is None else np.char.add(np.char.add(lines, ","), s)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as fh:
        fh.write(",".join(c[0] for c in cols) + "\n")
        if lines is not None and lines.size:
            fh.write("\n".join(lines.tolist()))
            fh.write("\n")


def spectral_types(stars):
    """光谱型。伴星比主星亮时写成合成光谱 "伴星+主星"（例如 A1V+DA）：主星是原来更重、先演化掉的那颗，但光主要来自伴星。"""
    spt = np.array(E.spectral_string(*E.classify(stars["Teff"], stars["phase"], stars["L"], stars["mass"])), dtype=object)
    if "c1_MV" in stars:
        k = np.nonzero(np.isfinite(stars["c1_MV"]) & (stars["c1_MV"] < stars["MV"]) & (stars["c1_L"] > 0))[0]
        if k.size:
            cs = E.spectral_string(*E.classify(stars["c1_Teff"][k], stars["c1_phase"][k], stars["c1_L"][k], stars["c1_mass"][k]))
            spt[k] = [f"{c}+{p}" for c, p in zip(cs, spt[k])]
    return spt.astype(str)


def write_truth(out, stars, dso, names, ids, spt):
    """真值：truth_catalog.csv 每颗恒星一行（多星系统拆成 A/B/C），truth_systems.csv 每个系统一行。"""
    t = component_table(stars, ids)
    si = t["sys"]
    cname = component_names(names, stars, t)
    cl = np.where(stars["cluster_id"] >= 0, dso["name"][np.maximum(stars["cluster_id"], 0)], "")
    dra, ddec = offsets_deg(t)
    cosd = np.maximum(np.cos(np.radians(stars["dec"][si])), 1e-3)
    is_a = t["comp"] == 0
    vt = np.array(VAR_TYPES)[t["var_type"]]
    var_P = np.where(is_a, stars["var_period_d"][si], np.where(t["var_type"] > 0, t["period_d"], np.nan))
    var_A = np.where(is_a, stars["var_amp_mag"][si], np.where(t["var_type"] > 0, stars["var_amp_mag"][si], np.nan))
    parent = np.array(["", "A", "AB"])[t["comp"]]
    multi = stars["multiplicity"][si] > 1
    cols = [
        ("id", t["id"], "%s"), ("system_id", t["sys_id"], "%s"),
        ("component", np.where(multi, np.array(list("ABC"))[t["comp"]], ""), "%s"), ("name", cname, "%s"),
        ("ra", (stars["ra"][si] + dra / cosd) % 360, "%.5f"), ("dec", stars["dec"][si] + ddec, "%.5f"),
        ("ecl_lon", stars["ecl_lon"][si], "%.5f"), ("ecl_lat", stars["ecl_lat"][si], "%.5f"),
        ("gal_lon", stars["gal_lon"][si], "%.5f"), ("gal_lat", stars["gal_lat"][si], "%.5f"),
        ("distance_pc", stars["d"][si], "%.2f"), ("parallax_mas", stars["parallax_mas"][si], "%.3f"),
        ("pm_ra_mas_yr", stars["pm_ra"][si], "%.3f"), ("pm_dec_mas_yr", stars["pm_dec"][si], "%.3f"), ("rv_kms", stars["rv"][si], "%.2f"),
        ("spectral_type", t["spt"], "%s"), ("app_mag", np.where(np.isfinite(t["V"]), t["V"], np.nan), "%.2f"),
        ("abs_mag", np.where(t["MV"] < 90, t["MV"], np.nan), "%.2f"), ("color_bv", t["BV"], "%.3f"),
        ("temp_k", t["Teff"], "%.0f"), ("luminosity_lsun", t["L"], "%.4g"), ("radius_rsun", t["R"], "%.4g"),
        ("mass_msun", t["mass"], "%.3f"), ("age_myr", stars["age"][si] * 1000, "%.1f"), ("evo_phase", t["phase_name"], "%s"),
        ("A_V", t["A_V"], "%.2f"),
        ("variable_type", vt, "%s"), ("var_period_days", var_P, "%.4f"), ("var_amplitude_mag", var_A, "%.2f"),
        ("orbits", parent, "%s"), ("orbit_period_days", t["period_d"], "%.4g"), ("semimajor_axis_au", t["a_au"], "%.4g"),
        ("eccentricity", t["e"], "%.3f"), ("inclination_deg", t["incl_deg"], "%.1f"),
        ("sep_arcsec", t["sep_arcsec"], "%.3f"), ("pa_deg", t["pa_deg"], "%.1f"),
        ("resolvable_by", np.where(is_a, "", RES_NAMES[t["resolvable"]]), "%s"),
        ("system_multiplicity", stars["multiplicity"][si], "%d"), ("system_app_mag", stars["V"][si], "%.2f"),
        ("cluster_id", cl[si], "%s"),
    ]
    write_csv(out / "truth_catalog.csv", cols)

    members = [" ".join(t["id"][t["first"][i]:t["first"][i] + int(stars["multiplicity"][i])]) for i in range(stars["x"].size)]
    sname = np.array([names.get((int(h), int(i)), "") for h, i in zip(stars["id_hi"], stars["id_idx"])])
    write_csv(out / "truth_systems.csv", [
        ("system_id", ids, "%s"), ("name", sname, "%s"), ("ra", stars["ra"], "%.5f"), ("dec", stars["dec"], "%.5f"),
        ("distance_pc", stars["d"], "%.2f"), ("multiplicity", stars["multiplicity"], "%d"), ("members", np.array(members), "%s"),
        ("spectral_type", spt, "%s"), ("app_mag", stars["V"], "%.2f"), ("abs_mag", stars["MV_sys"], "%.2f"),
        ("color_bv", stars["BV"], "%.3f"), ("variable_type", np.array(VAR_TYPES)[stars["var_type"]], "%s"),
        ("cluster_id", cl, "%s"),
    ])
    return t


def _color_word(bv):
    w = np.full(bv.shape, COLOR_WORDS[0][1], dtype=object)
    for t, name in COLOR_WORDS:
        w[bv >= t] = name
    return w.astype(str)


def _seen_entries(stars, ids, level, lim):
    """返回某种观测手段下"看到的"条目：没分开的双星合成一颗，分开的伴星单独一条。"""
    sys_vis = stars["V"] <= lim
    res1 = stars["c1_resolvable"] >= level
    res2 = stars["c2_resolvable"] >= level
    # 主星（分开时用主星自己的亮度）
    Vp = np.where(res1, stars["V_primary"], stars["V"])
    keep = np.where(res1, stars["V_primary"] <= lim, sys_vis)
    e = {"idx": np.nonzero(keep)[0]}
    e["id"] = np.where(res1[e["idx"]] | res2[e["idx"]], np.char.add(ids[e["idx"]], "A"), ids[e["idx"]])
    e["V"] = Vp[e["idx"]]
    e["dra"] = np.zeros(e["idx"].size)
    e["ddec"] = np.zeros(e["idx"].size)
    e["seen_double"] = res1[e["idx"]] | res2[e["idx"]]
    e["comp"] = np.zeros(e["idx"].size, np.int8)
    parts = [e]
    for c, res in ((1, res1), (2, res2)):
        k = np.nonzero(res & (stars[f"c{c}_V"] <= lim))[0]
        if k.size == 0:
            continue
        sep = stars[f"c{c}_sep_arcsec"][k] / 3600
        pa = np.radians(stars[f"c{c}_pa_deg"][k])
        parts.append({"idx": k, "id": np.char.add(ids[k], "BC"[c - 1]), "V": stars[f"c{c}_V"][k],
                      "dra": sep * np.sin(pa), "ddec": sep * np.cos(pa), "seen_double": np.ones(k.size, bool),
                      "comp": np.full(k.size, c, np.int8)})
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def write_in_world(out, stars, ids, cfg):
    ob = cfg["observing"]
    sh = stars["hash"]
    for level, key, lim, pos_acc, mag_step, mag_sig, var_thr in (
        (RES_NAKED, "naked_eye", ob["naked_eye"]["limit_mag"], ob["naked_eye"]["position_accuracy_arcsec"], 1.0,
         ob["naked_eye"]["magnitude_precision"] / 2, 0.6),
        (RES_TELESCOPE, "telescope", ob["telescope"]["limit_mag"], ob["telescope"]["position_accuracy_arcsec"], 0.5, 0.3, 0.4),
    ):
        e = _seen_entries(stars, ids, level, lim)
        i = e["idx"]
        h = H.combine(sh[i], level * 1000 + e["comp"].astype(np.int64))
        cosd = np.cos(np.radians(stars["dec"][i]))
        ra = stars["ra"][i] + e["dra"] / np.maximum(cosd, 1e-3) + H.normal(h, 201) * pos_acc / 3600 / np.maximum(cosd, 1e-3)
        dec = stars["dec"][i] + e["ddec"] + H.normal(h, 202) * pos_acc / 3600
        mag = np.round((e["V"] + H.normal(h, 203) * mag_sig) / mag_step) * mag_step
        mag = np.clip(mag, 1.0 if level == RES_NAKED else -2.0, None)
        bv = stars["BV"][i] + H.normal(h, 204) * 0.15
        noticed = (stars["var_amp_mag"][i] >= var_thr) & (stars["var_type"][i] != VAR_TYPES.index("DSCT"))
        order = np.argsort(mag, kind="stable")
        cols = [
            ("id", e["id"][order], "%s"), ("name", np.full(i.size, ""), "%s"),
            ("ra", (ra % 360)[order], "%.3f"), ("dec", dec[order], "%.3f"),
            ("app_mag_estimate", mag[order], "%.1f"), ("color_word", _color_word(bv)[order], "%s"),
            ("multiplicity_seen", np.where(e["seen_double"], "双", "")[order], "%s"),
            ("variable_noticed", np.where(noticed, "是", "")[order], "%s"),
        ]
        write_csv(out / f"in_world_{key}.csv", cols)


def write_epochs(out, stars, ids, clock, cfg):
    lim = cfg["observing"]["naked_eye"]["limit_mag"]
    k = stars["V"] <= lim
    sub = {kk: v[k] for kk, v in stars.items() if isinstance(v, np.ndarray) and v.shape[:1] == k.shape}
    for yr in cfg["output"]["epochs"].get("extra_years", []) or []:
        a = _alive(sub, yr)
        s2 = _subset(sub, a)
        p = sky_positions(s2, clock, yr)
        write_csv(out / f"naked_eye_epoch_{int(yr)}.csv", [
            ("id", ids[k][a], "%s"), ("ra", p["ra"], "%.4f"), ("dec", p["dec"], "%.4f"),
            ("ecl_lon", p["ecl_lon"], "%.4f"), ("ecl_lat", p["ecl_lat"], "%.4f"), ("app_mag", s2["V"], "%.2f")])


def write_viewer(out, stars, dso, events, ids, spt, clock, cfg, names=None):
    """观星器数据：stars.bin（Float32，每星 12 个数）+ stars_info.json + sky_meta.json + dso.json + events.json。"""
    vdir = out / "viewer"
    vdir.mkdir(parents=True, exist_ok=True)
    tel_lim = cfg["observing"]["telescope"]["limit_mag"]
    e = _seen_entries(stars, ids, RES_TELESCOPE, tel_lim + 0.5)
    i = e["idx"]
    M = clock.M_g2e
    p = np.column_stack([stars["x"], stars["y"], stars["z"]])[i]
    v = np.column_stack([stars["vx"], stars["vy"], stars["vz"]])[i]
    u0 = apply(M, p / np.linalg.norm(p, axis=1)[:, None])
    p1 = p + v * KMS_TO_PCYR * 1000.0
    du = apply(M, p1 / np.linalg.norm(p1, axis=1)[:, None]) - u0
    is_comp = e["comp"] > 0
    # 伴星：位置角在赤道坐标里定义；这里近似在黄道坐标里偏移（只差几角秒到几角分）
    off = np.nonzero(is_comp)[0]
    if off.size:
        from .coords import unit
        lam, bet = lonlat(u0[off])
        u0[off] = unit(lam + e["dra"][off] / np.cos(np.radians(bet)), bet + e["ddec"][off])
    t = component_table(stars, ids)
    r = t["first"][i] + e["comp"]                        # 每个条目对应的成员行
    bv = np.where(is_comp, t["BV"][r], stars["BV"][i])
    var_t = np.where(is_comp, 0, stars["var_type"][i]).astype(np.float32)
    # 肉眼视图下的亮度：肉眼分不开的双星显示合成亮度，分不开的伴星不显示（99）
    V_naked = np.where(stars["c1_resolvable"][i] == RES_NAKED, stars["V_primary"][i], stars["V"][i])
    for c in (1, 2):
        kk = e["comp"] == c
        V_naked[kk] = np.where(stars[f"c{c}_resolvable"][i[kk]] == RES_NAKED, e["V"][kk], 99.0)
    arr = np.column_stack([
        u0, du, e["V"], bv,
        np.nan_to_num(np.where(is_comp, 0, stars["var_amp_mag"][i])),
        np.nan_to_num(np.where(is_comp, 0, stars["var_period_d"][i])),
        stars["var_epoch_frac"][i], var_t, V_naked,
    ]).astype(np.float32)
    arr.tofile(vdir / "stars.bin")

    cl = np.where(stars["cluster_id"][i] >= 0, dso["name"][np.maximum(stars["cluster_id"][i], 0)], "")
    cname = component_names(names or {}, stars, t)
    sname = np.array([(names or {}).get((int(stars["id_hi"][j]), int(stars["id_idx"][j])), "") for j in i])
    seen_sep = np.where(is_comp, t["sep_arcsec"][r], stars["c1_sep_arcsec"][i])
    seen_res = np.where(is_comp, t["resolvable"][r], stars["c1_resolvable"][i])
    # 多星系统的成员表（comps.json）：观星器里点一个系统能看到每颗成员自己的数据
    multi = stars["multiplicity"][i] > 1
    sys_entry = {}
    for ent in np.nonzero(multi & ~is_comp)[0]:
        sys_entry[int(i[ent])] = int(ent)
    drawn = {(int(i[ent]), int(e["comp"][ent])): int(ent) for ent in range(i.size)}
    rows, c_start, c_n = [], np.full(i.size, -1), np.zeros(i.size, int)
    for sysi, ent0 in sys_entry.items():
        f = int(t["first"][sysi])
        m = int(stars["multiplicity"][sysi])
        c_start_val = len(rows)
        rows.extend(range(f, f + m))
        for c in range(m):
            ent = drawn.get((sysi, c))
            if ent is not None:
                c_start[ent], c_n[ent] = c_start_val, m
    rows = np.array(rows, dtype=int)
    if rows.size:
        rs = t["sys"][rows]
        ent_drawn = np.array([drawn.get((int(a), int(b)), sys_entry[int(a)]) for a, b in zip(rs, t["comp"][rows])])
        num = lambda a, d: [None if not np.isfinite(x) else round(float(x), d) for x in a]
        comps = {
            "entry": ent_drawn.tolist(), "sys_entry": [sys_entry[int(a)] for a in rs],
            "id": t["id"][rows].tolist(), "letter": np.array(list("ABC"))[t["comp"][rows]].tolist(),
            "name": cname[rows].tolist(), "spt": t["spt"][rows].tolist(), "phase": t["phase_name"][rows].tolist(),
            "mass": num(t["mass"][rows], 3), "teff": num(t["Teff"][rows], 0), "lum": [float("%.4g" % x) for x in t["L"][rows]],
            "rad": [float("%.4g" % x) for x in t["R"][rows]],
            "MV": num(np.where(t["MV"][rows] < 90, t["MV"][rows], np.nan), 2), "V": num(t["V"][rows], 2),
            "BV": num(t["BV"][rows], 2),
            "sep": num(t["sep_arcsec"][rows], 3), "period_d": num(t["period_d"][rows], 3), "a_au": num(t["a_au"][rows], 4),
            "e": num(t["e"][rows], 3), "res": RES_NAMES[t["resolvable"][rows]].tolist(), "var": t["var_type"][rows].tolist(),
        }
    else:
        comps = {k: [] for k in ("entry", "sys_entry", "id", "letter", "name", "spt", "phase", "mass", "teff", "lum", "rad",
                                 "MV", "V", "BV", "sep", "period_d", "a_au", "e", "res", "var")}
    with open(vdir / "comps.json", "w", encoding="utf-8") as fh:
        json.dump(comps, fh, ensure_ascii=False, separators=(",", ":"))
    info = {
        "fields": ["ux", "uy", "uz", "dux_per_kyr", "duy_per_kyr", "duz_per_kyr", "V", "BV", "var_amp", "var_period_d",
                   "var_epoch_frac", "var_type", "V_naked"],
        "var_types": VAR_TYPES,
        "id": e["id"].tolist(),
        "sys_id": ids[i].tolist(),
        "name": np.where(is_comp | (e["id"] != ids[i]), cname[r], sname).tolist(),
        "is_companion": is_comp.astype(int).tolist(),
        "spt": t["spt"][r].tolist(),                         # 这颗星自己的光谱型
        "spt_sys": spt[i].tolist(),                           # 系统的合成光谱型（单星同上）
        "dist_pc": np.round(stars["d"][i], 2).tolist(),
        "parallax_mas": np.round(stars["parallax_mas"][i], 3).tolist(),
        "MV": np.round(np.where(t["MV"][r] < 90, t["MV"][r], 99), 2).tolist(),
        "MV_sys": np.round(stars["MV_sys"][i], 2).tolist(),
        "mass": np.round(t["mass"][r], 3).tolist(),
        "age_myr": np.round(stars["age"][i] * 1000, 1).tolist(),
        "phase": t["phase_name"][r].tolist(),
        "teff": np.round(t["Teff"][r]).astype(int).tolist(),
        "lum": [float("%.4g" % x) for x in t["L"][r]],
        "rad": [float("%.4g" % x) for x in t["R"][r]],
        "mult": stars["multiplicity"][i].astype(int).tolist(),
        "sep_arcsec": np.round(np.nan_to_num(seen_sep, nan=-1), 3).tolist(),
        "resolvable": RES_NAMES[seen_res].tolist(),
        "cluster": cl.tolist(),
        "A_V": np.round(stars["A_V"][i], 2).tolist(),
        "comp_start": c_start.tolist(), "comp_n": c_n.tolist(),
    }
    with open(vdir / "stars_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, separators=(",", ":"))

    gal_dirs = np.column_stack([dso["x"], dso["y"], dso["z"]])
    gal_dirs = gal_dirs / np.linalg.norm(gal_dirs, axis=1)[:, None]
    ue = apply(M, gal_dirs)
    dlist = []
    for j in range(dso["kind"].size):
        if not (dso["V_int"][j] <= 10.0):
            continue
        dlist.append({"name": str(dso["name"][j]), "kind": DSO_KINDS[int(dso["kind"][j])],
                      "u": np.round(ue[j], 6).tolist(), "V": round(float(dso["V_int"][j]), 2),
                      "size_deg": round(float(dso["ang_size_deg"][j]), 3), "dist_pc": round(float(dso["d"][j]), 0),
                      "age_myr": None if not np.isfinite(dso["age_myr"][j]) else float("%.4g" % dso["age_myr"][j])})
    with open(vdir / "dso.json", "w", encoding="utf-8") as fh:
        json.dump(dlist, fh, ensure_ascii=False)
    with open(vdir / "events.json", "w", encoding="utf-8") as fh:
        json.dump(events, fh, ensure_ascii=False)

    sc = cfg["sky_clock"]
    meta = {
        "n_stars": int(arr.shape[0]), "floats_per_star": int(arr.shape[1]),
        "gal_to_ecl": M.tolist(),
        "precession_period_yr": clock.P, "obliquity_deg": clock.eps0,
        "obliquity_osc_amp_deg": clock.eps_amp, "obliquity_osc_period_yr": clock.eps_per,
        "equinox_ecl_lon_at_reference_deg": clock.lam0, "reference_year": clock.ref_year,
        "latitude_deg": sc["site"]["latitude_deg"], "longitude_deg": sc["site"].get("longitude_deg", 0),
        "year_days": sc["orbit"]["year_days"], "sidereal_day_hours": sc["rotation"]["sidereal_day_hours"],
        "naked_eye_limit": cfg["observing"]["naked_eye"]["limit_mag"], "telescope_limit": tel_lim,
        "telescope_resolution_arcsec": cfg["observing"]["telescope"]["resolution_arcsec"],
        "extinction_per_airmass": cfg["observing"]["atmosphere"]["extinction_mag_per_airmass"],
        "moons": sc["moons"], "lunar_mansions": sc["lunar_mansions"]["count"],
        "background": "background.png",
    }
    with open(vdir / "sky_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return arr.shape[0]


def _alive(stars, year):
    """在 year 年还没爆发成超新星的星。"""
    sy = stars.get("sn_year")
    return np.ones(stars["x"].size, bool) if sy is None else ~(sy <= year)


def _subset(stars, k):
    return {kk: v[k] for kk, v in stars.items() if isinstance(v, np.ndarray) and v.shape[:1] == k.shape}


def write_all(out, stars, dso, events, names, clock, cfg, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = star_ids(stars)
    spt = spectral_types(stars)
    now = _alive(stars, clock.ref_year)                 # 故事当下已经炸掉的星不进当下的星表，但观星器里看早年时还在
    t = write_truth(out, _subset(stars, now), dso, names, ids[now], spt[now])
    log(f"  真值星表 {t['id'].size:,} 颗恒星（{int(now.sum()):,} 个系统）→ truth_catalog.csv、truth_systems.csv")
    write_in_world(out, _subset(stars, now), ids[now], cfg)
    write_epochs(out, stars, ids, clock, cfg)
    with open(out / "events.json", "w", encoding="utf-8") as fh:
        json.dump(events, fh, ensure_ascii=False, indent=1)
    keys = ["year", "day", "type", "peak_mag", "visible_days", "ra", "dec", "gal_lon", "gal_lat", "distance_pc", "max_altitude_deg",
            "observable", "future", "star_id"]
    with open(out / "events.csv", "w", encoding="utf-8-sig") as fh:
        fh.write(",".join(keys) + "\n")
        for r in events:
            fh.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    dcols = [(k, dso[k], f) for k, f in (("name", "%s"), ("V_int", "%.2f"), ("d", "%.0f"), ("ang_size_deg", "%.3f"),
                                         ("age_myr", "%.4g"), ("n_members", "%d"), ("gal_lon", "%.3f"), ("gal_lat", "%.3f"))]
    dcols.insert(1, ("kind", np.array(DSO_KINDS)[dso["kind"]], "%s"))
    write_csv(out / "dso_catalog.csv", dcols)
    n_view = write_viewer(out, stars, dso, events, ids, spt, clock, cfg, names)
    log(f"  世界内星表、各年代肉眼星表、事件表、深空天体表、观星器数据（{n_view:,} 个点）已写出")
