"""把多星系统拆成一颗颗恒星：每颗成员星一行，有自己的编号、物理量、视星等和轨道。

内部生成时伴星存在主星那一行的 c1_* / c2_* 字段里；这里只是换一种组织方式，不改任何物理量。

编号：单星 = 系统编号（xxxxxxxxxxxx-NNN）；多星系统的成员 = 系统编号 + A/B/C。
  A = 主星（原本最重、最先演化的那颗；先演化成白矮星时可能比 B 暗）
  B = 绕 A 的伴星（内层轨道）
  C = 绕 A+B 的外层伴星
"""
import numpy as np

from . import evolution as E
from .variables import VAR_TYPES

LETTERS = "ABC"
PHASE_DEAD = "残骸"           # 伴星已经死亡（中子星/黑洞，看不见）
_ECL = {VAR_TYPES.index(t) for t in ("EA", "EB", "EW")}


def component_table(stars, ids):
    """返回 dict：每颗成员星一行，按 (系统, 成员) 排序；另含 first[i] = 系统 i 的 A 所在行。"""
    n = stars["x"].size
    DM = stars["V"] - stars["MV_sys"]                      # 距离模数 + 消光
    multi = stars["multiplicity"] > 1
    parts = []
    for c in range(3):
        if c == 0:
            k = np.arange(n)
            g = lambda f: stars[f][k]
            mass, MV, T, L, R = g("mass"), g("MV"), g("Teff"), g("L"), g("R")
            ph = stars["phase"][k].astype(np.int64)
            nanf = np.full(k.size, np.nan)
            orb = {"period_d": nanf, "a_au": nanf, "e": nanf, "incl_deg": nanf, "sep_arcsec": nanf, "pa_deg": nanf}
            res = np.maximum(stars["c1_resolvable"][k], stars["c2_resolvable"][k])
            vt = stars["var_type"][k]
        else:
            p = f"c{c}_"
            k = np.nonzero(np.isfinite(stars[p + "mass"]))[0]
            g = lambda f: stars[p + f][k]
            mass, MV, T, L, R = g("mass"), g("MV"), g("Teff"), g("L"), g("R")
            ph = np.nan_to_num(g("phase"), nan=-1).astype(np.int64)
            orb = {f: g(f) for f in ("period_d", "a_au", "e", "incl_deg", "sep_arcsec", "pa_deg")}
            res = g("resolvable")
            vt = np.where(np.isin(stars["var_type"][k], list(_ECL)) & (c == 1), stars["var_type"][k], 0)
        parts.append({"sys": k, "comp": np.full(k.size, c, np.int8), "mass": mass, "MV": MV, "Teff": T, "L": L, "R": R,
                      "phase": ph, "resolvable": res.astype(np.int8), "var_type": vt.astype(np.int8), **orb})
    t = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.lexsort((t["comp"], t["sys"]))
    t = {key: v[order] for key, v in t.items()}
    s, c = t["sys"], t["comp"]
    dead = t["phase"] < 0
    t["V"] = np.where(dead | (t["MV"] > 90), np.inf, t["MV"] + DM[s])
    t["A_V"] = stars["A_V"][s]
    t["BV"] = E.bv_from_teff(t["Teff"]) + t["A_V"] / 3.1
    ph_ok = np.maximum(t["phase"], 0)
    spt = np.array(E.spectral_string(*E.classify(t["Teff"], ph_ok, t["L"], t["mass"])), dtype=object)
    spt[dead] = "—"
    t["spt"] = spt.astype(str)
    t["phase_name"] = np.where(dead, PHASE_DEAD, np.array(E.PHASE_NAMES)[ph_ok])
    t["id"] = np.where(multi[s], np.char.add(ids[s], np.array(list(LETTERS))[c]), ids[s])
    t["sys_id"] = ids[s]
    t["first"] = np.nonzero(c == 0)[0]                      # first[i]：系统 i 的 A 行
    return t


def component_names(names, stars, t):
    """星名：成员自己的名字（覆盖层里按 xxxB 设的）优先；否则多星系统成员用"系统名 A/B/C"，单星用系统名。"""
    hi, idx = stars["id_hi"][t["sys"]], stars["id_idx"][t["sys"]]
    multi = stars["multiplicity"][t["sys"]] > 1
    out = []
    for h, i, c, m in zip(hi.tolist(), idx.tolist(), t["comp"].tolist(), multi.tolist()):
        own = names.get((int(h), int(i), int(c)))
        if own:
            out.append(own)
            continue
        sysn = names.get((int(h), int(i)), "")
        out.append(f"{sysn} {LETTERS[c]}" if (sysn and m) else sysn)
    return np.array(out)


def offsets_deg(t):
    """成员相对系统中心（A）的天球偏移（度）：Δα·cosδ 和 Δδ。"""
    sep = np.nan_to_num(t["sep_arcsec"]) / 3600.0
    pa = np.radians(np.nan_to_num(t["pa_deg"]))
    return sep * np.sin(pa), sep * np.cos(pa)
