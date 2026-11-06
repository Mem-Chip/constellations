"""双星与三合星（层级结构）+ 食变星。

伴星与主星同龄，用同一套演化轨迹。所有随机量取自主星哈希（属性号 20 起），结果稳定。
伴星字段前缀：c1_（内层伴星）、c2_（外层第三颗，绕内双星转）。
"""
import numpy as np

from . import evolution as E
from . import hashrng as H

RSUN_AU = 0.00465047
COMP_FIELDS = ["mass", "MV", "Teff", "L", "R", "phase", "period_d", "a_au", "e", "incl_deg", "sep_au", "pa_deg"]


def _flux(MV):
    return np.where(MV < 90, 10 ** (-0.4 * MV), 0.0)


def add_companions(stars, cfg):
    mc = cfg["multiplicity"]
    m1 = stars["mass"]
    sh = stars["hash"]
    age = stars["age"]
    n = m1.size

    frac = np.zeros(n)
    for row in mc["binary_fraction_by_mass"]:
        lo, hi = row["mass_msun"]
        frac = np.where((m1 >= lo) & (m1 < hi), row["fraction"], frac)
    frac = np.where(m1 >= mc["binary_fraction_by_mass"][-1]["mass_msun"][1], mc["binary_fraction_by_mass"][-1]["fraction"], frac)
    has1 = H.uniform(sh, 20) < frac
    has2 = has1 & (H.uniform(sh, 21) < float(mc["triple_fraction_of_multiples"]))

    q0, q1 = mc["mass_ratio"]["range"]
    e0, e1 = mc["eccentricity"]["range"]
    lp_mu, lp_sig = mc["period_log10_days"]["mean"], mc["period_log10_days"]["sigma"]
    R1_au = stars["R"] * RSUN_AU

    logP_inner = None
    for comp, has, base in ((1, has1, 30), (2, has2, 50)):
        pre = f"c{comp}_"
        for f in COMP_FIELDS:
            stars[pre + f] = np.full(n, np.nan)
        idx = np.nonzero(has)[0]
        if idx.size == 0:
            continue
        h = sh[idx]
        mm1 = m1[idx]
        q = q0 + (q1 - q0) * H.uniform(h, base)
        m2 = np.maximum(q * mm1, 0.08)
        logP = np.clip(lp_mu + lp_sig * H.normal(h, base + 1), -0.6, 10.0)
        if comp == 2:
            logP = np.maximum(logP, np.log10(stars["c1_period_d"][idx]) + 1.5)   # 层级稳定：外层周期远大于内层
            mtot = mm1 + stars["c1_mass"][idx] + m2
        else:
            mtot = mm1 + m2
        ev = E.evolve_any(m2, age[idx])
        R2_au = ev["R"] * RSUN_AU
        e = e0 + (e1 - e0) * H.uniform(h, base + 2)
        P = 10 ** logP
        a = (mtot * (P / 365.25) ** 2) ** (1 / 3)
        # 不能比两颗星的半径和还近：近到接触为止（保持周期与开普勒一致）
        a_min = 2.2 * (R1_au[idx] + R2_au)
        too_close = a * (1 - e) < a_min
        e = np.where(too_close | (P < 10), 0.0, e)
        a = np.maximum(a, a_min)
        P = 365.25 * np.sqrt(a ** 3 / mtot)
        cosi = H.uniform(h, base + 3)
        theta = 2 * np.pi * H.uniform(h, base + 4)
        r = a * (1 - e * np.cos(2 * np.pi * H.uniform(h, base + 5)))   # 粗略的轨道相位距离
        proj = np.sqrt(np.cos(theta) ** 2 + (np.sin(theta) * cosi) ** 2)
        vals = {"mass": m2, "MV": ev["MV"], "Teff": ev["Teff"], "L": ev["L"], "R": ev["R"],
                "phase": ev["phase"].astype(np.float64), "period_d": P, "a_au": a, "e": e,
                "incl_deg": np.degrees(np.arccos(cosi)), "sep_au": r * proj, "pa_deg": 360 * H.uniform(h, base + 6)}
        for f, v in vals.items():
            stars[pre + f][idx] = v

    stars["multiplicity"] = (1 + has1.astype(np.int8) + has2.astype(np.int8)).astype(np.int8)
    F = _flux(stars["MV"]) + np.nan_to_num(_flux(np.nan_to_num(stars["c1_MV"], nan=99))) \
        + np.nan_to_num(_flux(np.nan_to_num(stars["c2_MV"], nan=99)))
    stars["MV_sys"] = -2.5 * np.log10(F)
    _eclipses(stars)
    return stars


def _eclipses(stars):
    n = stars["mass"].size
    stars["ecl_type"] = np.zeros(n, dtype=np.int8)       # 0 无 1 EA 2 EB 3 EW
    stars["ecl_depth_mag"] = np.zeros(n)
    k = np.nonzero(stars["multiplicity"] >= 2)[0]
    if k.size == 0:
        return
    R1 = stars["R"][k] * RSUN_AU
    R2 = stars["c1_R"][k] * RSUN_AU
    a = stars["c1_a_au"][k]
    e = stars["c1_e"][k]
    cosi = np.cos(np.radians(stars["c1_incl_deg"][k]))
    rsum = (R1 + R2) / (a * (1 - e * e))
    b = cosi / np.maximum(rsum, 1e-12)               # 碰撞参数（以半径和为单位）
    ecl = (b < 1.0) & (R2 > 0)
    F1 = _flux(stars["MV"][k])
    F2 = _flux(np.nan_to_num(stars["c1_MV"][k], nan=99))
    Ft = _flux(stars["MV_sys"][k])
    big_is_1 = R1 >= R2
    Rb, Rs = np.where(big_is_1, R1, R2), np.where(big_is_1, R2, R1)
    Fb, Fs = np.where(big_is_1, F1, F2), np.where(big_is_1, F2, F1)
    occult = Fs / Ft                                  # 小星躲到大星后面
    transit = (Rs / np.maximum(Rb, 1e-12)) ** 2 * Fb / Ft   # 小星从大星前面经过
    depth = np.maximum(occult, transit) * np.clip(1 - b * b, 0, 1)
    depth_mag = -2.5 * np.log10(np.clip(1 - depth, 1e-3, 1))
    ecl &= depth_mag >= 0.02
    typ = np.where(rsum > 0.6, 3, np.where(rsum > 0.3, 2, 1))
    stars["ecl_type"][k[ecl]] = typ[ecl]
    stars["ecl_depth_mag"][k[ecl]] = depth_mag[ecl]
