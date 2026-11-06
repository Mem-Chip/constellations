"""变星：脉动变星由演化阶段决定；食变星来自 multiplicity；另加少量不规则变星。

星表里的 V 是"最亮时"的星等；变幅表示最暗时比它暗多少。
相位零点 var_epoch_frac ∈ [0,1)：故事纪年 0 年年初时处在周期的哪个位置。
"""
import numpy as np

from . import evolution as E
from . import hashrng as H

VAR_TYPES = ["", "DCEP", "RRAB", "DSCT", "BCEP", "M", "SR", "L", "EA", "EB", "EW"]
_CODE = {t: i for i, t in enumerate(VAR_TYPES)}


def _u(sh, a):
    return H.uniform(sh, a)


def assign_pulsators(stars, cfg):
    """在双星之前调用：天琴 RR 型会改变星的温度与光度。"""
    ev = cfg["evolution"]["pulsating_variables"]
    sh = stars["hash"]
    n = sh.size
    typ = np.zeros(n, dtype=np.int8)
    P = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    ph, T, L, m, MV = stars["phase"], stars["Teff"], stars["L"], stars["mass"], stars["MV"]

    # 天琴 RR：厚盘（贫金属）的水平分支星有一部分落在不稳定带里
    k = (stars["pop"] == 1) & (ph == E.CHEB) & (m <= E.M_LOW) & (_u(sh, 70) < 0.25)
    if k.any():
        T[k] = 6200 + 800 * _u(sh[k], 71)
        L[k] = 45 + 10 * _u(sh[k], 72)
        stars["R"][k] = E.radius(L[k], T[k])
        MV[k] = E.abs_mag_v(L[k], T[k])
        typ[k] = _CODE["RRAB"]
        P[k] = 0.4 + 0.4 * _u(sh[k], 73)
        amp[k] = 0.3 + 0.9 * _u(sh[k], 74)

    # 经典造父：中等/大质量氦燃烧星穿过不稳定带。周期-光度关系
    t0, t1 = ev["cepheids"]["instability_strip_temp_k"]
    # 光度低于约 1000 L☉ 时不稳定带很窄，这里不算（银河系里周期 < 3 天的经典造父很少）
    k = (typ == 0) & (ph == E.CHEB) & (m > E.M_LOW) & (T >= t0) & (T <= t1) & (L >= 1000)
    if k.any():
        typ[k] = _CODE["DCEP"]
        P[k] = 10 ** (1 - (MV[k] + 4.05) / 2.43)
        amp[k] = 0.3 + 0.9 * _u(sh[k], 75)

    # 盾牌座 δ：不稳定带与主序的交点，变幅多半很小
    k = (typ == 0) & (ph == E.MS) & (T >= 6300) & (T <= 8600) & (m >= 1.5) & (_u(sh, 76) < 0.3)
    if k.any():
        typ[k] = _CODE["DSCT"]
        P[k] = 0.02 + 0.23 * _u(sh[k], 77)
        amp[k] = np.exp(np.log(0.003) + (np.log(0.3) - np.log(0.003)) * _u(sh[k], 78))

    # 仙王座 β：早 B 型主序星，小幅快速
    k = (typ == 0) & (ph == E.MS) & (m >= 8) & (m <= 20) & (_u(sh, 79) < 0.3)
    if k.any():
        typ[k] = _CODE["BCEP"]
        P[k] = 0.1 + 0.2 * _u(sh[k], 80)
        amp[k] = 0.01 + 0.09 * _u(sh[k], 81)

    # 米拉：渐近巨星支末端
    mira = ev["long_period_mira"]
    k = (typ == 0) & (ph == E.AGB) & (T < 3300) & (L > 2000)
    if k.any():
        p0, p1 = mira["period_days"]
        x = np.clip((np.log10(L[k]) - 3.3) / 1.2, 0, 1)
        P[k] = np.exp(np.log(p0) + (np.log(p1) - np.log(p0)) * np.clip(x + 0.15 * (_u(sh[k], 82) - 0.5), 0, 1))
        a0, a1 = mira["amplitude_mag"]
        amp[k] = a0 + (a1 - a0) * _u(sh[k], 83) ** 1.5
        typ[k] = _CODE["M"]

    # 半规则：亮红巨星的一部分
    sr = ev["semiregular_giants"]
    k = (typ == 0) & ((ph == E.RGB) | (ph == E.AGB)) & (L > 300) & (T < 4100) & (_u(sh, 84) < sr["fraction_of_red_giants"])
    if k.any():
        typ[k] = _CODE["SR"]
        P[k] = 20 + 180 * np.clip(np.log10(L[k] / 300) / 1.5, 0, 1) * (0.6 + 0.8 * _u(sh[k], 85))
        amp[k] = 0.1 + 1.0 * _u(sh[k], 86) ** 2

    # 额外的不规则变星
    ir = cfg["evolution"]["irregular_extra"]
    # 只给演化后的冷巨星/超巨星（真实的 Lb、Lc 型），不给平静的主序星
    k = (typ == 0) & (ph >= E.RGB) & (ph <= E.AGB) & (T < 5000) & (_u(sh, 87) < ir["fraction_of_visible"])
    if k.any():
        typ[k] = _CODE["L"]
        a0, a1 = ir["amplitude_mag"]
        amp[k] = a0 + (a1 - a0) * _u(sh[k], 88)

    stars["var_type"] = typ
    stars["var_period_d"] = P
    stars["var_amp_mag"] = amp
    stars["var_epoch_frac"] = _u(sh, 89)
    return stars


def assign_eclipsing(stars):
    """在双星之后调用：没有其他变光类型的食双星记为 EA/EB/EW。"""
    k = (stars["var_type"] == 0) & (stars["ecl_type"] > 0)
    stars["var_type"][k] = (_CODE["EA"] - 1 + stars["ecl_type"][k]).astype(np.int8)
    stars["var_period_d"][k] = stars["c1_period_d"][k]
    stars["var_amp_mag"][k] = stars["ecl_depth_mag"][k]
    return stars
