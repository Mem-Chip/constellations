"""手动覆盖层（overrides.yaml）：在自动生成和地标之后，按编号改单颗星。

每一项是一个操作：

  - id: 39f54c7cf6c6-000          # 改一颗星（op 默认是 modify）
    set:
      name: 心宿二                  # 星名
      spectral_type: M1.5Iab       # 按光谱型反推质量、阶段、年龄（用同一套演化模型，所以温度、光度、半径都自洽）
      mass_msun: 15                # 质量；和 spectral_type 一起给时，作为反推的偏好
      phase: CHeB                  # 演化阶段 MS/SG/RGB/CHeB/AGB/WD，可配 phase_frac（0–1，阶段内进度）
      age_myr: 12                  # 年龄（和 mass_msun 一起重算演化）
      teff: 3600                   # 直接改表面温度（光度不变，半径跟着变）
      luminosity_lsun: 60000       # 直接改光度
      abs_mag: -5.2                # 直接改绝对星等
      distance_pc: 170             # 改距离（方向不变）
      gal_lon_lat_deg: [351.9, 15.1]   # 改方向（距离不变）
      apparent_mag: 1.0            # 改到这个视星等：沿视线移动距离（算上消光）
  - id: 39f54c7cf6c6-000           # 指定这颗星在某年爆发超新星
    op: supernova
    year: 812                      # 故事纪年；可以在 0 年之前（星已不在，留下遗迹）或之后（现在还在天上）
    day: 140                       # 可选，第几天
    peak_abs_mag: -17.5            # 可选，峰值绝对星等；默认按配置（白矮星是 Ia 型，其余是核坍缩）
  - id: c9f0e1a2b3c4-003
    op: remove
  - op: add
    star: {gal_lon_lat_deg: [10, 5], distance_pc: 4.2, spectral_type: M5V, name: 近邻}

改动后重算演化、脉动变星、食变星、合成亮度和消光。一个 set 里的多个键按上面的顺序生效：
先定位置，再定物理（光谱型 / 阶段 / 质量年龄），再直接改温度 / 光度 / 绝对星等，最后按视星等调距离。
"""
import os
import re

import numpy as np

from . import evolution as E
from .coords import lonlat, unit
from .landmarks import _phase_mid_age, concat, make_star
from .observe import add_extinction
from .variables import assign_eclipsing, assign_pulsators

PHASE = {n: i for i, n in enumerate(E.PHASE_NAMES)}
LUM_PHASES = {"V": [E.MS], "IV": [E.SG], "III": [E.RGB, E.CHEB, E.AGB], "II": [E.CHEB, E.AGB, E.SG],
              "Ib": [E.CHEB, E.SG, E.AGB], "Iab": [E.CHEB, E.SG], "Ia": [E.CHEB, E.SG], "Ia+": [E.CHEB], "D": [E.WD]}
LUM_CODE = {"V": [0], "IV": [1], "III": [2], "II": [3], "Ib": [4], "Iab": [4, 5], "Ia": [5], "Ia+": [5], "D": [6]}
AGE_MAX_GYR = 13.0
SN_FIELDS = ("sn_year", "sn_day", "sn_peak_abs")


def parse_spectral_type(s):
    """'M1.5Iab' → (数值码 61.5, 光度级 'Iab')；'DA' → (None, 'D')；没写光度级当作 V。"""
    s = str(s).strip().replace(" ", "")
    if s.upper().startswith("D"):
        return None, "D"
    m = re.match(r"^([OBAFGKM])(\d+(?:\.\d+)?)?(Ia\+|Iab|Ia|Ib|III|II|IV|V|I)?", s)
    if not m:
        raise ValueError(f"看不懂的光谱型：{s}")
    code = "OBAFGKM".index(m.group(1)) * 10 + float(m.group(2) or 5)
    lum = m.group(3) or "V"
    if lum == "I":
        lum = "Iab"
    return code, lum


def solve_spectral_type(spt, mass_hint=None):
    """在 质量 × 阶段 × 阶段内进度 的网格里找一颗演化后最像这个光谱型的星，返回 (质量, 阶段, 年龄 Gyr)。"""
    code, lum = parse_spectral_type(spt)
    phases = LUM_PHASES[lum]
    want_lc = LUM_CODE[lum]
    masses = np.geomspace(0.1, 100, 260)
    fr = np.linspace(0.02, 0.98, 25)
    M, P, F = [a.ravel() for a in np.meshgrid(masses, phases, fr, indexing="ij")]
    start, end = E.phase_times(M)
    idx = np.arange(M.size)
    s, e = start[P, idx], end[P, idx]
    ok = np.isfinite(s)
    with np.errstate(invalid="ignore"):
        age = np.where(np.isfinite(e), s + F * (e - s), s + F * 5.0)   # 白矮星：冷却 0–5 Gyr
    M, P, age = M[ok], P[ok], age[ok]
    ev = E.evolve(M, P, age)
    c, lc = E.classify(ev["Teff"], P, ev["L"], M)
    score = 5.0 * ~np.isin(lc, want_lc) + 3.0 * (age > AGE_MAX_GYR) + 0.02 * F[ok]
    if code is not None:
        score = score + np.abs(c - code)
    if mass_hint is not None:
        score = score + 2.0 * np.abs(np.log(M / float(mass_hint)))
    score = np.where(np.isfinite(score) & np.isfinite(ev["Teff"]) & np.isfinite(ev["L"]), score, np.inf)
    j = int(np.argmin(score))
    return float(M[j]), int(P[j]), float(age[j])


def _solve_distance(MV, u, V_target, dust):
    d = 10 ** ((V_target - MV + 5) / 5)
    for _ in range(5):
        A = dust.extinction(u[None, :], np.array([d]))[0]
        d = 10 ** ((V_target - MV - A + 5) / 5)
    return float(d)


def _recompute(stars, t, cfg, dust):
    """改过物理量的星：重算脉动变星、食变星、合成亮度、消光。"""
    sub = {k: v[t].copy() for k, v in stars.items() if isinstance(v, np.ndarray) and v.shape[:1] == stars["x"].shape}
    saved_ecl = sub.get("ecl_type")
    assign_pulsators(sub, cfg)
    if saved_ecl is not None:
        assign_eclipsing(sub)
    F = 10 ** (-0.4 * sub["MV"])
    for c in ("c1_MV", "c2_MV"):
        if c in sub:
            F = F + np.where(np.isfinite(sub[c]) & (sub[c] < 90), 10 ** (-0.4 * np.nan_to_num(sub[c], nan=99)), 0)
    sub["MV_sys"] = -2.5 * np.log10(F)
    add_extinction(sub, dust)
    for k, v in sub.items():
        if k in stars and isinstance(stars[k], np.ndarray) and stars[k].shape[:1] == stars["x"].shape:
            stars[k][t] = v


def _set_physics(stars, j, mass, phase, age):
    stars["mass"][j], stars["phase"][j], stars["age"][j] = mass, phase, age
    ev = E.evolve(np.array([mass]), np.array([phase]), np.array([age]))
    for c in ("L", "Teff", "R", "f"):
        stars[c][j] = ev[c][0]
    stars["MV"][j] = E.abs_mag_v(ev["L"], ev["Teff"])[0]


RSUN_AU = 0.00465047


def solve_spectral_type_at_age(spt, age, mass_hint=None):
    """伴星用：年龄已由系统定死，只在质量上找最像这个光谱型的星。"""
    code, lum = parse_spectral_type(spt)
    want_lc = LUM_CODE[lum]
    M = np.geomspace(0.08, 100, 1500)
    ev = E.evolve_any(M, np.full(M.size, age))
    ph = ev["phase"]
    ok = ph >= 0
    c, lc = E.classify(ev["Teff"], np.maximum(ph, 0), ev["L"], M)
    score = 5.0 * ~np.isin(lc, want_lc) + (np.abs(c - code) if code is not None else 0.0)
    if mass_hint is not None:
        score = score + 2.0 * np.abs(np.log(M / float(mass_hint)))
    score = np.where(ok & np.isfinite(score), score, np.inf)
    return float(M[int(np.argmin(score))])


def _set_companion(stars, j, c, mass):
    """改伴星质量：按系统年龄重算演化；周期不变，按开普勒重算轨道半长轴（太近就推到接触为止）。"""
    p = f"c{c}_"
    ev = E.evolve_any(np.array([mass]), np.array([stars["age"][j]]))
    stars[p + "mass"][j] = mass
    for f in ("L", "Teff", "R", "MV"):
        stars[p + f][j] = ev[f][0]
    stars[p + "phase"][j] = float(ev["phase"][0])
    _refresh_orbits(stars, j)


def _refresh_orbits(stars, j):
    """质量或半径改了以后让轨道仍满足开普勒定律、两星不相撞。"""
    for c in (1, 2):
        p = f"c{c}_"
        if not np.isfinite(stars[p + "mass"][j]):
            continue
        mt = stars["mass"][j] + stars["c1_mass"][j] + (stars["c2_mass"][j] if c == 2 else 0.0)
        P = stars[p + "period_d"][j]
        a_old = stars[p + "a_au"][j]
        a = (mt * (P / 365.25) ** 2) ** (1 / 3)
        rin = stars["R"][j] + (np.nan_to_num(stars["c1_R"][j]) if c == 2 else 0.0)
        a_min = 2.2 * (rin + np.nan_to_num(stars[p + "R"][j])) * RSUN_AU
        if a * (1 - stars[p + "e"][j]) < a_min:
            stars[p + "e"][j] = 0.0
            a = max(a, a_min)
            stars[p + "period_d"][j] = 365.25 * np.sqrt(a ** 3 / mt)
        stars[p + "a_au"][j] = a
        stars[p + "sep_au"][j] *= a / a_old


def _remove_companion(stars, j, c):
    from .multiplicity import COMP_FIELDS
    if c == 1 and np.isfinite(stars["c2_mass"][j]):          # 删 B 时让 C 补成新的 B
        for f in COMP_FIELDS:
            stars["c1_" + f][j] = stars["c2_" + f][j]
        c = 2
    for f in COMP_FIELDS:
        stars[f"c{c}_" + f][j] = np.nan
    stars["multiplicity"][j] -= 1


def _move(stars, j, u, d):
    p = np.asarray(u) * d
    stars["x"][j], stars["y"][j], stars["z"][j], stars["d"][j] = p[0], p[1], p[2], d


def load_ops(cfg):
    import yaml
    fn = (cfg.get("overrides") or {}).get("file")
    if not fn:
        return []
    path = os.path.join(os.path.dirname(os.path.abspath(cfg["_path"])), fn)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or []


def apply_overrides(stars, names, cfg, galaxy, dust, log=print):
    ops = load_ops(cfg)
    n = stars["x"].size
    for k in SN_FIELDS:
        if k not in stars:
            stars[k] = np.full(n, np.nan)
    stars["override"] = np.zeros(n, np.int8)          # 1 = 物理量被手动改过（验证时单独检查）
    if not ops:
        return stars, names
    key = {(int(h), int(i)): j for j, (h, i) in enumerate(zip(stars["id_hi"], stars["id_idx"]))}

    def find(sid):
        """'xxxx-NNN' → (行, None)；'xxxx-NNNA/B/C' → (行, 成员 0/1/2)。"""
        try:
            sid = str(sid).strip()
            c = None
            if sid[-1] in "ABC":
                c, sid = "ABC".index(sid[-1]), sid[:-1]
            hi, idx = sid.split("-")
            j = key.get((int(hi, 16), int(idx)))
            if j is not None and c is not None and c > 0 and not np.isfinite(stars[f"c{c}_mass"][j]):
                return None, None
            return j, c
        except (ValueError, IndexError):
            return None, None

    remove = np.zeros(n, bool)
    touched, added, n_sn = [], [], 0
    for n_op, op in enumerate(ops):
        kind = op.get("op", "modify")
        if kind == "add":
            sp = dict(op["star"])
            if "spectral_type" in sp:
                m, ph, age = solve_spectral_type(sp["spectral_type"], sp.get("mass_msun"))
            else:
                m = float(sp.get("mass_msun", 1.0))
                ph = PHASE.get(sp.get("phase", "MS"), E.MS)
                age = float(sp["age_gyr"]) if "age_gyr" in sp else _phase_mid_age(m, ph, float(sp.get("phase_frac", 0.5)))
            l, b = sp["gal_lon_lat_deg"]
            s = make_star(cfg, galaxy, dust, 5000 + n_op, m, ph, age, l, b,
                          d=sp.get("distance_pc"), V_target=sp.get("apparent_mag"))
            s.pop("_name", None)
            if sp.get("name"):
                names[(int(s["id_hi"][0]), 0)] = sp["name"]
            for k in SN_FIELDS:
                s[k] = np.full(1, np.nan)
            s["override"] = np.ones(1, np.int8)
            added.append(s)
            log(f"  [覆盖] 新增 {sp.get('name', '')} {E.spectral_string(*E.classify(s['Teff'], s['phase'], s['L'], s['mass']))[0]}"
                f" V={s['V'][0]:.2f} {s['d'][0]:.1f} pc")
            continue
        j, comp = find(op.get("id"))
        if j is None:
            log(f"  [覆盖] 找不到 {op.get('id')}，跳过")
            continue
        if comp in (1, 2):                                    # 改伴星
            hk = (int(stars["id_hi"][j]), int(stars["id_idx"][j]))
            st = op.get("set", {}) or {}
            if kind == "remove":
                _remove_companion(stars, j, comp)
                touched.append(j)
                continue
            if kind != "modify":
                log(f"  [覆盖] {op['id']}：伴星只支持改名、改质量 / 光谱型、删除，跳过 {kind}")
                continue
            if "name" in st:
                names[hk + (comp,)] = st["name"]
            m_new = None
            if "spectral_type" in st:
                m_new = solve_spectral_type_at_age(st["spectral_type"], stars["age"][j], st.get("mass_msun"))
            elif "mass_msun" in st:
                m_new = float(st["mass_msun"])
            if m_new is not None:
                if m_new > stars["mass"][j]:
                    log(f"  [覆盖] {op['id']}：伴星 {m_new:.2f} M☉ 比主星 {stars['mass'][j]:.2f} M☉ 还重（同龄的星里它会先演化）")
                _set_companion(stars, j, comp, m_new)
                touched.append(j)
            other = set(st) - {"name", "spectral_type", "mass_msun"}
            if other:
                log(f"  [覆盖] {op['id']}：伴星不支持 {sorted(other)}（年龄、位置、距离跟随整个系统），忽略")
            continue
        if comp == 0:
            st0 = op.get("set", {}) or {}
            if "name" in st0:
                names[(int(stars["id_hi"][j]), int(stars["id_idx"][j]), 0)] = st0["name"]
        if kind == "remove":
            remove[j] = True
            continue
        if kind == "supernova":
            stars["sn_year"][j] = float(op["year"])
            stars["sn_day"][j] = float(op.get("day", 0))
            stars["sn_peak_abs"][j] = float(op["peak_abs_mag"]) if "peak_abs_mag" in op else np.nan
            n_sn += 1
            continue
        st = dict(op.get("set", {}) or {})
        if "name" in st and comp == 0:
            st.pop("name")                                     # xxxA 的名字只给主星，上面已记
        if "name" in st:
            names[(int(stars["id_hi"][j]), int(stars["id_idx"][j]))] = st["name"]
        u = np.array([stars["x"][j], stars["y"][j], stars["z"][j]]) / stars["d"][j]
        if "gal_lon_lat_deg" in st:
            u = unit(*st["gal_lon_lat_deg"])
            _move(stars, j, u, stars["d"][j])
        if "distance_pc" in st:
            _move(stars, j, u, float(st["distance_pc"]))
        if "spectral_type" in st:
            m, ph, age = solve_spectral_type(st["spectral_type"], st.get("mass_msun"))
            _set_physics(stars, j, m, ph, age)
        elif "phase" in st:
            m = float(st.get("mass_msun", stars["mass"][j]))
            ph = PHASE[st["phase"]]
            _set_physics(stars, j, m, ph, _phase_mid_age(m, ph, float(st.get("phase_frac", 0.5))))
        elif "mass_msun" in st or "age_myr" in st:
            m = float(st.get("mass_msun", stars["mass"][j]))
            age = float(st["age_myr"]) / 1000 if "age_myr" in st else stars["age"][j]
            ph = int(E.phase_at(np.array([m]), np.array([age]))[0])
            if ph < 0:
                log(f"  [覆盖] {op['id']}：{m} M☉、{age * 1000:.0f} Myr 的星已经死亡，改成白矮星以外的阶段请换年龄")
                ph = E.WD
            _set_physics(stars, j, m, ph, age)
        if "teff" in st:
            stars["Teff"][j] = float(st["teff"])
            stars["R"][j] = E.radius(stars["L"][j], stars["Teff"][j])
            stars["MV"][j] = E.abs_mag_v(np.array([stars["L"][j]]), np.array([stars["Teff"][j]]))[0]
        if "luminosity_lsun" in st:
            stars["L"][j] = float(st["luminosity_lsun"])
            stars["R"][j] = E.radius(stars["L"][j], stars["Teff"][j])
            stars["MV"][j] = E.abs_mag_v(np.array([stars["L"][j]]), np.array([stars["Teff"][j]]))[0]
        if "abs_mag" in st:
            dm = float(st["abs_mag"]) - stars["MV"][j]
            stars["MV"][j] += dm
            stars["L"][j] *= 10 ** (-0.4 * dm)
        touched.append(j)
        if "apparent_mag" in st:
            _recompute(stars, np.array([j]), cfg, dust)
            _move(stars, j, u, _solve_distance(float(stars["MV_sys"][j]), u, float(st["apparent_mag"]), dust))
    if touched:
        for j in set(touched):
            _refresh_orbits(stars, j)
        _recompute(stars, np.array(sorted(set(touched))), cfg, dust)
        stars["override"][np.array(sorted(set(touched)))] = 1
    if remove.any():
        stars = {k: (v[~remove] if isinstance(v, np.ndarray) and v.shape[:1] == remove.shape else v) for k, v in stars.items()}
    if added:
        stars = concat([stars] + added)
    log(f"  [覆盖] 修改 {len(set(touched))}，删除 {int(remove.sum())}，新增 {len(added)}，指定超新星 {n_sn}")
    return stars, names
