"""物理合理性检验：读 out/ 里已经生成的结果，不重新生成。

用法：
  .venv/Scripts/python.exe validate.py              # 检查 out/
  .venv/Scripts/python.exe validate.py --out 别的目录

三类检查：
  一、自洽：同一颗星的各个量之间按公式对得上（斯特藩-玻尔兹曼、星等链、开普勒、演化阶段……）。
  二、单星合理：拿真实恒星的经验关系比（主序质量-光度、各光谱型的绝对星等、白矮星、变星所在区域、周光关系……）。
  三、全天统计：和真实天空比（各星等星数、光谱型构成、肉眼变星数、银纬集中度……）。
  另外：overrides.yaml 手动改过的星，偏离演化轨迹的单独列出（只警告，不拦）。

每项给 通过 / 警告 / 不通过，结果写到 out/validation_report.md，赫罗图画到 out/validation_hr.png。
真实天空的参考数都是近似值（注明了来源量级），容许范围放得比较宽。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from starfield import evolution as E
from starfield.export import spectral_types, star_ids
from starfield.variables import VAR_TYPES

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PASS, WARN, FAIL = "通过", "警告", "不通过"
RSUN_AU = 0.00465047
VT = {t: i for i, t in enumerate(VAR_TYPES)}

# 主序星各光谱型的绝对星等（Pecaut & Mamajek 2013 表的近似值），键是光谱数值码（O=0x, B=1x, …）
DWARF_MV = np.array([
    (5, -5.5), (9, -4.3), (10, -3.9), (11, -3.2), (12, -2.4), (13, -1.6), (15, -1.2), (18, -0.25),
    (20, 1.1), (22, 1.4), (25, 2.0), (30, 2.5), (35, 3.4), (40, 4.4), (42, 4.8), (45, 5.1),
    (50, 5.9), (52, 6.3), (55, 7.3), (60, 8.9), (62, 10.0), (64, 12.0), (65, 13.3), (68, 16.0)])


def eker_logL(m):
    """主序质量-光度关系（Eker et al. 2018，分段）。"""
    lm = np.log10(m)
    return np.select(
        [m <= 0.45, m <= 0.72, m <= 1.05, m <= 2.4, m <= 7.0],
        [2.028 * lm - 0.976, 4.572 * lm - 0.102, 5.743 * lm - 0.007, 4.329 * lm + 0.010, 3.967 * lm + 0.093],
        2.865 * lm + 1.105)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail, examples=None):
        self.rows.append((section, name, status, detail, examples or []))
        mark = {PASS: "  ✓", WARN: "  !", FAIL: "  ✗"}[status]
        print(f"{mark} [{status}] {name}：{detail}")
        for e in (examples or [])[:5]:
            print(f"        {e}")

    def frac(self, section, name, bad, total, warn=0.01, fail=0.05, what="", examples=None):
        """bad 个不合格 / total 个：比例 ≤ warn 通过，≤ fail 警告，否则不通过。"""
        f = bad / max(total, 1)
        st = PASS if f <= warn else (WARN if f <= fail else FAIL)
        self.add(section, name, st, f"{bad:,} / {total:,}（{f * 100:.2f}%）{what}", examples)

    def range(self, section, name, value, lo, hi, ref, unit=""):
        st = PASS if lo <= value <= hi else WARN
        self.add(section, name, st, f"{value:g}{unit}（真实约 {ref}；容许 {lo:g}–{hi:g}）")

    def markdown(self):
        out = ["# 物理合理性检验\n"]
        n = {s: sum(r[2] == s for r in self.rows) for s in (PASS, WARN, FAIL)}
        out.append(f"通过 {n[PASS]}，警告 {n[WARN]}，不通过 {n[FAIL]}。\n")
        sec = None
        for s, name, st, detail, ex in self.rows:
            if s != sec:
                out.append(f"\n## {s}\n\n| 检查 | 结果 | 说明 |\n|---|---|---|")
                sec = s
            d = detail + ("<br>" + "<br>".join(ex[:5]) if ex else "")
            out.append(f"| {name} | {st} | {d} |")
        return "\n".join(out) + "\n"


def describe(s, i, ids, spt):
    return (f"{ids[i]} {spt[i]} V={s['V'][i]:.2f} MV={s['MV'][i]:.2f} {s['d'][i]:.1f}pc "
            f"{s['mass'][i]:.2f}M☉ {E.PHASE_NAMES[s['phase'][i]]} T={s['Teff'][i]:.0f}K L={s['L'][i]:.3g}L☉")


def worst(s, mask, score, ids, spt, k=5):
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    o = idx[np.argsort(-np.abs(score[idx]))][:k]
    return [describe(s, i, ids, spt) for i in o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out)
    s = dict(np.load(out / "stars_raw.npz"))
    if "sn_year" in s:                                     # 故事当下已爆发的星不参与
        alive = ~(s["sn_year"] <= 0)
        s = {k: v[alive] for k, v in s.items() if v.shape[:1] == alive.shape}
    n = s["V"].size
    ids, spt = star_ids(s), spectral_types(s)
    ph, m, L, T, R, MV, age = s["phase"], s["mass"], s["L"], s["Teff"], s["R"], s["MV"], s["age"]
    code, lc = E.classify(T, ph, L, m)
    ov = s.get("override", np.zeros(n, np.int8)) == 1
    rep = Report()
    print(f"检查 {out}：{n:,} 颗星（手动改过的 {int(ov.sum())} 颗）\n")

    # ================= 一、自洽 =================
    S1 = "一、自洽（同一颗星的量之间对得上）"
    print(f"== {S1} ==")
    live = L > 0
    d_sb = np.abs(np.log10(np.where(live, L, 1)) - np.log10(np.where(live, R ** 2 * (T / 5772.0) ** 4, 1)))
    rep.frac(S1, "光度 = 4πR²σT⁴", int((d_sb > 0.01).sum()), n, 0, 0, "偏差 > 0.01 dex",
             worst(s, d_sb > 0.01, d_sb, ids, spt))
    d_mv = np.abs(MV - E.abs_mag_v(L, T))
    rep.frac(S1, "绝对星等 = 由光度和热改正算出", int((d_mv > 0.01).sum()), n, 0, 0, "", worst(s, d_mv > 0.01, d_mv, ids, spt))
    d_v = np.abs(s["MV_sys"] + 5 * np.log10(s["d"]) - 5 + s["A_V"] - s["V"])
    rep.frac(S1, "视星等 = 绝对星等 + 距离模数 + 消光", int((d_v > 0.01).sum()), n, 0, 0)
    d_bv = np.abs(s["BV"] - (E.bv_from_teff(T) + s["A_V"] / 3.1))
    rep.frac(S1, "颜色 = 本征 B−V + 红化", int((d_bv > 0.02).sum()), n, 0, 0)
    pa = E.phase_at(m, age)
    bad = (pa != ph) & ~ov
    rep.frac(S1, "演化阶段 = 由质量和年龄推出的阶段", int(bad.sum()), n, 0, 0.001, "（不含手动改过的）", worst(s, bad, m, ids, spt))
    rep.frac(S1, "年龄 ≤ 宇宙年龄 13.8 Gyr", int((age > 13.8).sum()), n, 0, 0)
    has1 = np.isfinite(s["c1_mass"])
    nb = int(has1.sum())
    heavy = has1 & (s["c1_mass"] > m * 1.0001)
    rep.frac(S1, "伴星初始质量 ≤ 主星", int((heavy & ~ov).sum()), nb, 0, 0, "（不含手动改过的）")
    mt = m + np.nan_to_num(s["c1_mass"])
    P_yr = s["c1_period_d"] / 365.25
    kep = np.abs(s["c1_a_au"] ** 3 / (mt * P_yr ** 2) - 1)
    rep.frac(S1, "开普勒第三定律 a³ = M·P²", int((kep[has1] > 0.01).sum()), nb, 0, 0.001, "（内层双星，用初始质量）")
    peri = s["c1_a_au"] * (1 - s["c1_e"])
    rsum = (R + np.nan_to_num(s["c1_R"])) * RSUN_AU
    touch = has1 & (peri < rsum)
    rep.frac(S1, "双星近星点不小于两星半径和（不会撞在一起）", int(touch.sum()), nb, 0.001, 0.01,
             examples=worst(s, touch, rsum / np.maximum(peri, 1e-9), ids, spt))
    ecl = s["ecl_type"] > 0
    cosi = np.abs(np.cos(np.radians(s["c1_incl_deg"])))
    geo = ecl & (cosi > rsum / (s["c1_a_au"] * (1 - s["c1_e"] ** 2)) * 1.05)     # 合点处的距离 ≈ a(1−e²)
    rep.frac(S1, "食双星的倾角确实会发生掩食", int(geo.sum()), int(ecl.sum()), 0, 0.01)
    cl = s["cluster_id"]
    good_cl, bad_cl, ex_cl = 0, 0, []
    for c in np.unique(cl[cl >= 0]):
        k = np.nonzero(cl == c)[0]
        if k.size < 3:
            continue
        spread_age = np.ptp(age[k]) / max(np.median(age[k]), 1e-6)
        pos = np.column_stack([s["x"][k], s["y"][k], s["z"][k]])
        spread_d = np.std(np.linalg.norm(pos - pos.mean(0), axis=1)) / max(np.median(s["d"][k]), 1)
        if spread_age > 0.05 or spread_d > 0.2:
            bad_cl += 1
            ex_cl.append(f"星团 #{c}：年龄相对差 {spread_age:.2f}，空间弥散/距离 {spread_d:.2f}，成员 {k.size}")
        else:
            good_cl += 1
    rep.frac(S1, "星团成员同龄、聚在一起", bad_cl, good_cl + bad_cl, 0, 0.05, "（≥3 个可见成员的星团）", ex_cl)

    # ================= 二、单星合理 =================
    S2 = "二、单星合理（对照真实恒星的经验关系）"
    print(f"\n== {S2} ==")
    ms = (ph == E.MS) & ~ov & (m >= 0.18)
    dl = np.log10(np.where(ms, L, 1)) - eker_logL(m)
    bad = ms & (np.abs(dl) > 0.45)
    rep.frac(S2, "主序质量-光度关系（Eker 2018，±0.45 dex）", int(bad.sum()), int(ms.sum()), 0.02, 0.08,
             examples=worst(s, bad, dl, ids, spt))
    dw = (lc == 0) & ~ov
    ref = np.interp(code, DWARF_MV[:, 0], DWARF_MV[:, 1])
    dmv = MV - ref
    bad = dw & ((dmv < -1.8) | (dmv > 0.8))       # 主序会随年龄变亮约 1–1.5 等，所以亮的一侧放宽
    rep.frac(S2, "主序星的绝对星等符合光谱型（Pecaut & Mamajek，−1.8/+0.8 等）", int(bad.sum()), int(dw.sum()), 0.02, 0.08,
             examples=worst(s, bad, dmv, ids, spt))
    wd = ph == E.WD
    start, _ = E.phase_times(m)
    tcool = np.where(wd, age - start[E.WD], np.nan)
    old_wd = wd & (tcool > 0.01)
    bad = old_wd & ((MV < 9.0) | (MV > 17.5))
    rep.frac(S2, "冷却 >1000 万年的白矮星 绝对星等 9–17.5", int(bad.sum()), int(old_wd.sum()), 0, 0.1,
             examples=worst(s, bad, MV, ids, spt))
    bad = wd & ((R < 0.006) | (R > 0.025))
    rep.frac(S2, "白矮星半径 0.006–0.025 R☉", int(bad.sum()), int(wd.sum()), 0, 0.1, examples=worst(s, bad, R, ids, spt))
    bad = wd & (T > 2.5e5)
    rep.frac(S2, "白矮星温度 ≤ 25 万 K", int(bad.sum()), int(wd.sum()), 0, 0)
    star_like = ~wd & live
    bad = star_like & ((T < 2300) | (T > 60000))
    rep.frac(S2, "非白矮星温度 2300–60000 K", int(bad.sum()), int(star_like.sum()), 0.001, 0.01, examples=worst(s, bad, T, ids, spt))
    post = star_like & (ph >= E.SG) & (ph <= E.AGB)
    hd = post & (((T < 10000) & (L > 6.5e5)) | (L > 2.5e6))
    rep.frac(S2, "演化后的星不超过汉弗莱斯-戴维森极限", int(hd.sum()), int(post.sum()), 0, 0.001, examples=worst(s, hd, L, ids, spt))
    msL = star_like & (ph == E.MS) & (L > 3.5e6)
    rep.frac(S2, "主序星光度 ≤ 约 3×10⁶ L☉（已知最亮的主序星量级）", int(msL.sum()), int((ph == E.MS).sum()), 0, 0.001,
             examples=worst(s, msL, L, ids, spt))
    rgb = (ph == E.RGB) & (m <= E.M_LOW)
    bad = rgb & (L > 3500)
    rep.frac(S2, "低质量红巨星不亮过氦闪（约 2500 L☉）", int(bad.sum()), int(rgb.sum()), 0, 0.01)
    vt = s["var_type"]
    P = s["var_period_d"]

    def var_region(name, cond, desc):
        k = (vt == VT[name]) & ~ov
        bad = k & ~cond
        rep.frac(S2, f"{name} 变星落在应有的区域：{desc}", int(bad.sum()), int(k.sum()), 0, 0.05,
                 examples=worst(s, bad, T, ids, spt))
    var_region("DCEP", (ph == E.CHEB) & (T > 4800) & (T < 7200) & (m > 3), "氦燃烧、4800–7200 K、>3 M☉")
    var_region("RRAB", (T > 5800) & (T < 7600) & (L > 25) & (L < 80), "5800–7600 K、25–80 L☉")
    var_region("DSCT", (ph <= E.SG) & (T > 6200) & (T < 9000), "主序/亚巨星、6200–9000 K")
    var_region("BCEP", (ph <= E.SG) & (T > 16000) & (T < 36000), "主序/亚巨星、16000–36000 K")
    var_region("M", (ph == E.AGB) & (T < 3600), "渐近巨星支、<3600 K")
    var_region("SR", (ph >= E.RGB) & (ph <= E.AGB) & (T < 4500), "红巨星、<4500 K")
    k = (vt == VT["L"])
    hot_irr = k & (ph == E.MS) & (T > 5000) & (s["var_amp_mag"] > 0.1)
    rep.frac(S2, "不规则变星（L 型）不是平静的主序星", int(hot_irr.sum()), int(k.sum()), 0.05, 0.2,
             "（真实 L 型多是红巨星/超巨星或年轻星）", worst(s, hot_irr, T, ids, spt))
    k = (vt == VT["DCEP"]) & ~ov
    pl = MV - (-2.43 * (np.log10(np.where(k, P, 1)) - 1) - 4.05)
    rep.frac(S2, "造父变星周光关系（Benedict 2007，±0.6 等）", int((k & (np.abs(pl) > 0.6)).sum()), int(k.sum()), 0, 0.1)
    k = (vt == VT["RRAB"])
    rep.frac(S2, "天琴 RR 型绝对星等 ≈ +0.6（±0.6）", int((k & (np.abs(MV - 0.6) > 0.6)).sum()), int(k.sum()), 0, 0.1)
    k = (vt == VT["M"])
    rep.frac(S2, "米拉变星周期 80–1000 天", int((k & ((P < 80) | (P > 1000))).sum()), int(k.sum()), 0, 0.05)

    # ================= 三、全天统计 =================
    S3 = "三、全天统计（对照真实天空）"
    print(f"\n== {S3} ==")
    V = s["V"]
    for lim, real in [(1.0, 15), (3.0, 171), (5.0, 1602), (6.5, 9000), (9.0, 120000)]:
        if V.max() >= lim - 0.5:
            rep.range(S3, f"亮于 {lim} 等的星数 / 真实", round(int((V <= lim).sum()) / real, 2), 0.6, 1.5, f"{real:,} 颗 → 比值 1")
    ne = V <= 6.5
    letter = np.minimum(code // 10, 6).astype(int)
    real_frac = {"O": 0.005, "B": 0.18, "A": 0.21, "F": 0.13, "G": 0.15, "K": 0.27, "M": 0.05}   # 亮星星表（BSC）的大致构成
    for i, ch in enumerate("OBAFGKM"):
        f = float(np.mean(letter[ne & ~wd] == i))
        r = real_frac[ch]
        st = PASS if r / 2 <= f <= r * 2 else WARN
        rep.add(S3, f"肉眼星中 {ch} 型的比例", st, f"{f * 100:.1f}%（亮星星表约 {r * 100:.1f}%；容许一半到两倍）")
    gi = float(np.mean((lc[ne] >= 2) & (lc[ne] <= 5)))
    rep.range(S3, "肉眼星中巨星和超巨星的比例", round(gi * 100, 1), 30, 65, "约 45–50%", "%")
    b = s.get("gal_lat")
    if b is not None:
        f = float(np.mean(np.abs(b[ne]) < 10))
        rep.range(S3, "肉眼星落在银纬 ±10° 内的比例", round(f * 100, 1), 22, 45, "约 30%（天区面积只占 17%）", "%")
    vmax = V - np.where(vt == VT["M"], 0, 0)                 # V 已是亮时的星等
    rep.range(S3, "肉眼可见（亮时 ≤6 等）的造父变星", int(((vt == VT["DCEP"]) & (vmax <= 6)).sum()), 3, 25, "约 10 颗")
    rep.range(S3, "肉眼可见（亮时 ≤6 等）的米拉变星", int(((vt == VT["M"]) & (vmax <= 6)).sum()), 4, 40, "约 10–15 颗")
    wd_own_V = MV + 5 * np.log10(s["d"]) - 5 + s["A_V"]
    bright_wd = wd & (wd_own_V < 8.0)
    rep.add(S3, "白矮星本身亮于 8 等的数量", PASS if bright_wd.sum() <= 1 else WARN,
            f"{int(bright_wd.sum())} 颗（真实天空 0 颗：最亮的天狼 B 也只有 8.4 等）",
            [describe(s, i, ids, spt) + f" 自身V={wd_own_V[i]:.2f}" for i in np.nonzero(bright_wd)[0][:5]])
    comp_dom = np.isfinite(s["c1_MV"]) & (s["c1_MV"] < MV)
    rep.range(S3, "伴星比主星亮的系统（肉眼可见中）", int((comp_dom & ne).sum()), 0, 200,
              "主星先演化成白矮星/巨星后会出现，光谱型已写成 伴星+主星")

    # ================= 手动覆盖 =================
    S4 = "四、手动覆盖（overrides.yaml）"
    print(f"\n== {S4} ==")
    k = np.nonzero(ov)[0]
    if k.size == 0:
        rep.add(S4, "手动改过的星", PASS, "没有")
    else:
        ev = E.evolve(m[k], ph[k], age[k])
        dlog_l = np.log10(np.maximum(L[k], 1e-12)) - np.log10(np.maximum(ev["L"], 1e-12))
        dlog_t = np.log10(T[k]) - np.log10(ev["Teff"])
        off = (np.abs(dlog_l) > 0.3) | (np.abs(dlog_t) > 0.05)
        ex = [describe(s, i, ids, spt) + f"  轨道上应为 T={ev['Teff'][j]:.0f}K L={ev['L'][j]:.3g}L☉"
              for j, i in enumerate(k) if off[j]]
        hv = [describe(s, i, ids, spt) + f"  伴星 B {s['c1_mass'][i]:.2f}M☉" for i in k if heavy[i]]
        rep.add(S4, "手动改过的系统里伴星不比主星重", PASS if not hv else WARN,
                f"{len(hv)} 个系统的 B 比 A 重（同龄时更重的星会先演化，A/B 的含义就反了）", hv)
        rep.add(S4, "手动改过的星仍在演化轨迹上（|Δlog L|≤0.3、|Δlog T|≤0.05）", PASS if not off.any() else WARN,
                f"{int(off.sum())} / {k.size} 颗偏离（直接写 teff / luminosity_lsun / abs_mag 时会这样；想保持自洽请用 spectral_type 或 phase）", ex)

    (out / "validation_report.md").write_text(rep.markdown(), encoding="utf-8")
    hr_diagram(s, out / "validation_hr.png")
    nst = {x: sum(r[2] == x for r in rep.rows) for x in (PASS, WARN, FAIL)}
    print(f"\n合计：通过 {nst[PASS]}，警告 {nst[WARN]}，不通过 {nst[FAIL]}。报告 → {out / 'validation_report.md'}，赫罗图 → {out / 'validation_hr.png'}")


def hr_diagram(s, path, W=1100, H=900):
    """赫罗图：横轴 log T（左热右冷），纵轴绝对星等；灰线是真实主序（按光谱型的绝对星等）。"""
    img = Image.new("RGB", (W, H), (10, 12, 20))
    dr = ImageDraw.Draw(img)
    x0, x1, y0, y1 = 80, W - 30, 40, H - 60
    lt0, lt1, mv0, mv1 = 5.0, 3.35, -11.0, 17.0

    def X(lt):
        return x0 + (lt0 - lt) / (lt0 - lt1) * (x1 - x0)

    def Y(mv):
        return y0 + (mv - mv0) / (mv1 - mv0) * (y1 - y0)
    for mv in range(-10, 18, 2):
        dr.line([(x0, Y(mv)), (x1, Y(mv))], fill=(30, 34, 46))
        dr.text((x0 - 40, Y(mv) - 6), f"{mv:+d}", fill=(140, 150, 170))
    for T in [50000, 30000, 20000, 10000, 7000, 5000, 4000, 3000]:
        lt = np.log10(T)
        dr.line([(X(lt), y0), (X(lt), y1)], fill=(30, 34, 46))
        dr.text((X(lt) - 18, y1 + 8), f"{T // 1000}kK", fill=(140, 150, 170))
    dr.text((x0, 10), "HR diagram  (x: Teff, y: M_V)   grey: real main sequence   colors: MS/SG/RGB/CHeB/AGB/WD", fill=(200, 200, 210))
    k = s["V"] <= 9.0
    cols = [(150, 190, 255), (200, 230, 160), (255, 170, 90), (255, 220, 120), (255, 110, 110), (220, 220, 255)]
    lt = np.log10(np.clip(s["Teff"][k], 10 ** lt1, 10 ** lt0))
    mv = np.clip(s["MV"][k], mv0, mv1)
    phs = s["phase"][k]
    px = np.asarray(img).copy()
    for p in range(6):
        q = phs == p
        xs = X(lt[q]).astype(int)
        ys = Y(mv[q]).astype(int)
        ok = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        px[ys[ok], xs[ok]] = cols[p]
    img = Image.fromarray(px)
    dr = ImageDraw.Draw(img)
    T_ref = 10 ** np.interp(DWARF_MV[:, 0], E._SPT_C, np.log10(E._SPT_T))
    pts = [(X(np.log10(t)), Y(v)) for t, v in zip(T_ref, DWARF_MV[:, 1])]
    dr.line(pts, fill=(170, 170, 170), width=2)
    for p, name in enumerate(E.PHASE_NAMES):
        dr.rectangle([x1 - 120, y0 + 10 + 16 * p, x1 - 110, y0 + 20 + 16 * p], fill=cols[p])
        dr.text((x1 - 104, y0 + 8 + 16 * p), name, fill=(210, 210, 220))
    img.save(path)


if __name__ == "__main__":
    main()
