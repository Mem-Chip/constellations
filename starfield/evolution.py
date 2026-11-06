"""简化演化轨迹：质量 + 年龄 → 演化阶段 → 光度、温度、半径、绝对星等、颜色、光谱型。

阶段编号：
  0 MS   主序
  1 SG   亚巨星 / 赫氏空隙
  2 RGB  红巨星支（只有 ≤2.2 M☉ 的星）
  3 CHeB 核心氦燃烧（红团簇、蓝回环、超巨星）
  4 AGB  渐近巨星支（≤8 M☉）
  5 WD   白矮星（≤8 M☉ 的归宿；更重的星成为中子星/黑洞，不发光，不生成）

时间单位 Gyr。所有函数都对数组向量化。
"""
import numpy as np

MS, SG, RGB, CHEB, AGB, WD = range(6)
PHASE_NAMES = ["MS", "SG", "RGB", "CHeB", "AGB", "WD"]
M_LOW = 2.2      # 低质量星（有氦闪、有红巨星支）的上限
M_MASSIVE = 8.0  # 超过它就成为超巨星，最后超新星
T_SUN = 5772.0
MBOL_SUN = 4.74


def zams(m):
    """参考主序光度、半径（大约是主序中段的值）。"""
    m = np.asarray(m, dtype=np.float64)
    L = np.where(m < 0.43, 0.23 * m ** 2.3,
        np.where(m < 2.0, m ** 4.0,
        np.where(m < 20.0, 1.4 * m ** 3.5, 5.0e4 * (m / 20.0) ** 2.2)))
    R = np.where(m < 1.0, m ** 0.8, m ** 0.57)
    return L, R


# 主序寿命 (Gyr)：按 Schaller 等 (1992) 一类模型的数值在对数-对数里插值
_TMS_M = np.log10([0.08, 0.5, 1.0, 2.0, 3.0, 5.0, 9.0, 15.0, 25.0, 40.0, 60.0, 120.0])
_TMS_T = np.log10([4.0e3, 1.0e2, 10.0, 1.16, 0.35, 0.094, 0.026, 0.0116, 0.0064, 0.0043, 0.0034, 0.0026])


def t_ms(m):
    lm = np.log10(np.asarray(m, dtype=np.float64))
    return 10.0 ** np.interp(lm, _TMS_M, _TMS_T)


def teff(L, R):
    return T_SUN * (L / R ** 2) ** 0.25


def radius(L, T):
    return np.sqrt(L) * (T_SUN / T) ** 2


def phase_times(m):
    """返回 start, end 两个 (6, N) 数组：每个阶段的开始/结束年龄。"""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    t0 = t_ms(m)
    low = m <= M_LOW
    massive = m > M_MASSIVE
    d_sg = t0 * (0.01 + 0.07 * np.clip((1.4 - m) / 0.4, 0.0, 1.0))
    d_rgb = np.where(low, 0.06 * t0, 0.0)
    d_cheb = np.where(low, np.minimum(0.2 * t0, 0.12), np.where(massive, 0.1 * t0, 0.2 * t0))
    d_agb = np.where(massive, 0.0, 0.15 * d_cheb)
    dur = np.stack([t0, d_sg, d_rgb, d_cheb, d_agb])
    start = np.zeros((6, m.size))
    end = np.zeros((6, m.size))
    t = np.zeros(m.size)
    for p in range(5):
        start[p] = t
        t = t + dur[p]
        end[p] = t
    start[WD] = np.where(massive, np.inf, t)
    end[WD] = np.inf
    return start, end


def phase_at(m, age):
    """给定质量与年龄，返回所处阶段；超过寿命的大质量星（中子星/黑洞）返回 -1。"""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    age = np.atleast_1d(np.asarray(age, dtype=np.float64))
    start, end = phase_times(m)
    ph = np.zeros(m.size, dtype=np.int8)
    for p in range(1, 6):
        ph = np.where(age >= start[p], p, ph).astype(np.int8)
    dead = (m > M_MASSIVE) & (age >= end[CHEB])
    ph[dead] = -1
    return ph


def evolve_any(m, age):
    """不知道阶段时：先判断阶段再演化。死掉的星光度为 0、绝对星等 99。"""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    age = np.atleast_1d(np.asarray(age, dtype=np.float64))
    ph = phase_at(m, age)
    out = {"L": np.zeros(m.size), "Teff": np.full(m.size, 3000.0), "R": np.zeros(m.size),
           "f": np.zeros(m.size), "phase": ph, "MV": np.full(m.size, 99.0)}
    k = ph >= 0
    if k.any():
        ev = evolve(m[k], ph[k], age[k])
        for key in ("L", "Teff", "R", "f"):
            out[key][k] = ev[key]
        out["MV"][k] = abs_mag_v(ev["L"], ev["Teff"])
    return out


def _lerp(a, b, f):
    return a + (b - a) * f


def _climb(L0, L1, f, k):
    """光度从 L0 爬到 L1，停留时间满足 N(>L) ∝ (L/L0)^-k；f=0→L0，f=1→L1。"""
    eps = (L1 / L0) ** -k
    return L0 * (1.0 - f * (1.0 - eps)) ** (-1.0 / k)


def evolve(m, phase, age):
    """返回 dict：L, Teff, R（太阳单位）以及 f（阶段内进度）。"""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    phase = np.atleast_1d(np.asarray(phase))
    age = np.atleast_1d(np.asarray(age, dtype=np.float64))
    n = m.size
    start, end = phase_times(m)
    idx = np.arange(n)
    s = start[phase, idx]
    e = end[phase, idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.clip(np.where(np.isfinite(e), (age - s) / (e - s), 0.0), 0.0, 1.0)

    Lr, Rr = zams(m)
    # 主序期间光度涨 gL 倍、半径涨 gR 倍。大质量星半径涨得比光度快，所以越老越冷（5 M☉ 约 17000 → 13500 K）；
    # 太阳附近的星几乎不变温。数值按 Geneva / Schaller 一类模型的零龄与终龄主序估计。
    lm = np.log10(m)
    gL = np.interp(lm, np.log10([0.5, 1.0, 3.0, 20.0, 60.0]), [1.6, 2.6, 2.5, 2.2, 1.9])
    gR = np.interp(lm, np.log10([0.5, 1.0, 1.5, 3.0, 10.0]), [1.15, 1.45, 1.9, 2.5, 2.7])
    L_tams = Lr * 0.7 * gL
    R_tams = Rr * 0.88 * gR
    T_tams = teff(L_tams, R_tams)
    low = m <= M_LOW
    massive = m > M_MASSIVE
    very_massive = m >= 25.0

    L = np.empty(n)
    T = np.empty(n)

    # 主序
    k = phase == MS
    L[k] = np.minimum(Lr[k] * 0.7 * gL[k] ** f[k], 3e6)
    T[k] = teff(L[k], Rr[k] * 0.88 * gR[k] ** f[k])

    # 亚巨星 / 赫氏空隙
    T_sg_end = np.where(low, 5000.0, np.where(~massive, 4300.0, np.where(very_massive, T_tams * 0.7, 3800.0)))
    L_sg_end = np.where(low, 1.3 * L_tams, np.where(massive, L_tams * 1.2, 1.5 * L_tams))
    k = phase == SG
    L[k] = np.exp(_lerp(np.log(L_tams[k]), np.log(L_sg_end[k]), f[k]))
    T[k] = np.exp(_lerp(np.log(T_tams[k]), np.log(T_sg_end[k]), f[k]))

    # 红巨星支：越往上越快，停留时间 N(>L) ∝ L^-0.85；越亮越冷
    L_tip = 2500.0
    k = phase == RGB
    L0 = L_sg_end[k]
    L1 = np.maximum(L_tip, L0 * 2)
    L[k] = _climb(L0, L1, f[k], 0.85)
    x = np.log(L[k] / L0) / np.log(L1 / L0)
    T[k] = np.exp(_lerp(np.log(5000.0), np.log(3700.0), x ** 1.5))

    # 核心氦燃烧
    L_cheb0 = np.where(low, np.maximum(45.0, 1.56 * L_tams),
              np.where(massive, 1.5 * L_tams, 1.8 * L_tams))
    L_cheb1 = L_cheb0 * np.where(massive, 1.15, 1.4)
    k = phase == CHEB
    L[k] = _lerp(L_cheb0[k], L_cheb1[k], f[k])
    # 低质量：红团簇
    T_cheb = 4900.0 - 200.0 * f
    # 中等质量：蓝回环。先在红端，快速穿过黄色区（不稳定带）到蓝端停留，再快速回到红端
    T_red = np.where(massive, 3700.0, 4200.0)
    # 回环约从 3.6 M☉ 起才伸进不稳定带，并且随质量很快伸过去（否则蓝端停在带里的造父太多）
    T_blue = np.where(massive, 3700.0 + np.maximum(0.0, 12.0 - m) * 4000.0,
                      np.clip(4200.0 + 8000.0 * (m - 3.8), 4200.0, 14000.0))
    loop = np.interp(f, [0.0, 0.25, 0.29, 0.82, 0.86, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    T_loop = np.exp(_lerp(np.log(T_red), np.log(T_blue), loop))
    # 很重的星（≥25 M☉，强星风）：大部分时间是蓝超巨星
    T_hot = np.exp(_lerp(np.log(T_tams * 0.6), np.log(np.maximum(9000.0, T_tams * 0.35)), f))
    T_cheb = np.where(low, T_cheb, np.where(very_massive, T_hot, T_loop))
    T[k] = T_cheb[k]

    # 渐近巨星支
    L_agb_tip = np.minimum(6e4, 5000.0 * m)
    k = phase == AGB
    L0 = L_cheb1[k]
    L1 = np.maximum(L_agb_tip[k], L0 * 1.5)
    L[k] = _climb(L0, L1, f[k], 1.0)
    x = np.log(L[k] / L0) / np.log(L1 / L0)
    T[k] = np.exp(_lerp(np.log(np.where(low[k], 4200.0, T_red[k])), np.log(2900.0), x ** 2))

    # 白矮星：按冷却年龄变暗
    k = phase == WD
    t_cool = np.maximum(age[k] - s[k], 0.0)
    m_wd = 0.109 * m[k] + 0.394
    L[k] = np.minimum(50.0, 2e-3 * (t_cool + 1e-4) ** -1.25)
    R_wd = 0.0125 * (m_wd / 0.6) ** (-1.0 / 3.0)
    T[k] = teff(L[k], R_wd)

    # 汉弗莱斯-戴维森极限：演化后的星不超过约 2e6 L☉，冷超巨星不超过约 6e5 L☉
    post = (phase >= SG) & (phase <= AGB)
    L = np.where(post, np.minimum(L, np.where(T < 10000.0, 6e5, 2e6)), L)

    R = radius(L, T)
    return {"L": L, "Teff": T, "R": R, "f": f}


# ---------------- 星等与颜色 ----------------
_BC_COEF = [
    (3.70, [-0.190537291496456e5, 0.155144866764412e5, -0.421278819301717e4, 0.381476328422343e3]),
    (3.90, [-0.370510203809015e5, 0.385672629965804e5, -0.150651486316025e5, 0.261724637119416e4, -0.170623810323864e3]),
    (9.99, [-0.118115450538963e6, 0.137145973583929e6, -0.636233812100225e5, 0.147412923562646e5,
            -0.170587278406872e4, 0.788731721804990e2]),
]


_BC_COOL_T = np.array([2400.0, 2500.0, 2800.0, 3000.0, 3200.0, 3500.0])
_BC_COOL_V = np.array([-6.0, -5.5, -4.3, -3.5, -2.8, -1.9])


def bc_v(T):
    """V 波段热改正。3700–40000 K 用 Torres (2010) 多项式；更冷的星用表格插值（多项式在那里发散）；
    更热的星（O 型、刚诞生的白矮星）用 BC ≈ 27.66 − 6.84·log T（Martins 2005 一类的热星关系，在 4 万 K 处与多项式衔接）。"""
    T = np.asarray(T, dtype=np.float64)
    lt = np.clip(np.log10(T), 3.40, 4.70)
    out = np.empty(lt.shape)
    lo = -np.inf
    for hi, c in _BC_COEF:
        k = (lt >= lo) & (lt < hi)
        x = lt[k]
        out[k] = sum(ci * x ** i for i, ci in enumerate(c))
        lo = hi
    cool = T < 3700.0
    if cool.any():
        bc3700 = sum(ci * np.log10(3700.0) ** i for i, ci in enumerate(_BC_COEF[0][1]))
        out[cool] = np.interp(T[cool], np.append(_BC_COOL_T, 3700.0), np.append(_BC_COOL_V, bc3700))
    hot = T > 40000.0
    if hot.any():
        out[hot] = 27.66 - 6.84 * np.log10(np.minimum(T[hot], 3e5))
    return out


def abs_mag_v(L, T):
    return MBOL_SUN - 2.5 * np.log10(L) - bc_v(T)


def bv_from_teff(T):
    """Ballesteros (2012) 公式反解。"""
    k = np.asarray(T, dtype=np.float64) / 4600.0
    a = k
    b = 2.32 * k - 2.0
    c = 1.054 * k - 2.32
    x = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))) / (2 * a)
    return np.clip(x / 0.92, -0.33, 2.2)


# 光谱型：数值码 = 字母序号*10 + 次型；O=0 B=1 A=2 F=3 G=4 K=5 M=6
_SPT_T = np.array([42000, 31500, 31000, 20600, 15700, 12500, 9700, 8100, 7200, 6500, 5900, 5600, 5200, 4400, 3850, 3500, 3000, 2600])
_SPT_C = np.array([5, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 68], dtype=np.float64)
_LETTERS = "OBAFGKM"
LUMCLASS = ["V", "IV", "III", "II", "Ib", "Ia", "D"]


# 巨星的温度标度：同一温度下比矮星早型（K0III ≈ 4800 K，而 K0V ≈ 5200 K）
_SPT_TG = np.array([42000, 31500, 30000, 18000, 14000, 11500, 9700, 8100, 7200, 6400, 5600, 5100, 4950, 4800, 4500, 3950, 3850, 3700, 3400, 3150, 2900])
_SPT_CG = np.array([5, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 48, 50, 52, 55, 60, 62, 65, 67, 69], dtype=np.float64)


def spectral_code(T, giant=None):
    lt = np.log10(np.clip(T, 2400, 50000))
    c = np.interp(-lt, -np.log10(_SPT_T), _SPT_C)
    if giant is not None:
        g = np.asarray(giant, dtype=bool)
        c = np.where(g, np.interp(-lt, -np.log10(_SPT_TG), _SPT_CG), c)
    return c


def classify(T, phase, L, m):
    """返回 (光谱数值码, 光度级码)。"""
    lc = lum_class(phase, L, m)
    return spectral_code(T, giant=(lc >= 2) & (lc <= 5)), lc


def lum_class(phase, L, m):
    phase = np.asarray(phase)
    c = np.full(phase.shape, 0, dtype=np.int8)
    giant = (phase >= SG) & (phase <= AGB)
    gc = np.where(L >= 1e5, 5, np.where(L >= 2e3, 4, np.where(L >= 600, 3, 2)))
    gc = np.where(m <= 4.0, np.minimum(gc, 3), gc)          # 低质量巨星表面重力高，最多到 II
    gc = np.where((phase == SG) & (m <= M_MASSIVE), 1, gc)
    c[giant] = gc[giant]
    c[phase == WD] = 6
    return c


def spectral_string(code, lclass):
    out = []
    for cc, lc in zip(np.asarray(code), np.asarray(lclass)):
        if lc == 6:
            out.append("DA")
            continue
        li = int(min(cc // 10, 6))
        sub = cc - li * 10
        sub_s = f"{sub:.1f}".rstrip("0").rstrip(".")
        out.append(f"{_LETTERS[li]}{sub_s}{LUMCLASS[lc]}")
    return out
