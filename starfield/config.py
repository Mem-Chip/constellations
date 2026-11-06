"""读取配置并推导所有 auto 项。"""
import copy
import math
from pathlib import Path

import yaml


def load_config(path="starfield_config.yaml", mag_limit=None):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    derived = {}
    warnings = []

    # --- 望远镜极限星等与分辨率 ---
    obs = cfg["observing"]
    tel = obs["telescope"]
    ap = float(tel["aperture_mm"])
    q = float(tel["optical_quality"])
    if tel.get("limit_mag", "auto") == "auto":
        tel["limit_mag"] = round(2.1 + 5 * math.log10(ap) - 1.2 * (1 - q), 2)
    if tel.get("resolution_arcsec", "auto") == "auto":
        tel["resolution_arcsec"] = round(max(116.0 / ap / q, obs["atmosphere"]["seeing_arcsec"]), 2)

    # --- 生成深度 ---
    meta = cfg["meta"]
    if meta.get("generation_mag_limit", "auto") == "auto":
        meta["generation_mag_limit"] = round(tel["limit_mag"] + float(meta["generation_margin_mag"]), 2)
    if mag_limit is not None:
        meta["generation_mag_limit"] = float(mag_limit)

    # --- 宿的数量 ---
    sc = cfg["sky_clock"]
    moons = {m["name"]: m for m in sc["moons"]}
    lm = sc["lunar_mansions"]
    if lm.get("count", "auto") == "auto":
        lm["count"] = int(round(moons[lm["from_moon"]]["sidereal_period_days"]))

    # --- 由卫星估算岁差周期，与配置值比较 ---
    # 进动速率 ∝ (卫星项 0.69·M/a³ + 太阳项 0.31·(年长比)⁻²) · cosε · 自转角速度
    moon_term = sum(0.69 * m["mass_rel_earth_moon"] / m["semi_major_axis_rel"] ** 3 for m in sc["moons"])
    sun_term = 0.31 * (365.25 / sc["orbit"]["year_days"]) ** 2
    eps = sc["rotation"]["obliquity_deg"]
    rate = (moon_term + sun_term) * math.cos(math.radians(eps)) / math.cos(math.radians(23.44))
    rate *= 23.934 / sc["rotation"]["sidereal_day_hours"]
    p_est = 25772.0 / rate
    derived["precession_period_estimate_yr"] = round(p_est)
    p_cfg = sc["precession"]["period_yr"]
    if abs(p_est - p_cfg) / p_cfg > 0.15:
        warnings.append(f"岁差周期：配置 {p_cfg} 年，按卫星参数估算约 {p_est:.0f} 年，相差超过 15%")

    if "present_year" not in cfg["output"]["epochs"] or cfg["output"]["epochs"]["present_year"] is None:
        cfg["output"]["epochs"]["present_year"] = 0

    cfg["_derived"] = derived
    cfg["_warnings"] = warnings
    cfg["_path"] = str(path)
    return cfg
